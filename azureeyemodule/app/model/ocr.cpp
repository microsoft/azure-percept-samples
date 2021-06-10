// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <string>
#include <thread>
#include <vector>
#include <fstream>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "ocr.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/ocr_kernels.hpp"
#include "../ocr/decoder.hpp"
#include "../ocr/ocrvis.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/labels.hpp"
#include "../util/helper.hpp"

namespace model {

// Our Text Detection model takes in a frame and outputs two tensors. The recognition model takes in a single tensor and outputs another.
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(TextDetection, <GMat2(cv::GMat)>, "sample.custom.text_detect");
G_API_NET(TextRecognition, <cv::GMat(cv::GMat)>,"sample.custom.text_recogn");

OCRModel::OCRModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
        :AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, OCRDecoder(ocr::TextDecoder {0, "0123456789abcdefghijklmnopqrstuvwxyz#", '#'})
{
}

void OCRModel::run(cv::GStreamingCompiled* pipeline)
{
    while(true)
    {
        // Wait for the VPU to come up.
        this->wait_for_device();

        // Log our meta data.
        this->log_parameters();

        // Build the camera pipeline with G-API
        *pipeline = this->compile_cv_graph();
        util::log_info("starting the pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);
        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled OCRModel::compile_cv_graph() const
{
    // The input node of the G-API pipeline. This will be filled in, one frame at time.
    cv::GMat in;

    // We have a custom preprocessing node for the Myriad X-attached camera.
    cv::GMat preproc = cv::gapi::mx::preproc(in, this->resolution);

    // This path is the H.264 path. It gets our frames one at a time from
    // the camera and encodes them into H.264.
    cv::GArray<uint8_t> h264;
    cv::GOpaque<int64_t> h264_seqno;
    cv::GOpaque<int64_t> h264_ts;
    std::tie(h264, h264_seqno, h264_ts) = cv::gapi::streaming::encH264ts(preproc);

    // We branch off from the preproc node into H.264 (above), raw BGR output (here),
    // and neural network inferences (below).
    cv::GMat img = cv::gapi::copy(cv::gapi::streaming::desync(preproc));
    auto img_ts = cv::gapi::streaming::timestamp(img);

    // This node branches off from the preproc node for neural network inferencing.
    cv::GMat bgr = cv::gapi::streaming::desync(preproc);
    auto nn_ts = cv::gapi::streaming::timestamp(bgr);

    // Text recognition input size
    cv::Size in_rec_sz{ 120, 32 };

    // The first network (the text detector) outputs two tensors: link and sgm
    cv::GMat link, segm;
    std::tie(link, segm) = cv::gapi::infer<TextDetection>(bgr);

    // Here we post-process the outputs of the text detection network
    cv::GOpaque<cv::Size> size = cv::gapi::streaming::size(bgr);
    cv::GArray<cv::RotatedRect> rrs = cv::gapi::streaming::PostProcess::on(link, segm, size, 0.8f, 0.8f);
    cv::GArray<cv::GMat> td_labels = cv::gapi::streaming::CropLabels::on(bgr, rrs, in_rec_sz);

    // Now we feed the post-processed output into the text recognition RNN
    cv::GArray<cv::GMat> text = cv::gapi::infer2<TextRecognition>(bgr, td_labels);


    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,   // The H.264 branch of the graph
                                           img, img_ts,                 // The raw BGR frame branch
                                           nn_ts, rrs, text));          // The neural network branch

    // There are two output layers from the text detection network. Specify them here. Also pass in the model file for the first network.
    auto textdetection_net = cv::gapi::mx::Params<TextDetection> {modelfiles.at(0)}.cfgOutputLayers({"model/link_logits_/add", "model/segm_logits/add"});

    // Feed in the model file for the second network.
    auto textrecognition_net = cv::gapi::mx::Params<TextRecognition> {modelfiles.at(1)};

    // Wrap up the networks.
    auto networks = cv::gapi::networks(textdetection_net, textrecognition_net);

    // Wrap up the kernels.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::OCVPostProcess>(), cv::gapi::kernels<cv::gapi::streaming::OCVCropLabels>());

    // Compile the graph in streamnig mode, set all the parameters.
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool OCRModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    // The raw BGR frames from the camera will fill this node.
    cv::optional<cv::Mat> out_bgr;
    cv::optional<int64_t> out_bgr_ts;

    // The H.264 outputs will fill these nodes.
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    // The neural network branch outputs will fill these nodes.
    cv::optional<int64_t> out_nn_ts;
    cv::optional<std::vector<cv::RotatedRect>> out_txtrcs;
    cv::optional<std::vector<cv::Mat>> out_text;

    // We cache our latest results in these variables, since they are coming at different times.
    cv::Mat last_bgr;
    std::vector<cv::RotatedRect> last_rcs;
    std::vector<std::string> last_text;
    int64_t last_bgr_timestamp;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_bgr_ts, out_nn_ts, out_txtrcs, out_text)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_nn_ts, out_txtrcs, last_rcs, out_text, last_text);
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_bgr_timestamp, last_rcs, last_text);

        if (this->restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr, last_bgr_timestamp);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void OCRModel::handle_bgr_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &out_bgr_ts, cv::Mat &last_bgr, int64_t &last_bgr_timestamp,
                                 const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text)
{
    if (!out_bgr.has_value())
    {
        return;
    }

    // This was derived from the same branch in the G-API graph as out_bgr, so must also be present.
    CV_Assert(out_bgr_ts.has_value());

    // Now that we got a useful value, let's cache this one as the most recent.
    last_bgr = *out_bgr;
    last_bgr_timestamp = *out_bgr_ts;

    // Mark up this frame with our preview function.
    cv::Mat marked_up_bgr;
    last_bgr.copyTo(marked_up_bgr);
    this->preview(marked_up_bgr, last_rcs, last_text);

    // Stream the latest BGR frame.
    this->stream_frames(last_bgr, marked_up_bgr, *out_bgr_ts);

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(last_bgr);
}

void OCRModel::preview(cv::Mat &bgr, const std::vector<cv::RotatedRect> &last_rcs, const std::vector<std::string> &last_text) const
{
    const auto num_labels = last_rcs.size();

    for (size_t i=0; i < num_labels; i++)
    {
        //Draw bounding box for this rotated rectangle
        const auto &rc = last_rcs[i];
        ocr::vis::drawRotatedRect(bgr, rc);

        // Draw text, if decoded
        ocr::vis::drawText(bgr, rc, last_text[i]);
    }
}

void OCRModel::handle_inference_output(const cv::optional<int64_t> &out_nn_ts,
                                       const cv::optional<std::vector<cv::RotatedRect>> &out_txtrcs, std::vector<cv::RotatedRect> &last_rcs,
                                       cv::optional<std::vector<cv::Mat>> &out_text, std::vector<std::string> &last_text)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_text.has_value());
    CV_Assert(out_txtrcs->size() == out_text->size());

    // Hold output in temp vars before pruning
    auto temp_text = *out_text;
    auto temp_rcs = *out_txtrcs;

    // To hold all the results from current text and rectangles
    std::vector<std::string> curr_textresults;
    std::vector<cv::RotatedRect> curr_rcsresults;

    // Collect all texts and send to IoT Hub
    std::string msg = "{\"Texts\": [";

    const auto num_labels = temp_rcs.size();
    for (std::size_t label_idx = 0; label_idx < num_labels; label_idx++)
    {
        // Decode the recognized text in the rectangle
        auto decoded = this->OCRDecoder.decode(temp_text[label_idx]);
        this->log_inference("Text: \"" + decoded.text + "\"");
        if (decoded.conf > 0.2)
        {
            curr_textresults.push_back(decoded.text);
            curr_rcsresults.push_back(temp_rcs[label_idx]);

            msg.append("\"" + decoded.text + "\", ");
        }
        else
        {
            msg.append("\"<COULD NOT DECODE>\", ");
        }
    }

    // If there was at least one label, we need to remove the trailing space and comma
    // because JSON is stupid and can't handle trailing commas. :/
    if (num_labels > 0)
    {
        msg = msg.substr(0, msg.length() - 2);
    }

    msg.append("]}");

    // Send all result into last_text and then dump all curr text results
    if(curr_textresults.size() > 0)
    {
        last_text = std::move(curr_textresults);
        last_rcs = std::move(curr_rcsresults);
    }
    else
    {
        last_text = {};
        last_rcs = {};
    }

    // Send resulting message over IoT
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, msg);

    // If we want to time-align our network inferences with camera frames, we need to
    // do that here (now that we have a new inference to align in time with the frames we've been saving).
    // The super class will check for us and handle this appropriately.
    auto f_to_call_on_each_frame = [last_rcs, last_text, this](cv::Mat &frame){ this->preview(frame, last_rcs, last_text); };
    this->handle_new_inference_for_time_alignment(*out_nn_ts, f_to_call_on_each_frame);
}

void OCRModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: OpenPose";
    util::log_info(msg);
}

} // namespace model
