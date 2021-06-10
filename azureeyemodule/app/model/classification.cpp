// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <functional>
#include <thread>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "classification.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/classification_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"


namespace model {

/** A classification network takes a single input and outputs a single output (which we will parse into labels and confidences) */
G_API_NET(ClassificationNetwork, <cv::GMat(cv::GMat)>, "classification-network");

ClassificationModel::ClassificationModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, labelfpath(labelfpath), class_labels({})
{
}

void ClassificationModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        // We must wait for the Myriad X VPU to come up.
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);

        // Now let's log our model's parameters.
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

cv::GStreamingCompiled ClassificationModel::compile_cv_graph() const
{
    // The input node of the G-API pipeline. This will be filled in, one frame at a time.
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
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(bgr);

    // Here's where we actually run the neural network. It runs on the VPU.
    cv::GMat nn = cv::gapi::infer<ClassificationNetwork>(bgr);

    // Grab some useful information: the frame number, the timestamp, and the size of the frame.
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    // Parse the output of the classification network into class IDs and confidence scores.
    cv::GArray<int> ids;
    cv::GArray<float> cfs;
    std::tie(ids, cfs) = cv::gapi::streaming::parse_class(nn);

    // Specify the boundaries of the G-API graph (the inputs and outputs).
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,    // H.264 branch
                                           img, img_ts,                  // Raw BGR frames branch
                                           nn_seqno, nn_ts, ids, cfs));  // Neural network inferences path

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of length at least 1.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<ClassificationNetwork>{this->modelfiles.at(0)});

    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseClass>());

    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool ClassificationModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    // The raw BGR frames from the camera will fill this node.
    cv::optional<cv::Mat> out_bgr;
    cv::optional<int64_t> out_bgr_ts;

    // The H.264 information will fill these nodes.
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    // Our neural network's outputs will fill these nodes.
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<int>> out_labels;
    cv::optional<std::vector<float>> out_confidences;

    // Because each node is asynchronusly filled, we cache them whenever we get them.
    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    cv::Mat last_bgr;
    int64_t last_bgr_timestamp;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running.
    // Every time we call pull(), G-API gives us whatever nodes it has ready.
    // So we have to make sure a node has useful contents before using it.
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_bgr_ts, out_nn_seqno, out_nn_ts, out_labels, out_confidences)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_nn_ts, out_nn_seqno, out_labels, out_confidences, last_labels, last_confidences);
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_bgr_timestamp, last_labels, last_confidences);

        if (restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr, last_bgr_timestamp);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void ClassificationModel::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &bgr_ts, cv::Mat &last_bgr, int64_t &last_bgr_timestamp,
                                            const std::vector<int> &last_labels, const std::vector<float> &last_confidences)
{
    if (!out_bgr.has_value())
    {
        return;
    }

    // This is derived on the same branch of the G-API graph as out_bgr, so should also have value.
    CV_Assert(bgr_ts.has_value());

    // Cache this most recent frame.
    last_bgr = *out_bgr;
    last_bgr_timestamp = *bgr_ts;

    // Mark up this frame with our preview function.
    cv::Mat marked_up_bgr;
    last_bgr.copyTo(marked_up_bgr);
    this->preview(marked_up_bgr, last_labels, last_confidences);

    // Stream the latest BGR frame (or cache it for later if we are time-aligning).
    this->stream_frames(last_bgr, marked_up_bgr, bgr_ts);

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(last_bgr, last_confidences);
}

void ClassificationModel::preview(const cv::Mat &rgb, const std::vector<int> &labels, const std::vector<float> &confidences) const
{
    const auto threshold_confidence = 0.5f;
    auto largest_confidence = 0.0f;
    int best_label = 0;

    // Go through each class and see if we have found anything larger than threshold,
    // and keep the largest-confidence detection if so.
    for (std::size_t i = 0; i < labels.size(); i++)
    {
        if (confidences.at(i) > largest_confidence)
        {
            largest_confidence = confidences.at(i);
            best_label = labels.at(i);
        }
    }

    // Get a new color for each item we detect,
    // though wrap around if there are a whole bunch of items.
    auto color_index = (largest_confidence >= threshold_confidence) ? best_label % label::colors().size() : 0;

    // Draw the label. Use the same color. If we can't figure out the label
    // (because the network output something unexpected, or there is no labels file),
    // we just use the class index.
    std::string label;
    if (largest_confidence >= threshold_confidence)
    {
        label = util::get_label(best_label, this->class_labels) + ": " + util::to_string_with_precision(largest_confidence, 2);
    }
    else
    {
        label = "No Detections";
    }

    auto origin = cv::Point(0, 20) + cv::Point(3, 20);
    auto font = cv::FONT_HERSHEY_SIMPLEX;
    auto fontscale = 0.7;
    auto color = cv::Scalar(label::colors().at(color_index));
    auto thickness = 2;
    cv::putText(rgb, label, origin, font, fontscale, color, thickness);
}

void ClassificationModel::handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                                  const cv::optional<std::vector<int>> &out_labels,
                                                  const cv::optional<std::vector<float>> &out_confidences, std::vector<int> &last_labels,
                                                  std::vector<float> &last_confidences)

{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // All of these items are derived together on the same branch of the G-API graph, so they
    // should all arrive together.
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_labels.has_value());
    CV_Assert(out_confidences.has_value());

    // Compose a single message for each detection, which follows this schema:
    //
    // {
    //      "label": string. Class label of the detected object,
    //      "confidence": float. Confidence of the detection,
    //      "timestamp": int. Timestamp of the detection.
    // }
    //
    // Push each of these detection messages into a vector.
    std::vector<std::string> messages;
    for (std::size_t i = 0; i < out_labels->size(); i++)
    {
        auto label = util::get_label(out_labels.value()[i], this->class_labels);
        auto confidence = std::to_string(out_confidences.value()[i]);
        auto timestamp = std::to_string(*out_nn_ts);

        std::string str = std::string("{");
        str.append("\"label\": \"").append(label).append("\", ")
           .append("\"confidence\": \"").append(confidence).append("\", ")
           .append("\"timestamp\": \"").append(timestamp).append("\"")
           .append("}");

        messages.push_back(str);
    }

    // Wrap the detection messages into a list. The send_message() function will wrap it back into curly braces for you.
    std::string str = std::string("[");
    for (size_t i = 0; i < messages.size(); i++)
    {
        if (i > 0)
        {
            str.append(", ");
        }
        str.append(messages[i]);
    }
    str.append("]");

    // Send the message over IoT
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);

    // Log using decaying filter so that the frequency of log messages goes down over time. This is to keep
    // the logs from getting out of hand on the device.
    this->log_inference(str);

    // Update the cached labels and confidences now that we have new ones.
    last_labels = *out_labels;
    last_confidences = *out_confidences;

    // If we want to time-align our network inferences with camera frames, we need to
    // do that here (now that we have a new inference to align in time with the frames we've been saving).
    // The super class will check for us and handle this appropriately.
    auto f_to_call_on_each_frame = [last_labels, last_confidences, this](cv::Mat &frame){ this->preview(frame, last_labels, last_confidences); };
    this->handle_new_inference_for_time_alignment(*out_nn_ts, f_to_call_on_each_frame);
}

void ClassificationModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: classification, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model