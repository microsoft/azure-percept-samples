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
#include "binaryunet.hpp"
#include "../device/device.hpp"
#include "../util/helper.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/binaryunet_kernels.hpp"
#include "../streaming/rtsp.hpp"

namespace model {

/** This network takes in a single frame at a time and outputs a frame (a segmentation mask). */
G_API_NET(UNetNetwork, <cv::GMat(cv::GMat)>, "unet-network");

BinaryUnetModel::BinaryUnetModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel( modelfpaths, mvcmd, videofile, resolution )
{
    // Nothing to do. Everything's done via the parent class's constructor.
}

void BinaryUnetModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        // We must wait for the Myriad X VPU to come up.
        this->wait_for_device();

        // Now let's log our model's parameters.
        this->log_parameters();

        // Build the camera pipeline with G-API and start it.
        *pipeline = this->compile_cv_graph();
        util::log_info("starting segmentation pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);
        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled BinaryUnetModel::compile_cv_graph() const
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

    // Here's where we actually run our neural network. It runs on the VPU.
    cv::GMat segmentation = cv::gapi::infer<UNetNetwork>(bgr);

    // Here's where we post-process our network's outputs into a segmentation mask.
    cv::GMat mask = cv::gapi::streaming::PostProcBinaryUnet::on(segmentation);

    // Specify the boundaries of the G-API graph (the inputs and outputs).
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,      // H.264 path
                                           img, img_ts,                    // Raw BGR frames path
                                           mask, nn_ts));                  // Neural network inference path

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of length at least 1.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<UNetNetwork>{this->modelfiles.at(0)});

    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVPostProcBinaryUnet>());

    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Percept DK's camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    util::log_info("Succesfully compiled segmentation pipeline");
    return pipeline;
}

bool BinaryUnetModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    // The raw BGR frames from the camera will fill this node.
    cv::optional<cv::Mat> out_bgr;
    cv::optional<int64_t> out_bgr_ts;

    // The H.264 information will fill these nodes.
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    // Our neural network's frames will fill out_mask, and timestamp will fill nn_ts.
    cv::optional<cv::Mat> out_mask;
    cv::optional<int64_t> out_nn_ts;

    // Because each node is asynchronusly filled, we cache them whenever we get them.
    cv::Mat last_bgr;
    cv::Mat last_mask;
    int64_t last_bgr_timestamp;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    util::log_info("Pull_data: prep to pull");

    // Pull the data from the pipeline while it is running.
    // Every time we call pull(), G-API gives us whatever nodes it has ready.
    // So we have to make sure a node has useful contents before using it.
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_bgr_ts, out_mask, out_nn_ts)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_mask, out_nn_ts, last_mask);
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_bgr_timestamp, last_mask);

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

void BinaryUnetModel::handle_bgr_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &bgr_ts, cv::Mat &last_bgr, int64_t &last_bgr_timestamp, const cv::Mat &last_mask)
{
    // If out_bgr does not have anything in it, we didn't get anything from the G-API graph at this iteration.
    if (!out_bgr.has_value())
    {
        return;
    }

    // This comes with out_bgr on the same branch of the G-API graph, so it must have a value if out_bgr does.
    CV_Assert(bgr_ts.has_value());

    // Now that we got a useful value, let's cache this one as the most recent.
    last_bgr = *out_bgr;
    last_bgr_timestamp = *bgr_ts;

    // Mark up this frame with our preview function.
    cv::Mat marked_up_bgr;
    last_bgr.copyTo(marked_up_bgr);
    this->preview(marked_up_bgr, last_mask);

    // Stream the latest BGR frame.
    this->stream_frames(last_bgr, marked_up_bgr, bgr_ts);

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(last_bgr);
}

void BinaryUnetModel::preview(cv::Mat &frame, const cv::Mat& last_mask) const
{
    // If we haven't gotten a neural network inference yet, we can't preview.
    if (last_mask.empty())
    {
        return;
    }

    // create a BGR mask
    cv::Mat g_prob = last_mask > 0;
    cv::Mat g;
    g_prob.convertTo(g, CV_8U);

    cv::Mat r = (1 - g);
    cv::Mat b(r.size(), CV_8UC1, cv::Scalar(0));

    std::vector<cv::Mat> channels{ b, g, r };
    cv::Mat show_mask;
    cv::merge(channels, show_mask);
    show_mask *= 255;

    cv::resize(show_mask, show_mask, frame.size());

    // These are blending coefficients for adding the mask
    // to the image: alpha * image + beta * mask
    float alpha = 0.8;
    float beta = 1 - alpha;

    cv::Mat dst;
    cv::addWeighted(frame, alpha, show_mask, beta, 0.0, dst);
    dst.copyTo(frame);
}

void BinaryUnetModel::handle_inference_output(const cv::optional<cv::Mat> &out_mask, const cv::optional<int64_t> &inference_ts, cv::Mat &last_mask, float threshold)
{
    if (!out_mask.has_value())
    {
        return;
    }

    // This comes from the same branch in the G-API graph as out_mask, and so must have a value if out_mask does.
    CV_Assert(inference_ts.has_value());

    // Create a mask - everywhere that the network has confidence greater than threshold
    cv::Mat mask_vals(*out_mask > threshold);

    // Compute the fraction of the image that is occupied by detections
    float relative_occupied = static_cast<float>(cv::countNonZero(mask_vals)) / (mask_vals.rows * mask_vals.cols);

    // Create JSON message of the following schema
    //
    // [
    //   {
    //     "occupied": <float> fraction of the image covered by detections
    //   }
    // ]
    //
    // The outer list is not strictly necessary, but because most (all?) of the other
    // networks have multiple detections per output, they all use lists, so this one
    // uses a list just to conform.
    std::string str = std::string("[");
    str.append("{")
        .append("\"occupied\": \"").append(std::to_string(relative_occupied)).append("\"")
        .append("}");
    str.append("]");

    // Log this network inference message using adaptive logging, so that we decay its frequency over time
    // (So we don't end up overwhelming the log files)
    this->log_inference(str);

    // Send this inference message over Azure IoT.
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);

    // Now that we have a new inference, let's cache it.
    last_mask = *out_mask;

    // If we want to time-align our network inferences with camera frames, we need to
    // do that here (now that we have a new inference to align in time with the frames we've been saving).
    // The super class will check for us and handle this appropriately.
    auto f_to_call_on_each_frame = [last_mask, this](cv::Mat &frame){ this->preview(frame, last_mask); };
    this->handle_new_inference_for_time_alignment(inference_ts, f_to_call_on_each_frame);
}

void BinaryUnetModel::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += "firmware: " + this->mvcmd;
    util::log_info(msg);
}

} // namespace model
