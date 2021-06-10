// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <assert.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "openpose.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/openpose_kernels.hpp"
#include "../openpose/human_pose.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace model {

/** Colors we use for OpenPose viewing */
static const cv::Scalar colors[] = {
    cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
    cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
    cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
    cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
};

/** Keypoint pairs that make up 'limbs' */
static const std::pair<int, int> limb_keypoints_ids[] = {
    {1, 2},  {1, 5},   {2, 3},
    {3, 4},  {5, 6},   {6, 7},
    {1, 8},  {8, 9},   {9, 10},
    {1, 11}, {11, 12}, {12, 13},
    {1, 0},  {0, 14},  {14, 16},
    {0, 15}, {15, 17}
};

/** Width of the 'limbs' in viewing stuff */
const int stick_width = 4;

/** Declare an OpenPose network type. Takes one matrix and outputs two matrices (part affinity fields and keypoints). */
using openpose_output = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(OpenPoseNetwork, <openpose_output(cv::GMat)>, "com.intel.azure.open-pose");

OpenPoseModel::OpenPoseModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }
{
}

void OpenPoseModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        // We have to wait for the VPU to come up.
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

cv::GStreamingCompiled OpenPoseModel::compile_cv_graph() const
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
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(bgr);

    // Here's where we actually run our neural network. It runs on the VPU.
    cv::GMat nn_pafs;
    cv::GMat nn_keys;
    std::tie(nn_pafs, nn_keys) = cv::gapi::infer<OpenPoseNetwork>(bgr);

    // Grab some useful metadata
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn_keys);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    // Here's where we post-process our network's outputs into a vector of HumanPose objects.
    cv::GArray<pose::HumanPose> skeletons = cv::gapi::streaming::parse_open_pose(nn_pafs, nn_keys, sz);

    // Specify the boundaries of the G-API graph (the inputs and outputs).
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,      // The H.264 branch
                                           img, img_ts,                    // The raw camera frame branch
                                           nn_seqno, nn_ts, skeletons));   // The neural network inference path

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of length at least 1.
    // We also configure our network to have two output layers.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<OpenPoseNetwork>{ this->modelfiles.at(0) }.cfgOutputLayers({"Mconv7_stage2_L1", "Mconv7_stage2_L2"}));

    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseOpenPose>());

    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool OpenPoseModel::pull_data(cv::GStreamingCompiled &pipeline)
{
    // The raw BGR frames from the camera will fill this node.
    cv::optional<cv::Mat> out_bgr;
    cv::optional<int64_t> out_bgr_ts;

    // The H.264 information will fill these nodes.
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    // Our neural network's frames will fill out_mask, and timestamp will fill nn_ts.
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<pose::HumanPose>> out_poses;

    // Because each node is asynchronusly filled, we cache them whenever we get them.
    std::vector<pose::HumanPose> last_poses;
    cv::Mat last_bgr;
    int64_t last_bgr_timestamp;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_bgr_ts, out_nn_seqno, out_nn_ts, out_poses)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_nn_ts, out_nn_seqno, out_poses, last_poses);
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_bgr_timestamp, last_poses);

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

void OpenPoseModel::handle_bgr_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &bgr_ts,
                                      cv::Mat &last_bgr, int64_t &last_bgr_timestamp, const std::vector<pose::HumanPose> &last_poses)
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

    // Mark up this frame with our preview function.
    cv::Mat marked_up_bgr;
    last_bgr.copyTo(marked_up_bgr);
    this->preview(marked_up_bgr, last_poses);

    // Stream the latest BGR frame.
    this->stream_frames(last_bgr, marked_up_bgr, *bgr_ts);

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(last_bgr);
}

void OpenPoseModel::preview(cv::Mat &bgr, const std::vector<pose::HumanPose> &poses) const
{
    CV_Assert(bgr.type() == CV_8UC3);

    const cv::Point2f absent_keypoint(-1.0f, -1.0f);

    // For each pose
    for (const auto &pose : poses)
    {
        assert(pose.keypoints.size() == util::array_size(colors));

        // For each keypoint
        for (size_t keypoint_idx = 0; keypoint_idx < pose.keypoints.size(); keypoint_idx++)
        {
            // If the keypoint is not absent
            if (pose.keypoints[keypoint_idx] != absent_keypoint)
            {
                // Draw a circle on it
                cv::circle(bgr, pose.keypoints[keypoint_idx], 4, colors[keypoint_idx], -1);
            }
        }
    }

    cv::Mat pane = bgr.clone();
    // For each pose
    for (const auto &pose : poses)
    {
        // For each limb
        for (const auto &limb_id : limb_keypoints_ids)
        {
            // Don't draw the limb if either of its joint keypoints is missing
            std::pair<cv::Point2f, cv::Point2f> limb_keypoints(pose.keypoints[limb_id.first], pose.keypoints[limb_id.second]);
            if ((limb_keypoints.first == absent_keypoint) || (limb_keypoints.second == absent_keypoint))
            {
                continue;
            }

            float meanX = (limb_keypoints.first.x + limb_keypoints.second.x) / 2;
            float meanY = (limb_keypoints.first.y + limb_keypoints.second.y) / 2;
            cv::Point difference = limb_keypoints.first - limb_keypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stick_width), angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[limb_id.second]);
        }
    }
    cv::addWeighted(bgr, 0.4, pane, 0.6, 0, bgr);
}

void OpenPoseModel::handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                            const cv::optional<std::vector<pose::HumanPose>> &out_poses, std::vector<pose::HumanPose> &last_poses)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // These are all derived on the same branch of the G-API graph, so should come at the same time.
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_poses.has_value());

    bool at_least_one_pose = false;
    std::string msg = "{\"Poses\": [";
    for (const auto &pose : out_poses.value())
    {
        msg.append("{").append(pose.to_string()).append("}, ");
        at_least_one_pose = true;
    }

    // If there is at least one pose, we need to remove the trailing space and comma
    if (at_least_one_pose)
    {
        msg = msg.substr(0, msg.length() - 2);
    }
    msg.append("]}");

    last_poses = std::move(*out_poses);

    // Log using a decaying filter so we don't bloat the log files.
    this->log_inference(msg);

    // Send over IoT
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, msg);

    // If we want to time-align our network inferences with camera frames, we need to
    // do that here (now that we have a new inference to align in time with the frames we've been saving).
    // The super class will check for us and handle this appropriately.
    auto f_to_call_on_each_frame = [last_poses, this](cv::Mat &frame){ this->preview(frame, last_poses); };
    this->handle_new_inference_for_time_alignment(*out_nn_ts, f_to_call_on_each_frame);
}

void OpenPoseModel::log_parameters() const
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
