/**
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * OpenPose is a pose detection network, which has very good accuracy and good run time performance,
 * but has an extremely complicated post-processing algorithm. See https://github.com/CMU-Perceptual-Computing-Lab/openpose
 * for a reference implementation and https://arxiv.org/abs/1812.08008 for the paper.
 *
 * The gist of it is that the neural network outputs two tensors: a heatmap of keypoints (human keypoints - eyes, nose, arms, or whatever),
 * and a "part affinity field", which is a tensor that encodes a vector field where each vector is pointing along a "limb" (like an arm or leg,
 * but could be torso or face) from one keypoint to another.
 *
 * The post-processing algorithm takes these two tensors and combines their information to produce a skeletal pose.
 * Most of the details are hidden in the kernel file and the openpose folder.
 */
#pragma once

#include "azureeyemodel.hpp"
#include "../openpose/human_pose.hpp"


namespace model {

/**
 * A class for detecting skeletal pose of humans.
 */
class OpenPoseModel : public AzureEyeModel
{
public:
    /**
     * Constructor.
     *
     * @param modelfpaths: In this model, we expect this to be a vector of length 1 (all model constructors take a vector of paths, but most only use 1 model path).
     *                     Model paths are strings that point to the model .blob files. If you are developing a cascaded model (see ocr for example), you will
     *                     need to pass a model file for each network in the cascade.
     * @param mvcmd:       The Myriad X firmware binary. A running model owns the VPU, so it is responsible for initializing it as well.
     * @param videofile:   If we pass a non-empty string here, the model will record frames to a file at this location as a .mp4.
     * @param resolution:  The resolution mode to put the camera in.
     */
    OpenPoseModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    /** We invoke this method from main to run the network. It should return if there is a model update (found by checking this->restarting). */
    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Compiles the G-API graph. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Pulls data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /**
     * The G-API graph in this (and most classes) is split into three branches: a branch that handles the H.264 encoding, a branch that
     * timestamps and forwards the raw camera BGR frames, and a branch that handles the neural network inferences.
     *
     * This method handles the outputs from the second branch - the raw BGR branch. It takes the latest BGR frame, updates `last_bgr` to it,
     * and streams the frame out over the RTSP stream.
     *
     * Note that there is an option in the module twin to turn on time alignment. If we do NOT have that option turned on,
     * we also stream the result frame here after applying `poses` to it.
     * If we DO have that option turned on, we ignore the marked up frame created by last_mask and instead store
     * out_bgr (the raw BGR frame) into a buffer for use later, when we have a new neural network inference. At that point,
     * we search through the buffer of old frames that we have to find the best matching one in time, and then mark up
     * that frame and all older ones that we have not yet sent to the RTSP server. Then we release all of the marked up frames
     * in a batch to the RTSP stream (which will churn through them at some specified frames per second rate).
     *
     * All of this time alignment stuff is mostly taken care of for you in the super class.
     *
     * @param out_bgr: The latest raw BGR frame from the G-API graph. This may be empty (because the graph is asynchronous), in
     *                 which case we immediately return from this method.
     * @param bgr_ts: The timestamp of out_bgr.
     * @param last_bgr: We cache out_bgr into this for use in other methods.
     * @param last_bgr_timestamp: We cache bgr_ts in this value.
     * @param poses: The latest poses to use if we are not time aligning the result frames.
     */
    void handle_bgr_output(const cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &bgr_ts, cv::Mat &last_bgr, int64_t &last_bgr_timestamp, const std::vector<pose::HumanPose> &poses);

    /** Compose the RGB based on the poses. */
    void preview(cv::Mat &bgr, const std::vector<pose::HumanPose> &poses) const;

    /**
     * The G-API graph in this (and most classes) is split into three branches: a branch that handles the H.264 encoding, a branch that
     * timestamps and forwards the raw camera BGR frames, and a branch that handles the neural network inferences.
     *
     * This method handles the third branch - the neural network output branch. Whenever we get a new neural network output,
     * we handle it by doing two things:
     *
     * 1. We compose a JSON message that says what percentage of the image is occupied by the detected object(s).
     *    Then we send this message out over IoT.
     * 2. If we are time-aligning frames (see handle_bgr_output's documentation), we alert the super class that we have
     *    a new inference. The super class will go search through the stored frames for the one most closely aligned in
     *    time with the frame that this network was working on for this inference, and it marks up all the frames
     *    from that point backwards, and then releases them to the RTSP server.
     *
     * @param out_nn_ts: The timestamp for the frame the network used to derive the poses.
     * @param out_nn_seqno: The frame number for the frame the network used to derive the poses.
     * @param out_poses: The latest poses from the neural network. If this is empty, we return immediately.
     * @param last_poses: We cache the latest poses into this for use in other methods.
     */
    void handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                 const cv::optional<std::vector<pose::HumanPose>> &out_poses, std::vector<pose::HumanPose> &last_poses);

    /** Print out all the model's meta information. */
    void log_parameters() const;
};

} // namespace model
