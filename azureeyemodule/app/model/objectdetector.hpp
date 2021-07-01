// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "azureeyemodel.hpp"


namespace model {

/**
 * An abstract class that derives from the AzureEyeModel base class
 * and which represents an Object Detector, such as SSD or YOLO.
 */
class ObjectDetector : public AzureEyeModel
{
public:
    ObjectDetector(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);
    virtual ~ObjectDetector();
    const std::string VIDEO_PREFIX = "video:";

protected:
    /** Path to the label file. */
    std::string labelfpath;

    /** Labels for the things we classify. */
    std::vector<std::string> class_labels;

    /**
     * The G-API graph in the object detector subclasses is split into three branches: a branch that handles the H.264 encoding, a branch that
     * timestamps and forwards the raw camera BGR frames, and a branch that handles the neural network inferences.
     *
     * This method handles the outputs from the second branch - the raw BGR branch. It takes the latest BGR frame, updates `last_bgr` to it,
     * and streams the frame out over the RTSP stream.
     *
     * Note that there is an option in the module twin to turn on time alignment. If we do NOT have that option turned on,
     * we also stream the result frame here after applying last_labels and last_confidences to it.
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
     * @param out_bgr_ts: The timestamp of out_bgr.
     * @param last_bgr: We cache out_bgr into this for use in other methods.
     * @param last_boxes: The latest neural network output bounding box(es) to use in marking up the RTSP stream if we are not time aligning.
     * @param last_labels: The latest neural network output label(s) to use in marking up the RTSP stream if we are not time aligning.
     * @param last_confidences: The latest neural network output confidence(s) to use in marking up the RTSP stream if we are not time aligning.
     */
    virtual void handle_bgr_output(cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &out_bgr_ts, cv::Mat &last_bgr, const std::vector<cv::Rect> &last_boxes, const std::vector<int> &last_labels, const std::vector<float> &last_confidences);

    /**
     * The G-API graph in the object detection subclasses is split into three branches: a branch that handles the H.264 encoding, a branch that
     * timestamps and forwards the raw camera BGR frames, and a branch that handles the neural network inferences.
     *
     * This method handles the third branch - the neural network output branch. Whenever we get a new neural network output,
     * we handle it by doing two things:
     *
     * 1. We compose a JSON message that says what objects are detected, where, and with what confidences.
     *    Then we send this message out over IoT.
     * 2. If we are time-aligning frames (see handle_bgr_output's documentation), we alert the super class that we have
     *    a new inference. The super class will go search through the stored frames for the one most closely aligned in
     *    time with the frame that this network was working on for this inference, and it marks up all the frames
     *    from that point backwards, and then releases them to the RTSP server.
     *
     * @param out_nn_ts: The timestamp associated with out_boxes, out_labels, and out_confidences.
     * @param out_nn_seqno: The frame number of the frame that the network was working on to produce this output.
     * @param out_boxes: The latest neural network output bounding boxes.
     * @param out_labels: The latest neural network output labels.
     * @param out_confidences: The latest neural network output confidences.
     * @param out_size: The size of the RGB frame. We need this for converting the coordinates from normalized to absolute.
     * @param last_boxes: We cache the bounding boxes into this for use in other methods.
     * @param last_labels: We cache the labels into this for use in other methods.
     * @param last_confidences: We cache the confidences into this for use in other methods.
     */
    virtual void handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                         const cv::optional<std::vector<cv::Rect>> &out_boxes, const cv::optional<std::vector<int>> &out_labels,
                                         const cv::optional<std::vector<float>> &out_confidences, const cv::optional<cv::Size> &out_size,
                                         std::vector<cv::Rect> &last_boxes, std::vector<int> &last_labels, std::vector<float> &last_confidences);

    /** Pull data through the given pipeline. Returns true if we run out of frames, false if we have been interrupted. Otherwise runs forever. */
    virtual bool pull_data(cv::GStreamingCompiled &pipeline);

    virtual bool pull_data_uvc_video(cv::GStreamingCompiled &pipeline);

private:
    /** Marks up the given rgb with the given labels, bounding boxes, and confidences. */
    void preview(cv::Mat &rgb, const std::vector<cv::Rect> &boxes, const std::vector<int> &labels, const std::vector<float> &confidences) const;
};

} // namespace model
