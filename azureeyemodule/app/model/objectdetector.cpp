// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "objectdetector.hpp"
#include "../iot/iot_interface.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace model {

ObjectDetector::ObjectDetector(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, labelfpath(labelfpath), class_labels({})
{
}

ObjectDetector::~ObjectDetector()
{
}

bool ObjectDetector::pull_data(cv::GStreamingCompiled &pipeline)
{
    // All of our object detector models have the same graph outputs.

    // These are the nodes from the raw camera frame path.
    cv::optional<cv::Mat> out_bgr;
    cv::optional<int64_t> out_bgr_ts;

    // These are the nodes from the H.264 path.
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;

    // These are the nodes from the neural network inference path.
    cv::optional<cv::Mat> out_nn;
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;
    cv::optional<std::vector<cv::Rect>> out_boxes;
    cv::optional<std::vector<int>> out_labels;
    cv::optional<std::vector<float>> out_confidences;
    cv::optional<cv::Size> out_size;

    // These are the values that we cache (since the graph is asynchronous).
    std::vector<cv::Rect> last_boxes;
    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    cv::Mat last_bgr;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    // Every time we call pull(), G-API gives us whatever nodes it has ready.
    // So we have to make sure a node has useful contents before using it.
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_bgr, out_bgr_ts, out_nn_seqno, out_nn_ts, out_boxes, out_labels, out_confidences, out_size)))
    {
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);
        this->handle_inference_output(out_nn_ts, out_nn_seqno, out_boxes, out_labels, out_confidences, out_size, last_boxes, last_labels, last_confidences);
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_boxes, last_labels, last_confidences);

        if (this->restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

bool ObjectDetector::pull_data_uvc_video(cv::GStreamingCompiled &pipeline)
{
    // All of our object detector models have the same graph outputs.
    // These are the nodes from the raw camera frame path.
    cv::optional<cv::Mat> out_bgr;

    // These are the nodes from the neural network inference path.
    cv::optional<std::vector<cv::Rect>> out_boxes;
    cv::optional<std::vector<int>> out_labels;
    cv::optional<std::vector<float>> out_confidences;
    cv::optional<cv::Size> out_size;
    cv::optional<int64_t> out_seqno;
    cv::optional<int64_t> out_ts;

    std::vector<cv::Rect> last_boxes;
    std::vector<int> last_labels;
    std::vector<float> last_confidences;
    cv::Mat last_bgr;

    // If the user wants to record a video, we open the video file.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    // Every time we call pull(), G-API gives us whatever nodes it has ready.
    // So we have to make sure a node has useful contents before using it.
    auto pipeline_outputs = cv::gout(out_boxes, out_labels, out_confidences, out_size, out_seqno, out_ts);

    pipeline_outputs += cv::gout(out_bgr);
    while (pipeline.pull(std::move(pipeline_outputs)))
    {
        //Generic UVC camera doesn't provide H264 stream so we don't handle h264 output here
        this->handle_inference_output(out_ts, out_seqno, out_boxes, out_labels, out_confidences, out_size, last_boxes, last_labels, last_confidences);
        //Intel doesn't provide frame timestamp for UVC camera input so we use the timestamp of inference results object here, 
        //which is identical to the frame timestamp
        this->handle_bgr_output(out_bgr, out_ts, last_bgr, last_boxes, last_labels, last_confidences);

        if (this->restarting)
        {
            // We've been interrupted
            this->cleanup(pipeline, last_bgr);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void ObjectDetector::handle_inference_output(const cv::optional<int64_t> &out_nn_ts, const cv::optional<int64_t> &out_nn_seqno,
                                             const cv::optional<std::vector<cv::Rect>> &out_boxes, const cv::optional<std::vector<int>> &out_labels,
                                             const cv::optional<std::vector<float>> &out_confidences, const cv::optional<cv::Size> &out_size, std::vector<cv::Rect> &last_boxes, std::vector<int> &last_labels,
                                             std::vector<float> &last_confidences)
{
    if (!out_nn_ts.has_value())
    {
        return;
    }

    // These nodes are all on the same branch of the G-API graph, so they should all be arriving together.
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_boxes.has_value());
    CV_Assert(out_labels.has_value());
    CV_Assert(out_confidences.has_value());
    CV_Assert(out_size.has_value());

    // Compose a message for each item we detected
    // Each object adheres to the following schema
    //
    // {
    //      "bbox": list of the form [float, float, float, float]. This is an object's bounding box (x0, y0, x1, y1),
    //      "label": string. Class label of the detected object,
    //      "confidence": float. Confidence of the network,
    //      "timestamp": int. Timestamp for this detection.
    // }
    std::vector<std::string> messages;
    for (std::size_t i = 0; i < out_labels->size(); i++)
    {
        // Bounding box is in (x, y, w, h), normalized coordinates.
        cv::Rect rect = out_boxes.value()[i];

        // Convert to (x, y, w, h) absolute pixel coordinates.
        cv::Rect2f rect_abs(static_cast<float>(rect.x) / out_size->width, static_cast<float>(rect.y) / out_size->height, static_cast<float>(rect.width) / out_size->width, static_cast<float>(rect.height) / out_size->height);

        // Convert bounding box to string of form (x0, y0, x1, y1) coordinates.
        std::stringstream bboxstr;
        auto x0 = rect_abs.x;
        auto y0 = rect_abs.y;
        auto x1 = rect_abs.x + rect_abs.width;
        auto y1 = rect_abs.y + rect_abs.height;
        bboxstr << std::fixed << std::setprecision(3) << "\"bbox\": [" << x0 << ", " << y0 << ", " << x1 << ", " << y1 << "]";

        // Get the label
        auto label = util::get_label(out_labels.value()[i], this->class_labels);
        auto confidence = std::to_string(out_confidences.value()[i]);
        auto timestamp = std::to_string(*out_nn_ts);

        std::string str = std::string("{");
        str.append(bboxstr.str()).append(",")
           .append("\"label\": \"").append(label).append("\", ")
           .append("\"confidence\": \"").append(confidence).append("\", ")
           .append("\"timestamp\": \"").append(timestamp).append("\"")
           .append("}");

        messages.push_back(str);
    }

    // Compose a single string out of all the detection messages
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

    // Log the neural network inference using a special decaying filter so we don't bloat our log files.
    this->log_inference("nn: seqno=" + std::to_string(*out_nn_seqno) + ", ts=" + std::to_string(*out_nn_ts) + ", " + str);

    // Send out the detection message to anyone who's listening (this will add curly braces around the inference message)
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, str);

    // Update our cache of items now that we have new ones
    last_boxes = std::move(*out_boxes);
    last_labels = std::move(*out_labels);
    last_confidences = std::move(*out_confidences);

    // If we want to time-align our network inferences with camera frames, we need to
    // do that here (now that we have a new inference to align in time with the frames we've been saving).
    // The super class will check for us and handle this appropriately.
    #ifdef DEBUG_TIME_ALIGNMENT
        util::log_debug("Sending a new inference to time algo.");
    #endif
    auto f_to_call_on_each_frame = [last_boxes, last_labels, last_confidences, this](cv::Mat &frame){ this->preview(frame, last_boxes, last_labels, last_confidences); };
    this->handle_new_inference_for_time_alignment(*out_nn_ts, f_to_call_on_each_frame);
}

void ObjectDetector::preview(cv::Mat &rgb, const std::vector<cv::Rect> &boxes, const std::vector<int> &labels, const std::vector<float> &confidences) const
{
    // This method is responsible for marking up the raw BGR frames with the inferences from the
    // neural network. Since all of our object detector networks output bounding boxes, labels, and confidences,
    // let's mark up the frames with those items.
    for (std::size_t i = 0; i < boxes.size(); i++)
    {
        // Draw a bounding box around the detected object. Use a new color each time
        // up to some point, at which point we wrap around and start reusing colors.
        int color_index = labels[i] % label::colors().size();
        cv::rectangle(rgb, boxes[i], label::colors().at(color_index), 2);

        // Draw the label. Use the same color. If we can't figure out the label
        // (because the network output something unexpected, or there is no labels file),
        // we just use the class index.
        auto label = util::get_label(labels[i], this->class_labels) + ": " + util::to_string_with_precision(confidences[i], 2);
        auto origin = boxes[i].tl() + cv::Point(3, 20);
        auto font = cv::FONT_HERSHEY_SIMPLEX;
        auto fontscale = 0.7;
        auto color = cv::Scalar(label::colors().at(color_index));
        auto thickness = 2;
        cv::putText(rgb, label, origin, font, fontscale, color, thickness);
    }
}

void ObjectDetector::handle_bgr_output(cv::optional<cv::Mat> &out_bgr, const cv::optional<int64_t> &out_bgr_ts, cv::Mat &last_bgr, const std::vector<cv::Rect> &last_boxes,
                                       const std::vector<int> &last_labels, const std::vector<float> &last_confidences)
{
    if (!out_bgr.has_value())
    {
        return;
    }

    // This branch of the G-API graph contains these things that are all coming together as one bunch.
    CV_Assert(out_bgr_ts.has_value());

    // Now that we got a useful value, let's cache it for later.
    last_bgr = *out_bgr;

    // Mark up this frame with our preview function.
    cv::Mat marked_up_bgr;
    last_bgr.copyTo(marked_up_bgr);
    this->preview(marked_up_bgr, last_boxes, last_labels, last_confidences);

    // Stream the latest BGR frame.
    this->stream_frames(last_bgr, marked_up_bgr, *out_bgr_ts);

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(last_bgr, last_confidences);
}

} // namespace model
