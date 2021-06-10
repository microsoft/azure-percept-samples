// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Local includes
#include "resolution.hpp"

// Standard library includes
#include <string>
#include <vector>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

namespace rtsp {

/** The different types of streams. */
enum class StreamType {
   RAW,        // Raw images from the camera
   RESULT,     // Raw images overlaid by whatever markups the network parser code has done
   H264_RAW,   // Raw images from the camera, encoded using H.264
};

/** A struct to represent an H264 frame. */
typedef struct {
   /** The actual data in the frame. */
   std::vector<uint8_t> data;

   /** The timestamp for this frame. */
   int64_t timestamp;
} H264;

/** Returns the current resolution of all the streams of the given type. */
Resolution get_resolution(const StreamType &type);

/** Main loop for the RTSP server thread. */
void* gst_rtsp_server_thread(void *unused);

/**
 * Set the given stream's parameters.
 *
 * Note that setting the resolution here is NECESSARY, but not SUFFICIENT. You must also restart
 * the model pipeline with iot::restart_model_with_new_resolution().
 */
void set_stream_params(const StreamType &type, bool enable);
void set_stream_params(const StreamType &type, int fps);
void set_stream_params(const StreamType &type, const Resolution &resolution);
void set_stream_params(const StreamType &type, int fps, bool enable);
void set_stream_params(const StreamType &type, const Resolution &resolution, bool enable);
void set_stream_params(const StreamType &type, const Resolution &resolution, int fps, bool enable);

/** This function would write the current frame to a specific location*/
void take_snapshot(const StreamType &type);

/** Update the RGB frame that we display in the raw RTSP stream. */
void update_data_raw(const cv::Mat &mat, int64_t timestamp);
void update_data_raw(const std::vector<cv::Mat> &mats, const std::vector<int64_t> &timestamps);

/** Update the RGB frame that we display in the result RTSP stream. */
void update_data_result(const cv::Mat &mat, int64_t timestamp);
void update_data_result(const std::vector<cv::Mat> &mats, const std::vector<int64_t> &timestamps);

/** Update the H.264 data that we display in the Raw H.264 stream. */
void update_data_h264(const H264 &frame);

} // namespace rtsp
