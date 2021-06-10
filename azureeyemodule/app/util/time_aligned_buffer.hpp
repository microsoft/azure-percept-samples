// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * This module represents a framebuffer that can be used for storing timestamped frames.
 * We use this for time-aligning frames with neural network inferences.
 */
#pragma once

// Standard library includes
#include <string>
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>

// Local includes
#include "helper.hpp"

namespace timebuf {

/** A tuple of cv::Mat and timestamp for that frame. */
using timestamped_frame_t = std::tuple<cv::Mat, int64_t>;

class TimeAlignedBuffer
{
public:
    /** Constructor. Takes a default value to return until we get frames in the buffer. */
    TimeAlignedBuffer(const cv::Mat &default_item, int64_t default_timestamp);

    /** Copies the given frame and timestamp into the buffer, overwriting an old one if the we end up wrapping. */
    void put(const timestamped_frame_t &frame_and_ts);

    /** Removes the best matching frame and all older ones and returns them as a vector. If no frames in buffer, we return the default one or the last one we returned. */
    void get_best_match_and_older(int64_t timestamp, std::vector<cv::Mat> &out_frames, std::vector<int64_t> &out_timestamps);

    /** Returns the current number of items in the buffer. */
    size_t size() const;

private:
    /** The index to write something to. */
    size_t index = 0;

    /** Number of timestamped frames that we can currently have in the buffer. Increases if we need more capacity. */
    size_t n_timestamped_frames;

    /** The frame we return if there are no frames to return (until we have some real ones). */
    cv::Mat default_value;

    /** The default timestamp we return if there are none to return (until we have some real ones). */
    int64_t default_timestamp;

    /** Circular buffer of frames with their timestamps. */
    std::vector<timestamped_frame_t> timestamped_frames;

    /** Find the oldest frame in the buffer, along with its timestamp and the timestamp that most closely matches `timestamp`. */
    void find_oldest_and_best_matching(int64_t timestamp, cv::Mat &oldest_frame, int64_t &oldest_ts, int64_t &best_match_ts) const;

    /** Find and remove the best matching frames and timestamps, and all older ones. */
    void remove_best_match_and_older(int64_t best_match_ts, std::vector<cv::Mat> &best_and_older, std::vector<int64_t> &best_and_older_ts);
};

} // namespace timebuf