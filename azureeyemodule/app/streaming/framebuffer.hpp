// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

// Standard library includes
#include <atomic>
#include <deque>
#include <thread>

// Local includes
#include "resolution.hpp"
#include "../util/circular_buffer.hpp"

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>


namespace rtsp {

/**
 * A FrameBuffer represents a circular queue of frames to be sent out over RTSP.
 * When the frame would be empty, we just keep feeding out the last frame again and again.
 *
 * Getting a frame from the buffer does not remove it, instead, we maintain a rate
 * at which we want to change frames, and remove them at that rate. Every time
 * a client wants to get a frame from this buffer, we return the current frame.
 *
 * Therefore, this class can be read by multiple clients who all expect to get the same
 * information.
 *
 * Each framebuffer starts with a default frame, which will be what it sends out until
 * it gets something else.
 *
 * This is thread-safe. You can read and write to this concurrently.
 *
 * The rationale for this class is time alignment. If the neural network
 * wants to align its inferences in time, it needs to buffer up some number
 * of old raw frames until it gets a network inference. Then it goes
 * through the old frames to find the one that best matches in time
 * with the frame the network used to derive its inferences.
 * Once it finds the right frame, it marks up that frame and all older frames
 * with that information, then dumps all of those frames into this buffer.
 *
 * Without the need for time alignment, we could just have a single latest
 * frame, which would be updated every time we get a new one, which is
 * what we used to do.
 */
class FrameBuffer
{
public:
    /**
     * Constructor for the FrameBuffer class.
     *
     * @param max_length: The maximum number of frames to store in the buffer before old ones get written over.
     *                    If this number is too small, we may get a dump of frames from the neural network
     *                    of a longer length than we can handle. In that case, we will dump some of those frames
     *                    as we wrap around the buffer, creating a jump in time in the stream that is jarring
     *                    to the viewer.
     * @param fps: The frames per second at which to update the `get` frame. GStreamer RTSP server will call
     *             the `get()` method at some frames per second, which may be different than this, but if the
     *             FPS values differ, you might send duplicate frames or you might not send frames as often as
     *             you could. This is the initial value we use, but we internally adjust this to match the
     *             average rate at which frames are dumped into the buffer.
     */
    explicit FrameBuffer(size_t max_length, int fps);

    /** Destructor. */
    ~FrameBuffer();

    /**
     * We return the current frame.
     * We adjust the retrieved frame to the right resolution if it isn't already that resolution.
     * This may block up to as long as it takes for the internal FPS updating thread to update
     * the frame, which should be quite quick.
     */
    cv::Mat get(const Resolution &resolution);

    /** Put a new frame into the buffer. */
    void put(const cv::Mat &frame, int64_t timestamp);

    /** Get the number of frames we still have room for before we start overwriting old ones. */
    size_t room() const;

private:
    /** The internal container we use for holding the frames. */
    circbuf::CircularBuffer<cv::Mat> circular_buffer;

    /** The last N timestamps. We use these to calculate the rate frames are coming in. */
    std::deque<int64_t> last_n_timestamps;

    /** This is the latest frame that we have sent (or a default if we haven't sent any yet). */
    cv::Mat cached_frame;

    /** Lock to guard access to the cached frame, which gets read from whatever thread, and written from our internal thread. */
    std::mutex cached_frame_mutex;

    /** The frames per second that we update our cached frame. We adjust this value over time to match the rate at which we are getting frames. */
    std::atomic<int> fps;

    /** The FPS thread. This thread updates the `get` frame `fps` times per second. */
    std::thread fps_thread;

    /** When set to true, this signals the internal fps_thread to join. It will join at the next opportunity. */
    bool shut_down = false;

    /** The method our fps_thread runs. */
    void periodically_update_frame();
};

} // namespace rtsp