// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <stdexcept>

// Third party includes
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Local includes
#include "rtsp.hpp"
#include "framebuffer.hpp"
#include "../util/helper.hpp"

namespace rtsp {

/** Parameters for an RTSP stream. */
typedef struct {
    /** Is this stream enabled? */
    bool enabled;

    /** Resolution. Should be one of the allowed resolutions: native, 1080p, 720p. */
    Resolution resolution;

    /** Frames per second. */
    int fps;

    /** The type of stream. */
    StreamType stream_type;

    /** Name of this stream's appsrc. */
    std::string name;

    /** A timestamp for each buffer we send out on this stream. */
    GstClockTime timestamp;

    /** The URI (or endpoint) that this stream accepts connections on. */
    std::string uri;

    /** The server. This pointer will be identical across all the stream. */
    GstRTSPServer *server;

    /** The factory that creates the GStreamer pipeline for this stream. */
    GstRTSPMediaFactory *factory;
} StreamParameters;

/** Parameters that we use whenever a new client attaches to the H.264 URI. */
typedef struct {
    /** Normal parameters. H.264 just has some extra stuff it needs to keep track of. */
    StreamParameters params;

    /** Is this the first frame processed so far? */
    gboolean first_frame_processed;

    /** The timestamp, which we will adjust. */
    GstClockTime base_timestamp;

    /** Pointer to the factory clock for H.264 factory. */
    GstClock *factory_clock;
} StreamParametersH264;

/** Declare a custom "class" that "inherits" from the GstRTSPMedia class */
#define CUSTOM_CLOCK_RTSP_MEDIA_TYPE (custom_clock_rtsp_media_get_type())

/** C-style struct for Gstreamer to figure out our custom class. */
struct CustomClockRTSPMediaClass {
    GstRTSPMediaClass parent;
};

/** C-style struct for Gstreamer to figure out our custom class. */
struct CustomClockRTSPMedia {
    GstRTSPMedia parent;
};

/** Here we actually define the class. */
G_DEFINE_TYPE(CustomClockRTSPMedia, custom_clock_rtsp_media, GST_TYPE_RTSP_MEDIA);

/** The name of the appsrc for raw UDP streams. */
const std::string rtsp_raw_udp_source_name = "rtsp-src-raw-udp";

/** The name of the appsrc for the raw TCP streams. */
const std::string rtsp_raw_tcp_source_name = "rtsp-src-raw-tcp";

/** The name of the appsrc for the result UDP streams. */
const std::string rtsp_result_udp_source_name = "rtsp-src-result-udp";

/** The name of the appsrc for the result TCP streams. */
const std::string rtsp_result_tcp_source_name = "rtsp-src-result-tcp";

/** The name of the appsrc for the H.264 stream. */
const std::string rtsp_h264_source_name = "rtsp-src-h.264";

/** Default FPS. */
const int DEFAULT_FPS = 10;

/** Struct to contain the parameters for the raw UDP stream. Read by a callback function. */
static StreamParameters raw_udp_context {
    .enabled        = true,
    .resolution     = Resolution::NATIVE,
    .fps            = DEFAULT_FPS,
    .stream_type    = StreamType::RAW,
    .name           = rtsp_raw_udp_source_name,
    .timestamp      = 0,
    .uri            = "/raw",
    .server         = nullptr,
    .factory        = nullptr,
};

/** Struct to contain the parameters for the raw TCP stream. Read by a callback function. */
static StreamParameters raw_tcp_context {
    .enabled        = true,
    .resolution     = Resolution::NATIVE,
    .fps            = DEFAULT_FPS,
    .stream_type    = StreamType::RAW,
    .name           = rtsp_raw_tcp_source_name,
    .timestamp      = 0,
    .uri            = "/rawTCP",
    .server         = nullptr,
    .factory        = nullptr,
};

/** Struct to contain the parameters for the result UDP stream. Read by a callback function. */
static StreamParameters result_udp_context {
    .enabled        = true,
    .resolution     = Resolution::NATIVE,
    .fps            = DEFAULT_FPS,
    .stream_type    = StreamType::RESULT,
    .name           = rtsp_result_udp_source_name,
    .timestamp      = 0,
    .uri            = "/result",
    .server         = nullptr,
    .factory        = nullptr,
};

/** Struct to contain the parameters for the result TCP stream. Read by a callback function. */
static StreamParameters result_tcp_context {
    .enabled        = true,
    .resolution     = Resolution::NATIVE,
    .fps            = DEFAULT_FPS,
    .stream_type    = StreamType::RESULT,
    .name           = rtsp_result_tcp_source_name,
    .timestamp      = 0,
    .uri            = "/resultTCP",
    .server         = nullptr,
    .factory        = nullptr,
};

/** Struct to contain the parameters for the H.264 stream. Read by a callback function. */
static StreamParameters h264_context {
    .enabled        = true,
    .resolution     = Resolution::NATIVE,
    .fps            = DEFAULT_FPS,
    .stream_type    = StreamType::H264_RAW,
    .name           = rtsp_h264_source_name,
    .timestamp      = 0,
    .uri            = "/h264raw",
    .server         = nullptr,
    .factory        = nullptr,
};

/** The maximum number of frames we keep in memory for each queue before we start overwriting old ones. */
static const size_t QUEUE_SIZE = 240;

/** We have a single unique FrameBuffer for the raw frames. */
FrameBuffer raw_buffer(QUEUE_SIZE, DEFAULT_FPS);

/** We have a single unique FrameBuffer for the result frames (the ones with inference results overlaid on top of them). */
FrameBuffer result_buffer(QUEUE_SIZE, DEFAULT_FPS);

/** This is the H.264 frame we feed out whenever we need more for the H.264 stream. Resolution is taken care of by the AI model. */
static std::queue<H264> h264_buffer;


/** Tell Gstreamer to read the media clock and use it as-is as the NTP timestamp. */
static gboolean custom_setup_rtpbin(GstRTSPMedia *media, GstElement *rtpbin)
{
    g_object_set(rtpbin, "ntp-time-source", 3 /* clock-time */, NULL);

    // If clock sync seems off, uncommenting the below line might help. This will tell the RTP Manager to set the NTP
    // timestamp in the RTCP SR to the "capture" time rather than the "send" time. This might help when there is a delay
    // between the time the frame was captured and when it is actually sent.
    // g_object_set(rtpbin, "rtcp-sync-send-time", FALSE, NULL);

    return TRUE;
}

/** Initializes the custom media class. */
static void custom_clock_rtsp_media_class_init(CustomClockRTSPMediaClass *class_param)
{
    // Register our callback function that will be invoked when the RTP sender stack is initialized
    class_param->parent.setup_rtpbin = custom_setup_rtpbin;
}

/** This init function is required for compiling even though it does not do anything. */
static void custom_clock_rtsp_media_init(CustomClockRTSPMedia *media)
{
}

/** Manually clean up exired sessions to prevent a tiny memory leak. */
static gboolean clean_up_expired_sessions(GstRTSPServer *server)
{
    GstRTSPSessionPool *pool = gst_rtsp_server_get_session_pool(server);
    gst_rtsp_session_pool_cleanup(pool);
    g_object_unref(pool);

    return TRUE;
}

/** Remove client connections of the given type. */
static GstRTSPFilterResult client_filter(GstRTSPServer *server, GstRTSPClient *client, gpointer user_data)
{
    // TODO: This is wrong: we want to remove only the clients that are attached to the stream that is being disabled.
    return GST_RTSP_FILTER_REMOVE;
}

/** Disconnect the given stream, kicking off any clients and preventing any further clients from connecting. */
static void disconnect(const StreamType &stream_type)
{
    std::string uri1;
    std::string uri2;
    switch (stream_type)
    {
        case StreamType::RAW:
            uri1 = raw_udp_context.uri;
            uri2 = raw_tcp_context.uri;
            break;
        case StreamType::RESULT:
            uri1 = result_udp_context.uri;
            uri2 = result_tcp_context.uri;
            break;
        case StreamType::H264_RAW:
            uri1 = h264_context.uri;
            uri2 = h264_context.uri; // hack to make the rest of the function nice
            break;
        default:
            util::log_error("Cannot disconnect from unknown stream.");
            assert(false);
            return;
    }

    // Remove the factory from the mount point so that no one else can connect to its URI
    auto server = raw_udp_context.server; // All the server pointers are the same, so it doesn't matter which context
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_remove_factory(mounts, uri1.c_str());
    gst_rtsp_mount_points_remove_factory(mounts, uri2.c_str());
    g_object_unref(mounts);

    // Remove existing clients
    gst_rtsp_server_client_filter(server, client_filter, NULL);
}

/**
 * If there is only one frame left in the queue, return a copy of it,
 * but if there is more than one frame left in the queue, move the earliest one
 * out of it.
 */
static cv::Mat get_frame(const std::string &stream_name)
{
    if (stream_name == rtsp_raw_udp_source_name)
    {
        return raw_buffer.get(raw_udp_context.resolution);
    }
    else if (stream_name == rtsp_raw_tcp_source_name)
    {
        return raw_buffer.get(raw_tcp_context.resolution);
    }
    else if (stream_name == rtsp_result_udp_source_name)
    {
        return result_buffer.get(result_udp_context.resolution);
    }
    else
    {
        return result_buffer.get(result_tcp_context.resolution);
    }
}

/** Callback to call whenever our app source needs another buffer to feed out. */
static void need_data_callback(GstElement *appsrc, guint unused, StreamParameters *params)
{
    // Feed out the buffer
    cv::Mat frame = get_frame(params->name);
    guint size = frame.size().width * frame.size().height * frame.channels();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    gst_buffer_fill(buffer, 0, frame.data, size);

    // Increment the timestamp.
    GST_BUFFER_PTS(buffer) = params->timestamp;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, params->fps);
    params->timestamp += GST_BUFFER_DURATION(buffer);

    // Push the RGB frame into the pipeline
    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);

    // Clean up after ourselves
    gst_buffer_unref(buffer);
}

/** Callback to call whenever our app source needs another buffer to feed out (H.264 version). */
static void need_data_callback_h264(GstElement *appsrc, guint unused, StreamParametersH264 *params)
{
    // Turn our H.264 frame into a Gstreamer buffer
    H264 frame = h264_buffer.front();
    guint size = frame.data.size();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    gst_buffer_fill(buffer, 0, frame.data.data(), size);

    if (!params->first_frame_processed)
    {
        GstClockTime internal_time = gst_clock_get_internal_time(params->factory_clock);

        // Convert the video frame timestamp that is in nanoseconds from January 1st, 1970 to NTP format,
        // which counts from January 1st, 1900.
        GstClockTime ntp_time = frame.timestamp + (2208988800LL * GST_SECOND);

        // Recalibrate the clock so that the current time is the same as the NTP timestamp of the first video frame
        gst_clock_set_calibration(params->factory_clock, internal_time, ntp_time, 1, 1);

        // Remember the timestamp of the first frame as this will be used to zero-adjust the PTS
        params->base_timestamp = frame.timestamp;
        params->first_frame_processed = TRUE;
    }

    // Use the provided timestamp and compute an estimated duration based on the fixed frame rate.
    // The PTS is adjusted to make the first frame have a PTS of zero.
    // The RTCP sender reports will use params->factory_clock for the NTP timestamp.
    GST_BUFFER_PTS(buffer) = frame.timestamp - params->base_timestamp; // TODO: Am I convinced that wrap-around will never mess us up here?
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, params->params.fps);

    // Push the H.264 frame into the pipeline
    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
    if (ret != GST_FLOW_OK)
    {
        util::log_error("Got an unexpected return value from pushing the h.264 frame to Gstreamer pipeline: " + std::to_string(ret));
    }

    // If we have any more frames behind this one, let's pop this one.
    // Otherwise, let's keep sending this one until we get another.
    if (h264_buffer.size() > 1)
    {
        h264_buffer.pop();
    }
}

/** Called when a new media pipeline is constructed. As such, operates in callback context. Make sure it is re-entrant! */
static void configure_stream(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    // Get our parameters from the user data
    StreamParameters *params = (StreamParameters *)user_data;

    // Get the appsrc for this factory
    GstElement *media_element = gst_rtsp_media_get_element(media);
    GstElement *appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(media_element), params->name.c_str());

    // Tell appsrc that we will be dealing with a timestamped buffer
    gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");

    // Determine the width and height of the stream from its resolution.
    auto buffer = get_frame(params->name);
    int width = buffer.size().width;
    int height = buffer.size().height;

    // Configure the video's caps (capabilities)
    g_object_set(G_OBJECT(appsrc), "caps",
        gst_caps_new_simple("video/x-raw",
            "format",    G_TYPE_STRING,     "BGR",
            "width",     G_TYPE_INT,        width,
            "height",    G_TYPE_INT,        height,
            "framerate", GST_TYPE_FRACTION, params->fps, 1,
            NULL),
        NULL);

    // Need to create a new context for each new stream's need-data callback. Otherwise you can only ever have one client ever.
    auto new_context = g_new0(StreamParameters, 1);
    new_context->enabled = params->enabled;
    new_context->resolution = params->resolution;
    new_context->fps = params->fps;
    new_context->stream_type = params->stream_type;
    new_context->name = params->name;
    new_context->timestamp = 0;
    new_context->uri = params->uri;
    new_context->server = params->server;
    g_object_set_data_full(G_OBJECT(media_element), "my-extra-data", new_context, (GDestroyNotify)g_free);

    // We call this callback whenever we need a new buffer to feed out.
    g_signal_connect(appsrc, "need-data", (GCallback)need_data_callback, new_context);

    // Clean up after ourselves
    gst_object_unref(appsrc);
    gst_object_unref(media_element);
}

/** Called when a new media pipeline is constructed. As such, operates in callback context. Make sure it is re-entrant! */
static void configure_stream_h264(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    // Get our parameters from the user data
    StreamParameters *params = (StreamParameters *)user_data;

    // Get the appsrc for this factory
    GstElement *media_element = gst_rtsp_media_get_element(media);
    GstElement *appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(media_element), params->name.c_str());

    // Tell appsrc that we will be dealing with a timestamped buffer
    gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");

    // Determine the width and height of the stream from its resolution.
    int width;
    int height;
    std::tie(height, width) = get_height_and_width(h264_context.resolution);

    // Configure the video's caps (capabilities)
    g_object_set(G_OBJECT(appsrc), "caps",
        gst_caps_new_simple("video/x-h264",
            "stream-format",    G_TYPE_STRING,     "byte-stream",
            "width",            G_TYPE_INT,        width,
            "height",           G_TYPE_INT,        height,
            "framerate",        GST_TYPE_FRACTION, params->fps, 1,
            NULL),
        NULL);

    // Need to create a new context for each new stream's need-data callback. Otherwise you can only ever have one client ever.
    auto new_context = g_new0(StreamParametersH264, 1);
    new_context->params.enabled = params->enabled;
    new_context->params.resolution = params->resolution;
    new_context->params.fps = params->fps;
    new_context->params.stream_type = params->stream_type;
    new_context->params.name = params->name;
    new_context->params.uri = params->uri;
    new_context->params.server = params->server;
    new_context->first_frame_processed = FALSE;
    new_context->base_timestamp = 0;
    new_context->factory_clock = gst_rtsp_media_factory_get_clock(factory);
    g_object_set_data_full(G_OBJECT(media_element), "my-extra-data", new_context, (GDestroyNotify)g_free);

    // We call this callback whenever we need a new buffer to feed out.
    g_signal_connect(appsrc, "need-data", (GCallback)need_data_callback_h264, new_context);

    // Clean up after ourselves
    gst_object_unref(appsrc);
    gst_object_unref(media_element);
}

/** Create and configure an RTSP stream factory. Factories are used to create GStreamer pipelines in response to client connections. */
static void configure_rtsp_stream_factory(GstRTSPMediaFactory *factory, const std::string &appsrc_name, GstRTSPLowerTrans protocol, void *configure_stream_arg)
{
    // This is the GStreamer pipeline that will be created whenever someone connects to this factory's endpoint.
    auto gstreamer_cmd = "( appsrc name=" + appsrc_name + " ! videoconvert ! video/x-raw,format=I420 ! jpegenc ! rtpjpegpay name=pay0 pt=96 )";
    gst_rtsp_media_factory_set_launch(factory, gstreamer_cmd.c_str());

    // Use the appropriate protocol (TCP or UDP)
    gst_rtsp_media_factory_set_protocols(factory, protocol);

    // Share the same GStreamer pipeline for each new client connection, rather than creating a brand new one for each.
    gst_rtsp_media_factory_set_shared(factory, true);

    // Hook up the callback for whenever a client connects to the server and the factory creates a new GStreamer pipeline in response
    // We hook up to this factory's "media-configure" signal.
    g_signal_connect(factory, "media-configure", (GCallback)configure_stream, configure_stream_arg);
}

/** Create and configure an RTSP stream factory for the H.264 stream. Factories are used to create GStreamer pipelines in response to client connections. */
static void configure_rtsp_stream_factory_h264(GstRTSPMediaFactory *factory, const std::string &appsrc_name, GstRTSPLowerTrans protocol, void *configure_stream_arg)
{
    // This is the GStreamer pipeline that will be created whenever someone connects to this factory's endpoint.
    auto gstreamer_cmd = "( appsrc name=" + appsrc_name + " ! h264parse ! rtph264pay name=pay0 pt=96 )";
    gst_rtsp_media_factory_set_launch(factory, gstreamer_cmd.c_str());

    // Use the appropriate protocol (TCP or UDP)
    gst_rtsp_media_factory_set_protocols(factory, protocol);

    // Use our custom class, CustomClockRTSPMedia, rather than the regular GstRTSPMedia object.
    gst_rtsp_media_factory_set_media_gtype(factory, CUSTOM_CLOCK_RTSP_MEDIA_TYPE);

    // Create a system clock to be used by this factory only.
    GstClock *factory_system_clock = reinterpret_cast<GstClock *>(g_object_new(gst_system_clock_get_type(), NULL));

    // Tell this factory to use the specified clock as the media clock. We will sync this clock
    // to the UTC time once we process our first packet.
    gst_rtsp_media_factory_set_clock(factory, factory_system_clock);

    // Release our reference to the clock. The factory owns it now.
    g_object_unref(factory_system_clock);

    // Share the same GStreamer pipeline for each new client connection, rather than creating a brand new one for each.
    gst_rtsp_media_factory_set_shared(factory, true);

    // Hook up the callback for whenever a client connects to the server and the factory creates a new GStreamer pipeline in response
    // We hook up to this factory's "media-configure" signal.
    g_signal_connect(factory, "media-configure", (GCallback)configure_stream_h264, configure_stream_arg);
}

/** Connect the given stream. */
static void connect(const StreamType &stream_type)
{
    // Configure the server's GstRTSPMountPoints.
    // This object is responsible for creating GStreamer pipelines that hook up to a client whenever
    // a client connects to a particular URL and thereby creates a session with our server.
    // To create these GStreamer pipelines on the fly, we need a properly configured factory.
    auto server = raw_udp_context.server; // All the server pointers are the same, so it doesn't matter which context
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);

    switch (stream_type)
    {
        case StreamType::RAW:
            raw_udp_context.factory = gst_rtsp_media_factory_new();
            raw_tcp_context.factory = gst_rtsp_media_factory_new();
            configure_rtsp_stream_factory(raw_udp_context.factory, rtsp_raw_udp_source_name, GST_RTSP_LOWER_TRANS_UDP, (void *)&raw_udp_context);
            configure_rtsp_stream_factory(raw_tcp_context.factory, rtsp_raw_tcp_source_name, GST_RTSP_LOWER_TRANS_TCP, (void *)&raw_tcp_context);
            gst_rtsp_mount_points_add_factory(mounts, raw_udp_context.uri.c_str(), raw_udp_context.factory);
            gst_rtsp_mount_points_add_factory(mounts, raw_tcp_context.uri.c_str(), raw_tcp_context.factory);
            break;
        case StreamType::RESULT:
            result_udp_context.factory = gst_rtsp_media_factory_new();
            result_tcp_context.factory = gst_rtsp_media_factory_new();
            configure_rtsp_stream_factory(result_udp_context.factory, rtsp_result_udp_source_name, GST_RTSP_LOWER_TRANS_UDP, (void *)&result_udp_context);
            configure_rtsp_stream_factory(result_tcp_context.factory, rtsp_result_tcp_source_name, GST_RTSP_LOWER_TRANS_TCP, (void *)&result_tcp_context);
            gst_rtsp_mount_points_add_factory(mounts, result_udp_context.uri.c_str(), result_udp_context.factory);
            gst_rtsp_mount_points_add_factory(mounts, result_tcp_context.uri.c_str(), result_tcp_context.factory);
            break;
        case StreamType::H264_RAW:
            h264_context.factory = gst_rtsp_media_factory_new();
            configure_rtsp_stream_factory_h264(h264_context.factory, rtsp_h264_source_name, GST_RTSP_LOWER_TRANS_TCP, (void *)&h264_context);
            gst_rtsp_mount_points_add_factory(mounts, h264_context.uri.c_str(), h264_context.factory);
            break;
        default:
            util::log_error("Cannot connect a factory to a stream I don't recognize.");
            assert(false);
            break;
    }

    g_object_unref(mounts);
}

Resolution get_resolution(const StreamType &type)
{
    switch (type)
    {
        case StreamType::RAW:
            return raw_udp_context.resolution;
        case StreamType::RESULT:
            return result_udp_context.resolution;
        case StreamType::H264_RAW:
            return h264_context.resolution;
        default:
            util::log_error("Could not get a resolution for a type of stream I don't know. Returning raw one instead.");
            assert(false);
            return raw_udp_context.resolution;
    }
}

void* gst_rtsp_server_thread(void *unused)
{
    // Initialize GStreamer and check if it failed.
    GError *err;
    gboolean success = gst_init_check(NULL, NULL, &err);
    if (!success)
    {
        util::log_error("Could not initialize GStreamer: " + ((err == NULL) ? "No info." : std::to_string(err->code) + ": " + std::string(err->message)));
        return NULL;
    }

    // Create the main GLib loop, using the default context
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    // Create the RTSP server and configure it.
    // An RTSP server manages four objects under the hood: GstRTSPSessionPool, which keeps track of all the sessions
    // created by client connections, GstRTSPAuth, which is responsible for authenticating connected users,
    // GstRTSPThreadPool, which is a pool of threads used internally, and GstRTSPMountPoints, which manages specific
    // RTSP stream configurations and URLs. By default, we listen on port 8554.
    // TODO: Should we configure GstRTSPAuth to make this private somehow?
    GstRTSPServer *server = gst_rtsp_server_new();
    raw_udp_context.server = server;
    raw_tcp_context.server = server;
    result_udp_context.server = server;
    result_tcp_context.server = server;
    h264_context.server = server;

    // Connect all the factories to their mount points
    connect(StreamType::RAW);
    connect(StreamType::RESULT);
    connect(StreamType::H264_RAW);

    // Set up the callback to clean up disconnected clients periodically
    g_timeout_add_seconds(60, (GSourceFunc)clean_up_expired_sessions, server);

    // Start the GLib main loop. The RTSP server hooks into the main loop's default context and runs as part of it.
    gst_rtsp_server_attach(server, NULL);
    g_main_loop_run(loop);

    // Should not return.
    return NULL;
}

static void add_bunch_of_frames(const std::vector<cv::Mat> &mats, const std::vector<int64_t> &timestamps, FrameBuffer &buffer)
{
    // If we have too many frames to put into the buffer,
    // we will overflow our buffer, leading to jumps in time.
    // But if we don't deliver enough, the stream will pause
    // waiting for new frames.
    //
    // So let's remove every Nth frame from the delivery to make
    // sure we don't overflow.
    const auto max_allowed = buffer.room();

    if (max_allowed == 0)
    {
        // No room. Ignore this request.
        return;
    }
    else if (mats.size() > max_allowed)
    {
        // Only take every Nth frame (up to however many we are allowed to take)
        size_t n = mats.size() / max_allowed;
        #ifdef DEBUG_TIME_ALIGNMENT
            util::log_debug("Not enough room. Only taking " + std::to_string(max_allowed) + " frames. So getting every " + std::to_string(n) + "th/rd/nd");
        #endif

        size_t taken = 0;
        for (size_t i = 0; i < mats.size(); i++)
        {
            if ((i % n) == 0)
            {
                buffer.put(mats.at(i), timestamps.at(i));
                taken++;
            }

            if (taken >= max_allowed)
            {
                break;
            }
        }
    }
    else
    {
        // Put them all in
        for (const auto &mat : mats)
        {
            buffer.put(mat);
        }
    }
}

void update_data_raw(const cv::Mat &mat, int64_t timestamp)
{
    update_data_raw(std::vector<cv::Mat>{mat}, std::vector<int64_t>{timestamp});
}

void update_data_raw(const std::vector<cv::Mat> &mats, const std::vector<int64_t> &timestamps)
{
    add_bunch_of_frames(mats, timestamps, raw_buffer);
}

void update_data_result(const cv::Mat &mat, int64_t timestamp)
{
    update_data_result(std::vector<cv::Mat>{mat}, std::vector<int64_t>{timestamp});
}

void update_data_result(const std::vector<cv::Mat> &mats, const std::vector<int64_t> &timestamps)
{
    add_bunch_of_frames(mats, timestamps, result_buffer);
}

void update_data_h264(const H264 &frame)
{
    // Make sure we don't leak memory or get super far behind on the frames
    // if we are consuming them slower than we are producing them.
    while (h264_buffer.size() > (size_t)(2 * h264_context.fps))
    {
        h264_buffer.pop();
    }

    h264_buffer.push(frame);
}

void set_stream_params(const StreamType &type, bool enable)
{
    bool previously_enabled;
    switch (type)
    {
        case StreamType::RAW:
            previously_enabled = raw_udp_context.enabled || raw_tcp_context.enabled;
            raw_udp_context.enabled = enable;
            raw_tcp_context.enabled = enable;
            util::log_info("Raw RTSP stream " + std::string(enable ? "enabled" : "disabled"));
            break;
        case StreamType::RESULT:
            previously_enabled = result_udp_context.enabled || result_tcp_context.enabled;
            result_udp_context.enabled = enable;
            result_tcp_context.enabled = enable;
            util::log_info("Result RTSP stream " + std::string(enable ? "enabled" : "disabled"));
            break;
        case StreamType::H264_RAW:
            previously_enabled = h264_context.enabled;
            h264_context.enabled = enable;
            util::log_info("H.264 RTSP stream " + std::string(enable ? "enabled" : "disabled"));
            break;
        default:
            util::log_error("Unrecognized stream type when trying to enable/disable an RTSP stream.");
            assert(false);
            previously_enabled = false;  // Make release build happy
            break;
    }

    if (previously_enabled && !enable)
    {
        // If we are going from enabled to disabled, we should disconnect any clients and remove the factory from URI mount.
        disconnect(type);
    }

    if (!previously_enabled && enable)
    {
        // If we are going from disabled to enabled, we need to reconnect the factories to their URIs.
        connect(type);
    }
}

void set_stream_params(const StreamType &type, int fps)
{
    if (fps <= 0)
    {
        util::log_error("Attempted to configure RTSP streams to invalid FPS of " + std::to_string(fps));
        return;
    }

    switch (type)
    {
        case StreamType::RAW:
            raw_buffer.set_fps(fps);
            raw_udp_context.fps = fps;
            raw_tcp_context.fps = fps;
            util::log_info("Raw RTSP Stream's FPS changed to " + std::to_string(fps));
            break;
        case StreamType::RESULT:
            result_buffer.set_fps(fps);
            result_udp_context.fps = fps;
            result_tcp_context.fps = fps;
            util::log_info("Result RTSP Stream's FPS changed to " + std::to_string(fps));
            break;
        case StreamType::H264_RAW:
            h264_context.fps = fps;
            util::log_info("H.264 Stream's FPS changed to " + std::to_string(fps));
            break;
        default:
            util::log_error("Unrecognized stream type when trying to change an RTSP stream's FPS.");
            assert(false);
            break;
    }
}

void set_stream_params(const StreamType &type, const Resolution &resolution)
{
    switch (type)
    {
        case StreamType::RAW:
            raw_udp_context.resolution = resolution;
            raw_tcp_context.resolution = resolution;
            util::log_info("Raw RTSP Stream's resolution changed to " + resolution_to_string(resolution));
            break;
        case StreamType::RESULT:
            result_udp_context.resolution = resolution;
            result_tcp_context.resolution = resolution;
            util::log_info("Result RTSP Stream's resolution changed to " + resolution_to_string(resolution));
            break;
        case StreamType::H264_RAW:
            h264_context.resolution = resolution;
            util::log_info("H.264 Stream's resolution changed to " + resolution_to_string(resolution));
            break;
        default:
            util::log_error("Unrecognized stream type when trying to change an RTSP stream's resolution.");
            assert(false);
            break;
    }
}

void set_stream_params(const StreamType &type, int fps, bool enable)
{
    set_stream_params(type, fps);
    set_stream_params(type, enable);
}

void set_stream_params(const StreamType &type, const Resolution &resolution, bool enable)
{
    set_stream_params(type, resolution);
    set_stream_params(type, enable);
}

void set_stream_params(const StreamType &type, const Resolution &resolution, int fps, bool enable)
{
    set_stream_params(type, resolution, enable);
    set_stream_params(type, fps);
}

void take_snapshot(const StreamType &type)
{
    cv::Mat snapshot;
    switch (type)
    {
        case (StreamType::RAW):
            snapshot = get_frame(rtsp_raw_udp_source_name);
            cv::imwrite("/snapshot/snapshot.jpg", snapshot);
            break;
        case (StreamType::RESULT):
            snapshot = get_frame(rtsp_result_udp_source_name);
            cv::imwrite("/snapshot/snapshot.jpg", snapshot);
            break;
        default:
            util::log_error("invalid stream type.");
            break;
    }
}
} // namespace rtsp
