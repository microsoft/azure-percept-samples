// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <thread>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/render.hpp>

// Local includes
#include "ssd.hpp"
#include "../kernels/ssd_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace custom {

using GDetections = cv::GArray<cv::Rect>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(BBoxes, <GPrims(GDetections)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // Converts the rectangles into G-API's rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
    }
};

} // namespace custom

namespace model {

/** An SSD network takes a single input and outputs a single output (which we will parse into boxes, labels, and confidences) */
G_API_NET(SSDNetwork, <cv::GMat(cv::GMat)>, "ssd-network");

SSDModel::SSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string inputsource, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : ObjectDetector{ labelfpath, modelfpaths, mvcmd, videofile, resolution }
{
    this->inputsource = inputsource;
}

void SSDModel::load_default()
{
    int ret = util::run_command("rm -rf /app/model && mkdir /app/model");
    if (ret != 0)
    {
        util::log_error("rm && mkdir failed with " + ret);
    }

    this->modelfiles = {"/app/data/ssd_mobilenet_v2_coco.blob"};
    this->labelfpath = "/app/data/labels.txt";
}

void SSDModel::run(cv::GStreamingCompiled *pipeline)
{
    while (true)
    {
        // Wait for the VPU to come up.
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);

        // Log some metadata.
        this->log_parameters();
        bool ran_out_naturally;

        if ((this->inputsource == "uvc") || (this->inputsource.rfind(this->VIDEO_PREFIX, 0) == 0)) 
        {
            // Build the camera pipeline with G-API
            *pipeline = this->compile_cv_graph_uvc_video();
            util::log_info("starting the pipeline with " + this->inputsource);
            pipeline->start();

            // Pull data through the pipeline
            ran_out_naturally = this->pull_data_uvc_video(*pipeline);
        } 
        else 
        {
            // Build the camera pipeline with G-API
            *pipeline = this->compile_cv_graph();
            util::log_info("starting the pipeline...");
            pipeline->start();

            // Pull data through the pipeline
            ran_out_naturally = this->pull_data(*pipeline);

        }
        
        if (!ran_out_naturally)
        {
            break;
        }
    }
}

cv::GStreamingCompiled SSDModel::compile_cv_graph() const
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
    cv::GOpaque<int64_t> img_ts = cv::gapi::streaming::timestamp(img);

    // This node branches off from the preproc node for neural network inferencing.
    cv::GMat bgr = cv::gapi::streaming::desync(preproc);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(bgr);

    // Here's where we actually run our neural network. It runs on the VPU.
    cv::GMat nn = cv::gapi::infer<SSDNetwork>(bgr);

    // Get some useful metadata.
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    // Here's where we post-process our network's outputs into bounding boxes, IDs, and confidences.
    cv::GArray<cv::Rect> rcs;
    cv::GArray<int> ids;
    cv::GArray<float> cfs;
    std::tie(rcs, ids, cfs) = cv::gapi::streaming::parseSSDWithConf(nn, sz);

    // Specify the boundaries of the G-API graph (the inputs and outputs).
    auto graph = cv::GComputation(cv::GIn(in),
                                  cv::GOut(h264, h264_seqno, h264_ts,               // H.264 branch
                                          img, img_ts,                              // Raw frame branch
                                          nn_seqno, nn_ts, rcs, ids, cfs, sz));     // Inference branch

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of at least length 1.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<SSDNetwork>{this->modelfiles.at(0)});

    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseSSDWithConf>());

    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Percept DK's camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

cv::GStreamingCompiled SSDModel::compile_cv_graph_uvc_video() const
{
    // The input node of the G-API pipeline. This will be filled in, one frame at time.
    cv::GMat in;
    cv::GMat bgr = in;

    // Here's where we actually run our neural network. It runs on the VPU.
    cv::GMat nn = cv::gapi::infer<SSDNetwork>(bgr);

    // Get some useful metadata.
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(nn);
    cv::GOpaque<int64_t> ts    = cv::gapi::streaming::timestamp(nn);
    cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(bgr);

    // Here's where we post-process our network's outputs into bounding boxes, IDs, and confidences.
    cv::GArray<cv::Rect> objs;
    cv::GArray<int> tags;
    cv::GArray<float> confidences;
    std::tie(objs, tags, confidences) = cv::gapi::streaming::parseSSDWithConf(nn, sz);
    auto rendered = cv::gapi::wip::draw::render3ch(bgr, custom::BBoxes::on(objs));
    auto graph_ins = cv::GIn(in);
    auto graph_outs = cv::GOut(objs, tags, confidences, sz, nn_seqno, ts);
    graph_outs += cv::GOut(rendered);
    auto graph = cv::GComputation(std::move(graph_ins), std::move(graph_outs));

    // Pass the actual neural network blob file into the graph. We assume we have a modelfiles of at least length 1.
    CV_Assert(this->modelfiles.size() >= 1);
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<SSDNetwork>{this->modelfiles.at(0)});
    // Here we wrap up all the kernels (the implementations of the G-API ops) that we need for our graph.
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<custom::OCVBBoxes, cv::gapi::streaming::GOCVParseSSDWithConf>());
    // Compile the graph in streamnig mode; set all the parameters; feed the firmware file into the VPU.
    auto pipeline = graph.compileStreaming(cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    //if find VIDEO_PREFIX from inputsource, continue with a video file
    if (this->inputsource.rfind(this->VIDEO_PREFIX, 0) == 0) 
    {
        std::string video_file_path = this->inputsource.substr(this->VIDEO_PREFIX.size(), this->inputsource.size());
        util::log_info("Input source is a video file with path: " + video_file_path);
        if (!util::file_exists(video_file_path))
        {
            util::log_error("The video file doesn't exist.");
        }
        pipeline.setSource<cv::gapi::wip::GCaptureSource>(video_file_path);
    } 
    //else continue as uvc camera, video0 as default value
    else 
    {
        util::log_info("Input source is a uvc camera (video0 as default value)");
        pipeline.setSource<cv::gapi::wip::GCaptureSource>(0);
    }

    return pipeline;
}

void SSDModel::log_parameters() const
{
    // Log all the stuff
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg += ", firmware: " + this->mvcmd + ", parser: SSD, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model
