// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <algorithm>
#include <iostream>
#include <sstream>
#include <map>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>

#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/infer/onnx.hpp>

// Local includes
#include "onnxssd.hpp"
#include "../kernels/ssd_kernels.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace {
void remap_ssd_ports(const std::unordered_map<std::string, cv::Mat> &onnx,
                           std::unordered_map<std::string, cv::Mat> &gapi) {
    // Assemble ONNX-processed outputs back to a single 1x1x200x7 blob
    // to preserve compatibility with OpenVINO-based SSD pipeline
    const cv::Mat &num_detections = onnx.at("num_detections:0");
    const cv::Mat &detection_boxes = onnx.at("detection_boxes:0");
    const cv::Mat &detection_scores = onnx.at("detection_scores:0");
    const cv::Mat &detection_classes = onnx.at("detection_classes:0");

    GAPI_Assert(num_detections.depth() == CV_32F);
    GAPI_Assert(detection_boxes.depth() == CV_32F);
    GAPI_Assert(detection_scores.depth() == CV_32F);
    GAPI_Assert(detection_classes.depth() == CV_32F);

    cv::Mat &ssd_output = gapi.at("detection_output");

    const int num_objects = static_cast<int>(num_detections.ptr<float>()[0]);
    const float *in_boxes = detection_boxes.ptr<float>();
    const float *in_scores = detection_scores.ptr<float>();
    const float *in_classes = detection_classes.ptr<float>();
    float *ptr = ssd_output.ptr<float>();

    for (int i = 0; i < num_objects; i++) {
        ptr[0] = 0.f;               // "image_id"
        ptr[1] = in_classes[i];     // "label"
        ptr[2] = in_scores[i];      // "confidence"
        ptr[3] = in_boxes[4*i + 1]; // left
        ptr[4] = in_boxes[4*i + 0]; // top
        ptr[5] = in_boxes[4*i + 3]; // right
        ptr[6] = in_boxes[4*i + 2]; // bottom

        ptr      += 7;
        in_boxes += 4;
    }
    if (num_objects < ssd_output.size[2]-1) {
        // put a -1 mark at the end of output blob if there is space left
        ptr[0] = -1.f;
    }
}

} // anonymous namespace

namespace model {

/** An ONNX SSD network takes a single input and outputs a single output */
G_API_NET(IntelONNXSSD, <cv::GMat(cv::GMat)>, "com.intel.onnx.ssd");

ONNXSSDModel::ONNXSSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd,
             const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : ObjectDetector{ labelfpath, modelfpaths, mvcmd, videofile, resolution }, onnxfpath(modelfpaths.at(0))
{
    //You should specify the model file (e.g. "-p=onnxssd -m=/app/data/ssd_mobilenet_v1_10.onnx"),
    //or the app will roll back to default mode
    this->labelfpath = "/app/data/labels.txt";
}

void ONNXSSDModel::run(cv::GStreamingCompiled* pipeline)
{
    while (true)
    {
        this->wait_for_device();

        // Read in the labels for classification
        label::load_label_file(this->class_labels, this->labelfpath);
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

cv::GStreamingCompiled ONNXSSDModel::compile_cv_graph() const{
    // Declare an empty GMat - the beginning of the pipeline
    cv::GMat in;
    auto graph_ins = cv::GIn(in); // inputs are identified early

    // Run the ISP preprocessing to produce BGR
    auto prep = cv::gapi::mx::preproc(in, this->resolution);
    auto bgr = cv::gapi::streaming::desync(prep);

    // Run Inference on the full frame
    auto blob = cv::gapi::infer<IntelONNXSSD>(bgr);

    // Parse the detections and project those to the original image frame
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(bgr);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(bgr);
    cv::GArray<cv::Rect> objs;
    cv::GArray<int> tags;
    cv::GArray<float> cfs;
    auto sz = cv::gapi::streaming::size(bgr);

    std::tie(objs, tags, cfs) = cv::gapi::streaming::parseSSDWithConf(blob, sz);
    auto graph_outs = cv::GOut(objs, tags, cfs, bgr, nn_seqno, nn_ts, sz);

    // Graph compilation
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseSSDWithConf>());
    auto detector = cv::gapi::onnx::Params<IntelONNXSSD>{ onnxfpath }
        .cfgOutputLayers({"detection_output"})
        .cfgPostProc({cv::GMatDesc{CV_32F, {1,1,200,7}}}, remap_ssd_ports);
    auto networks = cv::gapi::networks(detector);
    auto pipeline = cv::GComputation(std::move(graph_ins), std::move(graph_outs))
        .compileStreaming(cv::gapi::mx::Camera::params(),
                          cv::compile_args(kernels, networks,
                          cv::gapi::mx::mvcmdFile{this->mvcmd}));
    // Graph execution
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

bool ONNXSSDModel::pull_data(cv::GStreamingCompiled &pipeline){
    cv::util::optional<std::vector<cv::Rect>> out_boxes;
    cv::util::optional<std::vector<int>>      out_labels;
    cv::util::optional<std::vector<float>>    out_confidences;
    cv::util::optional<cv::Mat>               out_bgr;
    cv::util::optional<int64_t>               out_bgr_seqno;
    cv::util::optional<int64_t>               out_bgr_ts;
    cv::util::optional<cv::Size>               out_size;

    auto pipeline_outputs = cv::gout(out_boxes, out_labels, out_confidences,
                                     out_bgr, out_bgr_seqno, out_bgr_ts, out_size);

    std::vector<cv::Rect> last_boxes;
    std::vector<int>      last_labels;
    std::vector<float>    last_confidences;
    cv::Mat last_bgr;
    int64_t last_bgr_timestamp;

    // Pull the data from the pipeline while it is running
    while (pipeline.pull(std::move(pipeline_outputs)))
    {
        if(out_boxes.has_value())
        {
            this->handle_inference_output(out_bgr_ts, out_bgr_seqno, out_boxes, out_labels, out_confidences, out_size, last_boxes, last_labels, last_confidences);
        }
        this->handle_bgr_output(out_bgr, out_bgr_ts, last_bgr, last_bgr_timestamp, last_boxes, last_labels, last_confidences);

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

void ONNXSSDModel::log_parameters() const
{
    // Log all the stuff
    std::string msg = "onnx: ";
    msg += onnxfpath;
    msg += ", firmware: " + this->mvcmd + ", parser: ONNXSSD, label: " + this->labelfpath + ", classes: " + std::to_string((int)this->class_labels.size());
    util::log_info(msg);
}

} // namespace model
