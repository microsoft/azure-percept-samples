// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>
#include <tuple>

// Third party includes
#include <opencv2/gapi/mx.hpp>

// Local includes
#include "parser.hpp"
#include "../util/helper.hpp"
#include "../util/timing.hpp"
#include "../util/time_aligned_buffer.hpp"

namespace model {

/**
 * Abstract base class for all model types that run on the Azure Percept device.
 */
class AzureEyeModel
{
public:
    AzureEyeModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    virtual ~AzureEyeModel();

    /**
     * Takes in the given data and does the following:
     *
     * 1. Checks if data is a URL. If so, we download the model as a .zip file.
     *
     * 2. If we now have a .zip file, we unzip the package.
     *
     * 3. If the unpacked .zip file contains configuration files, we determine the
     *    type of model we should build and return the appropriate stuff from the
     *    config file(s).
     *
     * 4. If our model is a .xml file, we convert it to a .blob file.
     *
     * Ultimately, the point of this function is to fill in whatever parameters are wrong
     * in the command line invocation (either because they were passed in incorrectly
     * or because they are now outdated), and to fall back to a default if we can't.
     *
     * @returns True if we successfully parsed all the model dependencies. False means
     *          we should load a default model, as dependencies were not parsed correctly.
     */
    static bool load(std::string &labelfile, std::vector<std::string> &modelfiles, parser::Parser &modeltype);

    /**
     * This method will run the model by pulling data through its OpenCV GAPI graph.
     * This method is meant to handle the output of the model at each frame,
     * meaning sending messages or RTSP feed, for example.
     *
     * This method will return only once we run out of frames (if we have a finite source) or
     * if we call set_update_flag();
     */
    virtual void run(cv::GStreamingCompiled*) = 0;

    /**
     * Cause the run method to exit so that we can update the model to a new one.
     */
    void set_update_flag();

    /**
     * Update the model's data collecion loop parameters.
     */
    virtual void update_data_collection_params(bool enable, unsigned long int interval_seconds);

    /**
     * Update the model's time alignment property.
     */
    void update_time_alignment(bool enable);

    /**
     * Get the model's data collecion loop parameters.
     */
    void get_data_collection_params(bool &enable, unsigned long int &interval) const;

    /** Gets whether we are aligning frames in time. */
    bool get_time_alignment_setting() const;

    /**
     * Block until the Myriad X device is ready.
     */
    void wait_for_device();

    /**
     * Clear model storage before download any models
     */
    static void clear_model_storage();

    /** Returns the model's resolution. */
    cv::gapi::mx::Camera::Mode get_resolution() const;

    /** Set the model's resolution. The change will only be picked up after restarting the model pipeline. */
    void set_resolution(const cv::gapi::mx::Camera::Mode &res);

protected:
    /** Use `update_data_collection_params` to update this. If true, we should collect data and send it to the cloud. */
    bool data_collection_enabled = false;

    /** Use `update_data_collection_params` to update this. This is how often we collect an image for cloud-sending. */
    unsigned long int data_collection_interval_sec = 0;

    /** Path to the model file */
    std::vector<std::string> modelfiles = {};

    /** Location of the firmware file. */
    std::string mvcmd = "";

    /** If we want to record video, this will be a path to a file. Otherwise it is empty. */
    std::string videofile = "";

    /** The camera's resolution (needed for preprocessing) */
    cv::gapi::mx::Camera::Mode resolution = cv::gapi::mx::Camera::MODE_NATIVE;

    /** A status message to display on the RTSP feed */
    std::string status_msg = "";

    /** Should we align RTSP frames with inferences in time? */
    volatile bool align_frames_in_time = false;

    /** If this gets set to true, we stop the pipeline and return from the run function */
    volatile bool restarting = false;

    /** Cleanup after ourselves */
    void cleanup(cv::GStreamingCompiled &pipeline, const cv::Mat &last_bgr, int64_t last_bgr_timestamp);

    /** Write the H264 outputs to a file if videofile is non-empty and we have a result ready in the out_264 node. Also writes to the RTSP feed. */
    void handle_h264_output(cv::optional<std::vector<uint8_t>> &out_h264, const cv::optional<int64_t> &out_h264_ts, const cv::optional<int64_t> &out_h264_seqno, std::ofstream &ofs) const;

    /** Aligns the new inference in time with frames and calls the given lambda on each frame. Use this lambda to mark up the frames using this new inference. */
    void handle_new_inference_for_time_alignment(int64_t inference_ts, std::function<void(cv::Mat&)> f_to_apply_to_each_frame);

    /** Use adpative logging to log the inference message so that it does not pollute the log files */
    void log_inference(const std::string &msg);

    /** Save the retraining data if data collection is enabled, this frame lands on the right period, and the confidences are low enough to make it worthwhile. */
    void save_retraining_data(const cv::Mat &original_bgr, const std::vector<float> &confidences);

    /** Save the retraining data if data collection is enabled and this frame lands on the right period. */
    void save_retraining_data(const cv::Mat &original_bgr);

    /** Handles the streaming of frames over RTSP by dumping them into the server if we don't want to time-align, or by dumping them into a buffer if we do. */
    void stream_frames(const cv::Mat &raw_frame, const cv::Mat &result_frame, int64_t frame_ts);

private:
    /** Framebuffer of timestamped frames. */
    timebuf::TimeAlignedBuffer timestamped_frames;

    /** Timer for the data collection stuff. */
    ourtime::Timer data_collection_timer;

    /** Adaptive logger for inference messages. */
    util::AdaptiveLogger inference_logger;

    /** Load potentially several models from a single blob of data (say, if it is a URL that leads to a cascaded model in a .zip file). */
    static bool load(std::string &labelfile, const std::string &data, parser::Parser &modeltype, std::vector<std::string> &blob_files);

    /** Convert the .xml (or .onnx file) into a model.blob file and push the resulting .blob file path to blob_files. Returns true on success. Default option is .xml file */
    static bool convert_model(const std::string &modelfile, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files, bool is_xml = true);

    /** Download the model from url and then call `unzip_model` on the result. Returns true on success. */
    static bool download_model(const std::string &url, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files);

    /** Unzips the model and deals with whatever contents are in the .zip file. Returns true on success. */
    static bool unzip_model(const std::string &zippath, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files);

    /** Parse the given configuration file as part of loading the model. */
    static bool load_config(const std::string &configfpath, std::vector<std::string> &modelfiles, std::string &labelfile, parser::Parser &modeltype, bool ignore_modelfiles);

    /** Parse the given manifest file as part of loading the model. */
    static bool load_manifest(const std::string &manifestfpath, std::vector<std::string> &modelfiles, std::string &labelfile, parser::Parser &modeltype);
};

} // namespace model
