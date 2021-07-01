// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <map>
#include <pthread.h>
#include <signal.h>
#include <string>
#include <vector>
#include <cstdlib>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <VPUTempRead.hpp>

// Local includes
#include "device/device.hpp"
#include "device/validator.h"
#include "imgcapture/dataloop.hpp"
#include "iot/iot_interface.hpp"
#include "iot/iot_update.hpp"
#include "model/azureeyemodel.hpp"
#include "model/classification.hpp"
#include "model/binaryunet.hpp"
#include "model/fasterrcnn.hpp"
#include "model/openpose.hpp"
#include "model/ocr.hpp"
#include "model/parser.hpp"
#include "model/s1.hpp"
#include "model/ssd.hpp"
#include "model/yolo.hpp"
#include "model/onnxssd.hpp"
#include "secure_ai/secureai.hpp"
#include "streaming/rtsp.hpp"
#include "util/helper.hpp"

const std::string keys =
"{ h help      |        | print this message }"
"{ f mvcmd     |        | mvcmd firmware }"
"{ h264_out    |        | Output file name for the raw H264 stream. No files written by default }"
"{ l label     |        | label file }"
"{ m model     |        | model zip file }"
"{ q quit      | false  | If given, we quit on error, rather than loading a default model. Useful for testing }"
"{ p parser    | ssd100 | Parser kind required for input model. Possible values: ssd100, ssd200, yolo, classification, s1, openpose, onnxssd, faster-rcnn-resnet50, unet, ocr }"
"{ s size      | native | Output video resolution. Possible values: native, 1080p, 720p }"
"{ t timealign | false  | Align the RTSP result frames with their corresponding neural network outputs in time }"
"{ fps         | 10     | Output video frame rate. }"
"{ i input     |        | Source of input frames. Inbox MIPI camera attached to Eye SoM by default. Possible value: uvc, video:<video path in the container> }";

const std::map<rtsp::Resolution, cv::gapi::mx::Camera::Mode> modes = {
    {rtsp::Resolution::NATIVE, cv::gapi::mx::Camera::MODE_NATIVE},
    {rtsp::Resolution::HD1080P, cv::gapi::mx::Camera::MODE_1080P},
    {rtsp::Resolution::HD720P, cv::gapi::mx::Camera::MODE_720P} };

/** Pointer to the single model we have at a time. */
static model::AzureEyeModel *the_model = nullptr;

/** If we update our model, this is what we will use to update it to. */
static std::string update_model_data = "";

/** Update using the secure AI lifecycle stuff. */
static bool update_using_secure_ai = false;

/** We tell the model to stop running and return so we can update it. */
static void update_model(const std::string &data, bool secure)
{
    if (the_model == nullptr)
    {
        util::log_error("Trying to update the model before we have a model to update.");
        return;
    }
    else if ((data == update_model_data) && (update_using_secure_ai == secure))
    {
        util::log_info("Foregoing model update, as the model meta data has not changed.");
        return;
    }

    util::log_info("update data: " + data + ", secure: " + (secure ? "true" : "false"));
    update_using_secure_ai = secure;
    update_model_data = data;
    the_model->set_update_flag();
}

/** We tell the model to update its data collection parameters; this is for the retraining loop */
static void update_data_collection_params(bool enable, unsigned long int interval_seconds)
{
    if (the_model == nullptr)
    {
        util::log_error("Trying to update the model's data collection params before we have a model.");
        return;
    }

    util::log_info("Update data collection params: (enable: " + std::string((enable ? "true" : "false")) + ", interval (s): " + std::to_string(interval_seconds) + ")");
    the_model->update_data_collection_params(enable, interval_seconds);
}

/** This function gets called when we update the resolution. */
static void update_resolution(const rtsp::Resolution &resolution)
{
    if (the_model == nullptr)
    {
        util::log_error("Trying to update the model's resolution before we have a model.");
        return;
    }

    util::log_info("Update resolution callback called with \"" + rtsp::resolution_to_string(resolution) + "\"");
    the_model->set_resolution(modes.at(resolution));
    the_model->set_update_flag();
}

/** This function gets called when we update the time alignment feature in the module twin. */
static void update_time_alignment(bool align)
{
    if (the_model == nullptr)
    {
        util::log_error("Trying to update the model's time alignment feature before we have a model.");
        return;
    }

    util::log_info("Update the time alignment to: " + (align ? std::string("yes") : std::string("no")));
    the_model->update_time_alignment(align);
}

/** On a signal, we clean up after ourselves and exit cleanly. */
static void interrupt(int sig)
{
    util::log_info("received interrupt signal");

    stop_validator();
    iot::msgs::stop_iot();

    exit(0);
}

static void determine_model_type(const std::string &labelfile, const std::vector<std::string> &modelfiles, const std::string &mvcmd,
                                 const std::string inputsource, const std::string &videofile, const model::parser::Parser &parser_type, 
                                 const cv::gapi::mx::Camera::Mode &resolution, bool quit_on_failure)
{
    the_model = nullptr;
    switch (parser_type)
    {
        case model::parser::Parser::CLASSIFICATION:
            the_model = new model::ClassificationModel(labelfile, modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::OPENPOSE:
            the_model = new model::OpenPoseModel(modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::OCR:
            the_model = new model::OCRModel(modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::S1:
            the_model = new model::S1Model(labelfile, modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::SSD100: // fall-through
        case model::parser::Parser::SSD200: // fall-through
        case model::parser::Parser::DEFAULT:
            the_model = new model::SSDModel(labelfile, modelfiles, mvcmd, inputsource, videofile, resolution);
            break;
        case model::parser::Parser::YOLO:
            the_model = new model::YoloModel(labelfile, modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::ONNXSSD:
            the_model = new model::ONNXSSDModel(labelfile, modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::UNET:
            the_model = new model::BinaryUnetModel(modelfiles, mvcmd, videofile, resolution);
            break;
        case model::parser::Parser::FASTER_RCNN_RESNET50:
            the_model = new model::FasterRCNNModel(labelfile, modelfiles, mvcmd, videofile, resolution);
            break;
        default:
            util::log_error("No parser for the given model type: " + model::parser::to_string(parser_type));
            exit(__LINE__);
    }

    if ((parser_type == model::parser::Parser::DEFAULT) && quit_on_failure)
    {
        // If we are a default model, but supposed to quit on error, we quit.
        util::log_error("Quitting on failure to load appropriate type of model, due to --quit being passed.");
        exit(__LINE__);
    }
    else if (parser_type == model::parser::Parser::DEFAULT)
    {
        // If we are a default model, we need to load the defaults
        static_cast<model::SSDModel *>(the_model)->load_default();
    }
}

/** This function should update the_model to be whatever we've been told to update to via the update callback. */
static void load_new_model(const std::string &mvcmd, const std::string inputsource, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution, bool quit_on_failure)
{
    std::string labelfile = "";
    std::vector<std::string> modelfiles;
    model::parser::Parser modeltype;
    bool worked = true;

    model::AzureEyeModel::clear_model_storage();

    if (update_using_secure_ai)
    {
        // Try downloading and decrypting the model using secure AI
        worked = secure::download_model(modelfiles, update_model_data);
        if (!worked)
        {
            util::log_error("Could not download model using secure AI lifecycle. Loading a default model instead.");
            modeltype = model::parser::Parser::DEFAULT;
        }
    }
    else
    {
        // Nothing special needs to be done
        modelfiles = {update_model_data};
    }

    // If we failed to download using the secure model download, we skip trying to load the model here
    if (worked)
    {
        worked = model::AzureEyeModel::load(labelfile, modelfiles, modeltype);
        if (!worked)
        {
            util::log_error("Could not load the desired type of model. Using a default one instead.");
            modeltype = model::parser::Parser::DEFAULT;
        }
    }

    // Fill in the model values based on the type
    determine_model_type(labelfile, modelfiles, mvcmd, inputsource, videofile, modeltype, resolution, quit_on_failure);
}

/** This function stops the MyriadX pipeline and wait for 2 seconds as Intel suggested */
static void stop_pipeline(cv::GStreamingCompiled* pipeline)
{
    util::log_info("stopping the pipeline...");
    pipeline->stop();
    // Sleep some time to let the device properly
    // deregister in the system
    // TODO: Is there a real way to do this? Like by calling some driver function?
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main(int argc, char** argv)
{
    // Print the version
    util::version();

    // Set up a signal callback for SIGINT so we can gracefully close the application
    signal(SIGINT, interrupt);

    // Set up the model update callback
    iot::update::set_update_callback(&update_model);

    // Set up the retraining loop callback
    iot::update::set_update_collection_params_callback(&update_data_collection_params);

    // Set up the resolution change callback
    iot::update::set_update_resolution_callback(&update_resolution);

    // Set up the time alignment change callback
    iot::update::set_update_time_alignment_callback(&update_time_alignment);

    // Parse out the arguments
    cv::CommandLineParser cmd(argc, argv, keys);

    if (cmd.has("help"))
    {
        cmd.printMessage();
        return 0;
    }

    auto labelfile = cmd.get<std::string>("label");
    auto modelfiles = util::splice_comma_separated_list(cmd.get<std::string>("model"));
    auto mvcmd = cmd.get<std::string>("mvcmd") != "" ? cmd.get<std::string>("mvcmd") : "/eyesom/mx.mvcmd";
    auto videofile = cmd.get<std::string>("h264_out");
    auto parser_type = model::parser::from_string(cmd.get<std::string>("parser"));
    auto str_resolution = cmd.get<std::string>("size");
    auto quit_on_failure = cmd.get<bool>("quit");
    auto timealign = cmd.get<bool>("timealign");
    auto fps = cmd.get<int>("fps");
    auto inputsource = cmd.get<std::string>("input");

    // Sanity check resolution is allowed
    if (!rtsp::is_valid_resolution(str_resolution))
    {
        util::log_error("Given a resolution that is not allowed: " + str_resolution);
        exit(__LINE__);
    }
    auto resolution_camera_mode = modes.at(rtsp::resolution_string_to_enum(str_resolution));

    // Sanity check the labelfile exists (if given)
    if ((labelfile != "") && !util::file_exists(labelfile))
    {
        util::log_error("Label file given does not seem to be a valid path: " + labelfile);
        exit(__LINE__);
    }

    // Sanity check the mvcmd file exists as well
    if (!util::file_exists(mvcmd))
    {
        util::log_error("Given Azure Percept firmware file that does not seem to be a valid path: " + mvcmd);
        exit(__LINE__);
    }

    // Sanity check the FPS makes sense
    if (fps <= 0)
    {
        util::log_error("Given an FPS that does not make sense. Should be > 0, but is " + std::to_string(fps));
        exit(__LINE__);
    }

    // Now possibly overwrite some of these parameters based on what we find in modelfiles
    bool loaded = model::AzureEyeModel::load(labelfile, modelfiles, parser_type);
    if (!loaded)
    {
        util::log_error("Could not load the desired type of model. Using a default one instead.");
        parser_type = model::parser::Parser::DEFAULT;
    }

    // Fill in `the_model` with the appropriate type of model
    determine_model_type(labelfile, modelfiles, mvcmd, inputsource, videofile, parser_type, resolution_camera_mode, quit_on_failure);

    // See if the device is already opened, if not, open it and authenticate
    bool opened_usb_device = device::open_device();
    if (!opened_usb_device)
    {
        device::authenticate_device();
    }

    // Create RTSP thread
    auto stream_resolution = rtsp::resolution_string_to_enum(str_resolution);
    rtsp::set_stream_params(rtsp::StreamType::RAW, stream_resolution, fps, true);
    rtsp::set_stream_params(rtsp::StreamType::RESULT, stream_resolution, fps, true);
    rtsp::set_stream_params(rtsp::StreamType::H264_RAW, stream_resolution, fps, true);
    pthread_t thread_rtsp;
    if (pthread_create(&thread_rtsp, NULL, rtsp::gst_rtsp_server_thread, NULL))
    {
        util::log_error("pthread_create(&thread_rtsp, NULL, gst_rtsp_server_thread, NULL) failed.");
        return __LINE__;
    }
    util::log_info("RTSP thread created.");

    // Set whether we want to time-align the inferences (this can be overridden by Module Twin)
    the_model->update_time_alignment(timealign);

    // Create data uploading thread
    pthread_t thread_data_collection;
    if (pthread_create(&thread_data_collection, NULL, loop::export_data, NULL))
    {
        util::log_error("pthread_create(thread_data_collection, NULL, export_data, NULL) failed.");
        return __LINE__;
    }
    util::log_info("Data collection thread created.");

    // Start the IoT SDK stuff
    iot::msgs::start_iot();

    bool data_collection_enabled = false;
    unsigned long int data_collection_interval_sec = 0;

    // Main loop
    while (true)
    {
        // Run until the model is told to stop via the model update callback
        cv::GStreamingCompiled pipeline;
        the_model->run(&pipeline);

        // Save data collection settings
        the_model->get_data_collection_params(data_collection_enabled, data_collection_interval_sec);

        // The resolution may have changed
        resolution_camera_mode = the_model->get_resolution();

        // The time alignment settings may have changed
        timealign = the_model->get_time_alignment_setting();

        // Clean up after ourselves
        delete the_model;
        the_model = nullptr;

        // Load a new model
        load_new_model(mvcmd, inputsource, videofile, resolution_camera_mode, quit_on_failure);

        // Update data collection settings
        update_data_collection_params(data_collection_enabled, data_collection_interval_sec);

        // Update the time alignment settings
        the_model->update_time_alignment(timealign);

        // Stop the MyriadX pipeline. Note only after the Model conversion can the pipeline be stopped. Could be a bug from Intel or by design.
        stop_pipeline(&pipeline);
    }

    iot::msgs::stop_iot();
    return 0;
}
