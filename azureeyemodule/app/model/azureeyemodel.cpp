// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <fstream>
#include <parson.h>
#include <thread>

// Third party includes
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "parser.hpp"
#include "../device/device.hpp"
#include "../imgcapture/dataloop.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/timing.hpp"


namespace model {

AzureEyeModel::AzureEyeModel(const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : modelfiles(modelfpaths), mvcmd(mvcmd), videofile(videofile), resolution(resolution),
      timestamped_frames({cv::Mat(rtsp::DEFAULT_HEIGHT, rtsp::DEFAULT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0))}, 1577931508), // Thursday, January 2nd, 2020 for no great reason
      inference_logger({})
{
    // Nothing to do
}

AzureEyeModel::~AzureEyeModel()
{
}

void AzureEyeModel::set_update_flag()
{
    this->restarting = true;
}

void AzureEyeModel::update_time_alignment(bool enable)
{
    this->align_frames_in_time = enable;
}

void AzureEyeModel::update_data_collection_params(bool enable, unsigned long int interval)
{
    bool switched_on = (this->data_collection_enabled == false) && enable;

    this->data_collection_enabled = enable;
    this->data_collection_interval_sec = interval;

    if (switched_on)
    {
        this->data_collection_timer.reset();
    }
}

void AzureEyeModel::get_data_collection_params(bool &enable, unsigned long int &interval) const
{
    enable = this->data_collection_enabled;
    interval = this->data_collection_interval_sec;
}

bool AzureEyeModel::get_time_alignment_setting() const
{
    return this->align_frames_in_time;
}

void AzureEyeModel::wait_for_device()
{
    while (!device::open_device())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void AzureEyeModel::log_inference(const std::string &msg)
{
    this->inference_logger.log_info(msg);
}

void AzureEyeModel::save_retraining_data(const cv::Mat &bgr, const std::vector<float> &confidences)
{
    // Check whether the timer has gone over the collection interval.
    // Note that the timer has double precision and deals in seconds, so an overflow would be pretty unlikely,
    // you know, since 4,294,967,295 seconds equals ~136 years.
    bool time_to_collect = (unsigned long int)this->data_collection_timer.elapsed() > this->data_collection_interval_sec;
    if (this->data_collection_enabled && time_to_collect)
    {
        auto n = confidences.size();
        if (n != 0)
        {
            auto average_confidence = accumulate(confidences.begin(), confidences.end(), 0.0) / n;
            auto snapshot = std::to_string((int)(average_confidence * 100)) + "-" + ourtime::get_timestamp() + ".jpg";
            loop::write_data_file(snapshot, bgr);
        }

        this->data_collection_timer.reset();
    }
}

void AzureEyeModel::save_retraining_data(const cv::Mat &bgr)
{
    this->save_retraining_data(bgr, {1.0f});
}

void AzureEyeModel::stream_frames(const cv::Mat &raw_frame, const cv::Mat &result_frame, int64_t frame_ts)
{
    cv::Mat new_raw_frame;
    cv::Mat new_result_frame;
    if (this->status_msg.empty())
    {
        // No copy
        new_raw_frame = raw_frame;
        new_result_frame = result_frame;
    }
    else
    {
        // Copy the frames so we can mark them up with a status message
        raw_frame.copyTo(new_raw_frame);
        result_frame.copyTo(new_result_frame);

        util::put_text(new_raw_frame, this->status_msg);
        util::put_text(new_result_frame, this->status_msg);
    }

    if (this->align_frames_in_time)
    {
        // Don't add a status message to the frames that we are keeping back for time alignment,
        // as it could be confusing to someone watching the stream, since the stream will
        // be delayed by some amount of time.
        cv::Mat frame_to_mark_up_later;
        raw_frame.copyTo(frame_to_mark_up_later);
        this->timestamped_frames.put(std::make_tuple(frame_to_mark_up_later, frame_ts));

        // Add status message to the raw frame that we send out right now though.
        rtsp::update_data_raw(new_raw_frame, frame_ts);
    }
    else
    {
        rtsp::update_data_raw(new_raw_frame, frame_ts);
        rtsp::update_data_result(new_result_frame, frame_ts);
    }
}

bool AzureEyeModel::load(std::string &labelfile, std::vector<std::string> &modelfiles, parser::Parser &modeltype)
{
    std::vector<std::string> resulting_blob_files;

    // Loop over all the data items we have in `modelfiles`, converting each one into
    // potentially several .blob files. Each of those .blob files gets pushed into a temporary result vector,
    // which then overwrites the `modelfiles` vector.
    bool parsed_everything = true;
    for (const auto &modelfile : modelfiles)
    {
        std::vector<std::string> blob_files;
        parsed_everything = parsed_everything && load(labelfile, modelfile, modeltype, blob_files);

        // Now update the result vector
        for (auto &blob : blob_files)
        {
            resulting_blob_files.push_back(std::move(blob));
        }
    }

    // Now overwrite the strings with what we actually want them to be (.blob file paths)
    modelfiles = std::move(resulting_blob_files);

    return parsed_everything;
}

bool AzureEyeModel::load(std::string &labelfile, const std::string &data, parser::Parser &modeltype, std::vector<std::string> &blob_files)
{
    if ((std::string::npos != data.find("https://")) || (std::string::npos != data.find("http://")))
    {
        return download_model(data, labelfile, modeltype, blob_files);
    }
    else if (std::string::npos != data.find(".zip"))
    {
        return unzip_model(data, labelfile, modeltype, blob_files);
    }
    else if (std::string::npos != data.find(".xml"))
    {
        return convert_model(data, labelfile, modeltype, blob_files);
    }
    else if (std::string::npos != data.find(".blob"))
    {
        // Make sure the file exists
        if (!util::file_exists(data))
        {
            util::log_error("Could not find the given file at path: " + data);
            return false;
        }
        else
        {
            blob_files.push_back(data);
            return true;
        }
    }
    else if (std::string::npos != data.find(".onnx"))
    {
        // Make sure the file exists
        if (!util::file_exists(data))
        {
            util::log_error("Could not find the given file at path: " + data);
            return false;
        }
        else
        {
            if(modeltype == parser::Parser::ONNXSSD)
            {
                blob_files.push_back(data);
                return true;
            }
            else
            {
                return convert_model(data, labelfile, modeltype, blob_files, false);
            }
        }
    }
    else
    {
        util::log_error("Unrecognized string format for model data. Given a URL or zip or XML or blob file of path: " + data);
        return false;
    }
}

bool AzureEyeModel::convert_model(const std::string &modelfile, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files, bool is_xml /*= true*/)
{
    // Get the modelfile path without the .xml or .onnx extension
    std::string modelfile_no_extension = (is_xml == true) ? modelfile.substr(0, modelfile.size() - 4) : modelfile.substr(0, modelfile.size() - 5);

    // Strip any leading "/app/model/" off of it so we should just have a base name now
    const std::string modelstr = "/app/model/";
    std::string modelfile_lrstripped = modelfile_no_extension.substr(modelfile_no_extension.find_last_of('/') + 1);
    std::string result_location = modelstr + modelfile_lrstripped + ".blob";

    // To run myriad_compile, the pipeline is needed to be running, that is, pipeline.start();
    // Not sure if this is by design or a bug from Intel. But we simply follow the rule that
    // pipeline stops after the myriad_compile is done.
    //
    // Important note: for ONNX fall back to CPU case, as it doesn't use the pipeline,
    // the pipeline hasn't been started. Thus myriad_compile will report error.
    // It will happen with the following scenario.
    // If user runs ONNX fall back to CPU case followed by downloading an IR(Intermediate Representation) format,
    // he/she will hit this error.
    //
    // Possible solution to be done: before myriad_compile, check if the pipeline is running.
    // May need Intel to provide additional API to check if the pipeline is running.
    const std::string executable = (is_xml == true) ? "/openvino/bin/aarch64/Release/myriad_compile" : "/openvino/bin/aarch64/Release/custom_myriad_compile";
    int ret = util::run_command(executable + (" \
                     -m " + modelfile + " \
                     -ip U8 \
                     -VPU_NUMBER_OF_SHAVES 8 \
                     -VPU_NUMBER_OF_CMX_SLICES 8 \
                     -o " + result_location + "\
                     -op FP32").c_str());

    if (ret != 0)
    {
        util::log_error("myriad_compile failed with " + ret);
        return false;
    }

    blob_files.push_back(result_location);
    return true;
}

void AzureEyeModel::clear_model_storage()
{
    int ret = util::run_command("rm -rf /app/model && mkdir /app/model");
    if (ret != 0)
    {
        util::log_error("rm && mkdir failed with " + ret);
    }
}

bool AzureEyeModel::download_model(const std::string &url, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files)
{
    const std::string zippath = "/app/model/model.zip";
    int ret = util::run_command(("wget --no-check-certificate -O " + zippath + " \"" + url + "\"").c_str());
    if (ret != 0)
    {
        util::log_error("wget failed with " + ret);
        return false;
    }

    return unzip_model(zippath, labelfile, modeltype, blob_files);
}

bool AzureEyeModel::unzip_model(const std::string &zippath, std::string &labelfile, parser::Parser &modeltype, std::vector<std::string> &blob_files)
{
    // Unzip the archive
    int ret = util::run_command(("unzip -o \"" + zippath + "\" -d /app/model").c_str());
    if (ret != 0)
    {
        util::log_error("unzip failed with " + ret);
        return false;
    }

    // Unzipping succeeded. Look at the contents and decide what to do from that.
    std::vector<std::string> modelfiles;
    if (util::file_exists("/app/model/config.json"))
    {
        bool worked = load_config("/app/model/config.json", modelfiles, labelfile, modeltype, false);
        if (!worked)
        {
            return false;
        }
    }
    else if (util::file_exists("/app/model/cvexport.manifest"))
    {
        bool worked = load_manifest("/app/model/cvexport.manifest", modelfiles, labelfile, modeltype);
        if (!worked)
        {
            return false;
        }
    }
    else if (util::file_exists("/app/model/model.blob"))
    {
        modelfiles = {"/app/model/model.blob"};
    }
    else
    {
        util::log_error("No config.json, cvexport.manifest, or model.blob file found in the zip archive.");
        return false;
    }

    // For each model file, if it is a .xml or .onnx file, we have to convert it to a .blob file.
    for (const auto &modelfile : modelfiles)
    {
        if ((modelfile.size() > 4) && modelfile.substr(modelfile.size() - 4, 4) == ".xml")
        {
            bool worked = convert_model(modelfile, labelfile, modeltype, blob_files);
            if (!worked)
            {
                util::log_error("Could not convert " + modelfile);
                return false;
            }
        }
        if ((modelfile.size() > 5) && modelfile.substr(modelfile.size() - 5, 5) == ".onnx")
        {
            if(modeltype == parser::Parser::ONNXSSD)
            {
                blob_files.push_back(modelfile);
            }
            else
            {
                bool worked = convert_model(modelfile, labelfile, modeltype, blob_files, false);
                if (!worked)
                {
                    util::log_error("Could not convert " + modelfile);
                    return false;
                }
            }
        }
        else
        {
            blob_files.push_back(modelfile);
        }
    }

    return true;
}

bool AzureEyeModel::load_config(const std::string &configfpath, std::vector<std::string> &modelfiles, std::string &labelfile, parser::Parser &modeltype, bool ignore_modelfiles)
{
    JSON_Value *root_value = json_parse_file(configfpath.c_str());
    JSON_Object *root_object = json_value_get_object(root_value);

    if (!ignore_modelfiles && (json_object_get_value(root_object, "ModelFileName") != NULL))
    {
        std::vector<std::string> tmp = util::splice_comma_separated_list(json_object_get_string(root_object, "ModelFileName"));
        for (const auto &modelfile : tmp)
        {
            modelfiles.push_back("/app/model/" + modelfile);
        }
    }
    else if (!ignore_modelfiles)
    {
        util::log_error("JSON file does not contain a 'ModelFileName' property.");
        return false;
    }

    if (json_object_get_value(root_object, "DomainType") != NULL)
    {
        modeltype = parser::from_string(util::to_lower(json_object_get_string(root_object, "DomainType")));
    }
    else
    {
        util::log_error("JSON file does not contain a 'DomainType' property.");
        return false;
    }

    if (json_object_get_value(root_object, "LabelFileName") != NULL)
    {
        labelfile = "/app/model/" + std::string(json_object_get_string(root_object, "LabelFileName"));
    }

    return true;
}

bool AzureEyeModel::load_manifest(const std::string &manifestfpath, std::vector<std::string> &modelfiles, std::string &labelfile, parser::Parser &modeltype)
{
    bool worked = load_config("/app/model/cvexport.manifest", modelfiles, labelfile, modeltype, true);
    if (!worked)
    {
        return false;
    }

    // cvexport.manifest files can specify "objectdetection", and if so, we need to further decide if that
    // is S1 or YOLO
    if (modeltype == parser::Parser::OBJECT_DETECTION)
    {
        if (util::search_keyword_in_file("mobilenetv2ssdlitev2_pytorch", "/app/model/model.xml") || util::search_keyword_in_file("compact_od_s1_v2", "/app/model/model.xml"))
        {
            modeltype = parser::Parser::S1;
        }
        else
        {
            modeltype = parser::Parser::YOLO;
        }
    }

    int ret = util::run_command("python3 /app/update_cvs_openvino.py /app/model/model.xml /app/model/model.bin /app/model/out.bin && mv /app/model/out.bin /app/model/model.bin");
    if (ret != 0)
    {
        util::log_error("Update_cvs_openvino.py && mv failed with " + ret);
        return false;
    }

    // Update the model file to point to the .xml file
    modelfiles.push_back("/app/model/model.xml");

    return true;
}

void AzureEyeModel::cleanup(cv::GStreamingCompiled &pipeline, const cv::Mat &last_bgr, int64_t last_timestamp)
{
    this->restarting = false;

    util::put_text(last_bgr, "Loading Model");
    rtsp::update_data_result(last_bgr, last_timestamp);
}

void AzureEyeModel::handle_h264_output(cv::optional<std::vector<uint8_t>> &out_h264, const cv::optional<int64_t> &out_h264_ts,
                                       const cv::optional<int64_t> &out_h264_seqno, std::ofstream &ofs) const
{
    if (!out_h264.has_value())
    {
        return;
    }

    CV_Assert(out_h264_seqno.has_value());
    CV_Assert(out_h264_ts.has_value());

    if (ofs.is_open())
    {
        ofs.write(reinterpret_cast<const char*>(out_h264->data()), out_h264->size());
    }

    rtsp::H264 frame;
    frame.data = *out_h264;
    frame.timestamp = *out_h264_ts;

    rtsp::update_data_h264(frame);
}

void AzureEyeModel::handle_new_inference_for_time_alignment(int64_t inference_ts, std::function<void(cv::Mat&)> f_to_apply_to_each_frame)
{
    if (!this->align_frames_in_time)
    {
        return;
    }

    #ifdef DEBUG_TIME_ALIGNMENT
        util::log_debug("New Inference: Drawing on frames from " + util::timestamp_to_string(inference_ts) + " and older.");
    #endif

    // Find all the frames that are either time-aligned or older than the time-aligned frame
    // (and remove them from the buffer)
    std::vector<cv::Mat> frames_to_draw_on;
    std::vector<int64_t> timestamps;
    this->timestamped_frames.get_best_match_and_older(inference_ts, frames_to_draw_on, timestamps);

    // Draw our bounding boxes (or masks, or whatever) over each one and release them in a batch to the RTSP server
    for (auto &frame : frames_to_draw_on)
    {
        f_to_apply_to_each_frame(frame);
        if (!this->status_msg.empty())
        {
            util::put_text(frame, this->status_msg);
        }
    }

    #ifdef DEBUG_TIME_ALIGNMENT
        util::log_debug("New Inference: Sending " + std::to_string(frames_to_draw_on.size()) + " to RTSP stream");
    #endif
    rtsp::update_data_result(frames_to_draw_on, timestamps);
}

cv::gapi::mx::Camera::Mode AzureEyeModel::get_resolution() const
{
    return this->resolution;
}

void AzureEyeModel::set_resolution(const cv::gapi::mx::Camera::Mode &res)
{
    this->resolution = res;
}

} // namespace model
