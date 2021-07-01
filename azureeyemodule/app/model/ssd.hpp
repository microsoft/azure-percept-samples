// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard library includes
#include <string>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

// Local includes
#include "objectdetector.hpp"


namespace model {

/**
 * A class to represent a Single Shot Detector model for the Azure Percept device.
 */
class SSDModel : public ObjectDetector
{
public:
    /**
     * Constructor.
     *
     * @param labelfpath: Path to a file containing one label per line. The labels should correspond to the class index
     *                    such that the first line should contain the label for the first class index.
     * @param modelfpaths: In this model, we expect this to be a vector of length 1 (all model constructors take a vector of paths, but most only use 1 model path).
     *                     Model paths are strings that point to the model .blob files. If you are developing a cascaded model (see ocr for example), you will
     *                     need to pass a model file for each network in the cascade.
     * @param mvcmd:       The Myriad X firmware binary. A running model owns the VPU, so it is responsible for initializing it as well.
     * @param videofile:   If we pass a non-empty string here, the model will record frames to a file at this location as a .mp4.
     * @param resolution:  The resolution mode to put the camera in.
     */
    SSDModel(const std::string &labelfpath, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string inputsource, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    /**
     * The SSD model acts as our default model in case we don't get direction
     * one way or another or need to fall back to a working model.
     */
    void load_default();

    /** We invoke this method from main to run the network. It should return if there is a model update (found by checking this->restarting). */
    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Compile the pipeline graph for SSD when MIPI camera is input source. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Compile the pipeline graph for SSD when either uvc camera or video file is input source. The steps inside are slightly different from the inbox MIPI camera */
    cv::GStreamingCompiled compile_cv_graph_uvc_video() const;

    /** Print out all the model's meta information. */
    void log_parameters() const;
    std::string inputsource;
};

} // namespace model
