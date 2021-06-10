# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Some OpenCV integration nonsense
if(DEFINED OPENCV_INITIAL_PASS)
  return()
endif()

# CMake Project setup
cmake_minimum_required(VERSION 3.1)
project(InferenceApp)
set(CMAKE_CXX_STANDARD 11)

# Set the application name.
set(APP_NAME inference)

# Locate the OpenCV directory. The user or a higher level CMake
# file needs to provide the location of OpenCV, as it does not contain
# a FindOpenCV.cmake or OpenCVConfig.cmake file.
find_package(OpenCV REQUIRED gapi_azure highgui)

# Same for mxIf
find_package(mxIf REQUIRED)

# onnxruntime
find_library(ORT_LIB onnxruntime)

# Find PkgConfig dependency, which must contain GStreamer and RTSP server
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER gstreamer-1.0>=1.14 REQUIRED)
pkg_check_modules(RTSP gstreamer-rtsp-server-1.0 REQUIRED)

# Add USE_EDGE_MODULES flags to the C and CXX flags for the compiler
# (This is for Azure IoT C SDK)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_EDGE_MODULES")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_EDGE_MODULES")

# Here's where the Azure IoT C SDK headers are
set(AZUREIOT_INCLUDE_DIRS "/usr/local/include/azureiot")

# Slurp up all the .c* and .h* files for the application from the various locations
file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c*)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
file (GLOB DEVICE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/device/*.c*)
file (GLOB DEVICE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/device/*.h*)
file (GLOB IMG_CAPTURE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/imgcapture/*.c*)
file (GLOB IMG_CAPTURE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/imgcapture/*.h*)
file (GLOB KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.c*)
file (GLOB KERNEL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.h*)
file (GLOB MODEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/model/*.c*)
file (GLOB MODEL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/model/*.h*)
file (GLOB OCR_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ocr/*.h*)
file (GLOB OCR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ocr/*.c*)
file (GLOB OPEN_POSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/openpose/*.c*)
file (GLOB OPEN_POSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/openpose/*.h*)
file (GLOB OUR_IOT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/iot/*.c*)
file (GLOB OUR_IOT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/iot/*.h*)
file (GLOB SECURE_AI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/secure_ai/*.c*)
file (GLOB SECURE_AI_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/secure_ai/*.h*)
file (GLOB STREAMING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/streaming/*.c*)
file (GLOB STREAMING_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/streaming/*.h*)
file (GLOB UTIL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/util/*.c*)
file (GLOB UTIL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/util/*.h*)

##### REMOVE ME #############
add_definitions(-DDEBUG_TIME_ALIGNMENT)
##################################

# This is our target
add_executable(${APP_NAME}
  ${MAIN_SRC} ${MAIN_HEADERS}
  ${DEVICE_SRC} ${DEVICE_HEADERS}
  ${IMG_CAPTURE_SRC} ${IMG_CAPTURE_HEADERS}
  ${MODEL_SRC} ${MODEL_HEADERS}
  ${KERNEL_SRC} ${KERNEL_HEADERS}
  ${OCR_SRC} ${OCR_HEADERS}
  ${OPEN_POSE_SRC} ${OPEN_POSE_HEADERS}
  ${OUR_IOT_SRC} ${OUR_IOT_HEADERS}
  ${SECURE_AI_SRC} ${SECURE_AI_HEADERS}
  ${STREAMING_SRC} ${STREAMING_HEADERS}
  ${UTIL_SRC} ${UTIL_HEADERS}
)

# Add warnings (except the super annoying unused-parameter warning that makes callbacks difficult)
target_compile_options(${APP_NAME} PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)

# We need a few directories
target_include_directories(${APP_NAME}
  PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${RTSP_INCLUDE_DIRS}
    ${AZUREIOT_INCLUDE_DIRS}
)

# We need a bunch of shared libraries
target_link_libraries(${APP_NAME}
  PRIVATE
    ${OpenCV_LIBS}
    ${RTSP_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${ORT_LIB}
    gobject-2.0
    glib-2.0
    iothub_client
    iothub_client_mqtt_transport
    m
    mxIf
    prov_auth_client
    pthread
    udev
    umqtt
    aziotsharedutil
    parson
    hsm_security_client
    curl
    ssl
    crypto
    uhttp
    usb-1.0
    uuid
)
