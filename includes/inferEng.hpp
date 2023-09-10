/**
 * Inference Engine for YoLoV5
 * author@ vimal william 
 * e-mail@ vimalwilliam99@gmail.com 
*/

#pragma once 
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define INFERENG_HPP
#ifdef INFERENG_HPP

#include <opencv2/opencv.hpp>
#include <CL/opencl.hpp>

#include <iostream>
#include <string>
#include <filesystem>

class Infera{
    private:
        std::string BASE_DIR = std::filesystem::current_path(); 
        std::string model_path = BASE_DIR + "/models/yolov5s.onnx"; 

        bool is_loaded = false; 
        bool is_gpu = false; 

        /*opencl params*/
        std::vector<cl::Platform> platforms; 
        std::vector<cl::Device> devices; 

        cv::dnn::Net net = nullptr;

    public:
        bool infera_load_model(); 
};


#endif