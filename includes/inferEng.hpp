/**
 * Inference Engine for Vstream
 * author@ vimal william 
 * e-mail@ vimalwilliam99@gmail.com 
*/

#pragma once 
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define INFERENG_HPP
#ifdef  INFERENG_HPP

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <CL/opencl.hpp>

#include <iostream>
#include <string>
#include <filesystem>

class Infera{
    private:

        bool is_loaded = false; 
        bool is_gpu = false; 

        /*opencl params*/
        std::vector<cl::Platform> platforms; 
        std::vector<cl::Device> devices; 

        cv::dnn::Net net = nullptr;

    public:
        bool infera_load_model(); 
        void neural_engine(cv::Mat& img); 
};


#endif /*INFERENG_HPP*/