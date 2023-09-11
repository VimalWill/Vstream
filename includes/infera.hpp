/**
 * Infera - neural engine for Vstream 
 * author@ vimal william 
 * email@ vimlwiliam99@gmail.com
*/

#pragma once 
#ifndef INFERA_HPP
#define INFERA_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <CL/opencl.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class neural_engine{

    public:
        bool load_model(); 
        cv::Mat format2sq(cv::Mat& source); 
        cv::Mat detect(cv::Mat& img); 
    
    private:
        cv::dnn::Net net; 

        int INPUT_ROW = 640; 
        int INPUT_COL = 640;

};

#endif /*INFERA_HPP*/
