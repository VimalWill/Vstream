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
    
    private:
        cv::dnn::Net net; 

};

#endif /*INFERA_HPP*/
