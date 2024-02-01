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
#include <random>

class neural_engine{

    public:
        neural_engine(std::string path); 
        bool load_model(); 
        cv::Mat format2sq(cv::Mat& source); 
        cv::Mat detect(cv::Mat& img); 

        ~neural_engine(); 
    
    private:
        cv::dnn::Net net; 
        std::string model_path; 

        int INPUT_HEIGHT = 640; 
        int INPUT_WIDTH = 640;

        std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

        struct Detection
        {
            int class_id; 
            float confidence; 
            cv::Rect box; 
            cv::Scalar color{};
            std::string className{};
        };
        
};

#endif /*INFERA_HPP*/
