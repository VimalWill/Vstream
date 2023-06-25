#pragma once
#ifndef INFERENG_HPP
#define INFERENG_HPP


#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <bits/stdc++.h>


class GstAppSrcInfer{
    private:
        std::string BASEDIR = std::filesystem::current_path(); 
        std::string modelPath = BASEDIR + "/yolo_nas_s_coco.onnx"; 
        std::string instance {"Gstreamer-Appsrc-InferenceEngine"};

        Ort::Env env{nullptr}; 
        Ort::Session session{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};

        std::vector<std::vector<int64_t>> input_node_dims;
        std::vector<std::vector<int64_t>> output_node_dims;

        std::vector<std::string> InputNames; 
        std::vector<std::string> OuputNames;
    
    public:
        GstAppSrcInfer(); 

        void InferenceEngine(cv::Mat& frame); 
        void GetInputDetails(Ort::AllocatorWithDefaultOptions allocator); 
        void GetOutputDetails(Ort::AllocatorWithDefaultOptions allocator); 
};

#endif /*INFERENG_HPP*/
