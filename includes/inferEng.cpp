#include <includes/InferEng.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

GstInferaEng::GstInferaEng(cv::Mat& Image){
    Input_img = Image;
}

cv::Mat GstInferaEng::InferenceEngine(){

    std::string instanceName{"Gst-Object-Detection"};
    std::string ModelPath{"/home/vimal/Edge AI/Vstream/model/yolov8n.onnx"};

    //initalize the onnxruntime 
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str()); 
    Ort::SessionOptions sessionOptions; 
    sessionOptions.AppendExecutionProvider_CUDA(); 

    //initalize the session
    Ort::Session session(env, ModelPath, sessionOptions); 



    
}

cv::Mat GstInferaEng::Preproc(cv::Mat& image){

    //Image preprocessing for YOLOv8 
    cv::Mat frame, frame_resize, processed_frame; 
    frame = image; 

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 
    cv::resize(frame, frame_resize, cv::Size(InputWidth, InputHeight)); 

    cv::Mat image_data; 
    frame_resize.convertTo(image_data, CV_32FC3, 1.0/255.0); 
    cv::transpose(image_data, image_data); 

    processed_frame = image_data.reshape(1, cv::Size(InputWidth, InputHeight)); 
    return processed_frame; 
}

cv::Mat GstInferaEng::Postproc(cv::Mat& outputImage){

    //Image Post-processing includes Non-maximal supression and overlay function
    cv::Mat frame = outputImage; 
    frame = frame.reshape(0, {0});
    cv::transpose(frame, frame); 

    //get the no. of rows
    int rows = frame.rows; 
    
    //calculate the scaling factor 



}