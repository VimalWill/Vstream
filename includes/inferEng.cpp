#include <includes/InferEng.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

cv::Mat GstInferaEng::Preproc(cv::Mat& image, const int InputWidth, const int InputHeight){

    //Image preprocessing for YOLOv8 
    cv::Mat frame, frame_resize, processed_frame; 
    frame = image; 

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 
    cv::resize(frame, frame_resize, cv::Size(InputHeight, InputHeight)); 

    cv::Mat image_data; 
    frame_resize.convertTo(image_data, CV_32FC3, 1.0/255.0); 
    cv::transpose(image_data, image_data); 

    processed_frame = image_data.reshape(1, cv::Size(InputWidth, InputHeight)); 
    return processed_frame; 
}

cv::