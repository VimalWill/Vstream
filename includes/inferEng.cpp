#include <opencv2/opencv.hpp>
#include <includes/InferEng.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

cv::Mat GstInferaEng::InferenceEngine(cv::Mat &input_image){

    cv::Mat frame = input_image; 
    
}

cv::Mat GstInferaEng::Preproc(cv::Mat &frame){
    cv::Mat image = frame; 

    //color conversion and resize
    cv::Mat resized_img, processed_img;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); 
    cv::resize(image, resized_img, cv::Size(480, 640)); 

    //normalization
    


}