#include <opencv2/opencv.hpp>
#include <includes/InferEng.hpp>

cv::Mat GstInferaEng::InferenceEngine(cv::Mat image){

    //image preprocessing
    cv::Mat frame; 
    cv::resize(image, frame, (240, 240)); 

    
}