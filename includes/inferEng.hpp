#ifndef INFERENG_HPP
#define INFERENG_HPP

//initalize the inference engine 
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

class GstInferaEng{
    private:
        int InputWidth; 
        int InputHeight;

        cv::Mat Input_img;

    public:
        GstInferaEng(cv::Mat& image);


        auto InferenceEngine() -> cv::Mat; 
        auto Preproc(cv::Mat) -> std::vector<float>;
        auto Postproc(cv::Mat& outputImage) -> cv::Mat;
        void letterBox(cv::Mat& Input_image, cv::Mat& OutputImage);

};

#endif /*INFERENG_HPP*/