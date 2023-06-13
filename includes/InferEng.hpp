#ifndef INFERENG_HPP
#define INFERENG_HPP

//initalize the inference engine 
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

class GstInferaEng{
    private:
        const int InputWidth = 480; 
        const int InputHeight = 640;

    public:
        GstInferaEng()
        auto InferenceEngine(cv::Mat) -> cv::Mat; 
        auto Preproc(cv::Mat, const int, const int) -> cv::Mat; 
        auto Postproc(cv::Mat& outputImage) -> cv::Mat;

};

#endif /*INFERENG_HPP*/