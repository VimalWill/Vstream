#ifndef INFERENG_HPP
#define INFERENG_HPP

//initalize the inference engine 
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

class GstInferaEng{
    public:
        auto InferenceEngine(cv::Mat) -> cv::Mat; 

};

#endif /*INFERENG_HPP*/