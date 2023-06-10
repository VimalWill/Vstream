/*Pre - processing unit: acts an Inference Engine which handles
model deployment with onnx runtime, post processing the meta data
with the frame*/

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

class InferenceEngine{
    private:
        std::string ModelPath; 
        std::string HW_CLASS; 
    
    public:
        InferenceEngine(std::string ModelLocation, std::string HW_type){
            HW_CLASS = HW_type; 
            ModelPath = ModelLocation;
        }

        auto Vinference(cv::Mat frame) -> cv::Mat{
            //color conversion and resize
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 

            if(strcmp(HW_CLASS, "GPU") == 0){
                //construct the ONNX runtime for GPU
            }else{
                //construct the ONNX runtime for cpu
            }
            
        }
}