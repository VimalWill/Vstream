//https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp

#include <includes/inferEng.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

GstInferaEng::GstInferaEng(cv::Mat& Image){
    Input_img = Image;
}

cv::Mat GstInferaEng::InferenceEngine(){

   //onnx runtime
   std::string instanceName{"Gst-ONNX-Inference-Engine"};
   std::string modelPath{"/home/vimal/Edge AI/Vstream/model/yolov8n.onnx"}; 

   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, instanceName.c_str()); 
   Ort::SessionOptions sessionOptions; 
   sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
   Ort::Session session(env,modelPath.c_str(), sessionOptions); 

   Ort::AllocatorWithDefaultOptions allocator; 
   std::vector<const char*> InputNames = session.GetInputNameAllocated(0, allocator);
   Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0); 
   auto input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape(); 

   InputWidth = input_shape[2]; 
   InputHeight = input_shape[3];

   cv::Mat preproc_img = Preproc(Input_img);




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

    processed_frame = image_data.reshape(1, InputWidth * InputHeight); 
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