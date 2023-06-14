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

   auto input_layer = session.GetInputNameAllocated(0, allocator); 
   std::string in_layer_name = input_layer.get(); 

   Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0); 
   auto input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape(); 

   InputWidth = input_shape[2]; 
   InputHeight = input_shape[3];

   size_t tensorSize = input_shape[1] * input_shape[2] * input_shape[3]; 

   //get the preprocessed vector 
   std::vector<float> preprocessed_img = Preproc(Input_img); 
   
   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); 
   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocessed_img.data(), tensorSize, input_shape.data(), 4);
   assert(input_tensor.IsTensor()); 

}

std::vector<float> GstInferaEng::Preproc(cv::Mat& image){

    //Image preprocessing for YOLOv8 
    cv::Mat frame, frame_resize, processed_frame; 
    frame = image; 

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 
    cv::resize(frame, frame_resize, cv::Size(InputWidth, InputHeight)); 

    cv::Mat image_data; 
    frame_resize.convertTo(image_data, CV_32F, 1.0/255.0); 
    cv::transpose(image_data, image_data); 

    //https://stackoverflow.com/questions/64084646/expand-dimensions-for-opencv-mat-object-in-c
    //int buf[4] = {1, image_data.channels(), image_data.rows, image_data.cols}; 
    //processed_frame(4, buf, image_data.type(), image_data.data); 

    //converting cv::Mat to vector
    std::vector<float> processed_vector(image_data.ptr<float>(), image_data.ptr<float>() + (image_data.rows * image_data.cols * image_data.channels())); 
    
    return processed_vector; 
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