//https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp

#include <includes/inferEng.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

GstInferaEng::GstInferaEng(cv::Mat& Image){
    Input_img = Image;
}
void GstInferaEng::letterBox(cv::Mat& Input_img, cv::Mat& Output_img){
    float target_width = 640.0; 
    float target_heigth = 640.0; 

    float org_width = Input_img.size().width; 
    float org_height = Input_img.size().height; 

    //computing scaling factor 
    float ratio_w = target_width / org_width; 
    float ratio_h = target_heigth / org_height; 

    float scaling_factor = std::min(ratio_w, ratio_h); 
    int new_width = org_width * scaling_factor; 
    int new_height = org_height * scaling_factor; 

    //image resize
    cv::Mat resized_frame; 
    cv::resize(Input_img, resized_frame, cv::Size(new_width, new_height)); 

    int top_padding = (target_heigth - new_height)/2; 
    int bottom_padding = target_heigth - new_height - top_padding; 
    int left_padding = (target_width - new_width)/2; 
    int right_padding = target_width - target_heigth - left_padding; 

    cv::copyMakeBorder(resized_frame, Output_img, top_padding, bottom_padding, left_padding, right_padding, cv::BORDER_CONSTANT, cv::Scalar(0)); 

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
   char* in_layer_name = input_layer.get(); 
   auto output_layer = session.GetOutputNameAllocated(0, allocator); 
   char* out_layer_name = output_layer.get();

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

   //get inference 
   auto output_layer = session.GetOutputNameAllocated(0, allocator); 
   char* out_layer_name = output_layer.get(); 
   std::cout << out_layer_name << std::endl; 

   std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr}, &in_layer_name, &input_tensor, 1, &out_layer_name, 1); 
   assert(outputs.front().IsTensor());
   float* tensorData = outputs.front().GetTensorMutableData<float>();
   auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape(); 


   const int64_t batchSize = output_shape[0];
   const int64_t cols = output_shape[1]; 
   const int64_t rows = output_shape[2]; 

}

std::vector<float> GstInferaEng::Preproc(cv::Mat& image){

    //color-conversion and resize 
    cv::Mat frame = image; 
    cv::Mat resized_frame; 
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 
    cv::resize(frame, resized_frame, cv::Size(InputWidth, InputHeight)); 

    //transpose 
    cv::transpose(resized_frame, resized_frame); 

    //normalize 
    cv::Mat image_data; 
    resized_frame.convertTo(image_data, CV_32F, 1.0/255.0);
    //https://stackoverflow.com/questions/64084646/expand-dimensions-for-opencv-mat-object-in-c
    //int buf[4] = {1, image_data.channels(), image_data.rows, image_data.cols}; 
    //processed_frame(4, buf, image_data.type(), image_data.data); 

    //converting cv::Mat to vector
    std::vector<float> processed_vector(image_data.ptr<float>(), image_data.ptr<float>() + (image_data.rows * image_data.cols * image_data.channels())); 
    return processed_vector; 
}

cv::Mat GstInferaEng::Postproc(std::vector<std::vector<float>>& output, cv::Size& org_shape){

    //calculate the scaling factor
    int org_width = org_shape.width;
    int org_height = org_shape.height;

    int64_t x_factor = 640.0 / org_width; 
    int64_t y_factor = 640.0 / org_height; 

    int left, right, height, width; 

    float conf = 0.5; 
    float iou = 0.5; 

    for(auto& row : output){
        float maxValue = row[4]; 
        int maxIndex = 4; 

        for(int i=5; i<row.size(); ++i){
            if(row[i] > maxValue){
                maxValue = row[i]; 
                maxIndex = i; 
            }
        }

        if(maxValue >= conf){
                
            //get the bounding box data
            float x = row[0]; 
            float y = row[1]; 
            float w = row[2]; 
            float h = row[3]; 

            //calculate the scaled bounding box 
            left = int((x - w/2)*x_factor); 
            right = int((y - h/2)*y_factor); 
            width = int(w*x_factor);
            height = int(h*y_factor);
        }
    } 

}
