//inference engine for Gstreamer Appsrc 
#include "inferEng.hpp"

GstAppSrcInfer::GstAppSrcInfer(){

    //initalize the ONNX-Runtime
   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, instance.c_str()); 
   sessionOptions = Ort::SessionOptions();
   sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); 

   session = Ort::Session(env, modelPath.c_str(), sessionOptions);
   Ort::AllocatorWithDefaultOptions allocator; 

   GetInputDetails(allocator); 
   GetOutputDetails(allocator);
}

void GstAppSrcInfer::GetInputDetails(Ort::AllocatorWithDefaultOptions allocator){
    
    std::cout << "------- Input Info -------" << std::endl; 
    for(int i=0; i<session.GetInputCount(); i++){

        //get input node name
        auto input_name_ptr = session.GetInputNameAllocated(i, allocator); 
        std::string input_name = input_name_ptr.get(); 
        InputNames.push_back(input_name); 
        std::cout << "Input Name:" << input_name << std::endl; 

        //get input tensor shape 
        std::vector<int64_t> inputTensorShape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        input_node_dims.push_back(inputTensorShape); 

        std::cout << "Input Shape:" << std::endl;
        for(auto elem : inputTensorShape){
            std::cout << elem <<  " ";
        }
        std::cout << std::endl;
    }

}

void GstAppSrcInfer::GetOutputDetails(Ort::AllocatorWithDefaultOptions allocator){

    std::cout << "------- Output Info -------" << std::endl; 
    for(int i=0; i<session.GetOutputCount(); i++){

        //get output node name
        auto output_name_ptr = session.GetOutputNameAllocated(i, allocator); 
        std::string output_name = output_name_ptr.get(); 
        OuputNames.push_back(output_name); 
        std::cout << "Output Name:" << output_name << std::endl;

        //get output tensor shape
        std::vector<int64_t> outputTensorShape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        output_node_dims.push_back(outputTensorShape);

        std::cout << "Output Shape:" << std::endl; 
        for(auto elem : outputTensorShape){
            std::cout << elem << " "; 
        }
        std::cout << std::endl; 
    }
}