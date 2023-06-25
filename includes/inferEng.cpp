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
    
    std::cout << "---------- Input Info -------" << std::endl; 
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

    std::cout << "--------- Output Info -------" << std::endl; 
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

//Image preprocessing for model fit 
void GstAppSrcInfer::Preprocessor(cv::Mat& frame, float*& blob, std::vector<int64_t>& InputTensorShape){

    //color conversion 
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); 

    //resize with padding 
    cv::Mat resized_img; 
    letterBox(frame, resized_img);

    InputTensorShape[2] = resized_img.size().height; 
    InputTensorShape[3] = resized_img.size().width; 

    //normalization 
    cv::Mat float_img; 
    resized_img.convertTo(float_img, CV_32FC3, 1/255.0);
    blob = new float[float_img.cols * float_img.rows * float_img.channels()]; 
    cv::Size floatImageSize(float_img.cols, float_img.rows);

    //hwc -> chw 
    std::vector<cv::Mat> chw(float_img.channels());
    for(int i = 0; i < float_img.channels(); ++i){
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(float_img, chw);
}

//resize with padding
void GstAppSrcInfer::letterBox(cv::Mat& InputImage, cv::Mat& OutputImage){

    float org_width, org_height, target_width = 640.0, target_height = 640.0, ratio_w, ratio_h; 
    org_width = InputImage.size().width; 
    org_height = InputImage.size().height; 

    ratio_w = target_width / org_width; 
    ratio_h = target_height / org_height; 

    //scaling factor 
    float scaling_factor = std::min(ratio_w, ratio_h); 
    int new_width = org_width * scaling_factor; 
    int new_height = org_height * scaling_factor; 

    cv::Mat resized_img; 
    cv::resize(InputImage, resized_img, cv::Size(new_width, new_height)); 

    //paddings 
    int top_padding = (target_height - new_height)/2; 
    int bottom_padding = target_height - new_height - top_padding; 
    int left_padding = (target_width - new_width)/2; 
    int right_padding = target_width - new_width - left_padding; 

    cv::copyMakeBorder(resized_img, OutputImage, top_padding, bottom_padding, left_padding, right_padding, cv::BORDER_CONSTANT, cv::Scalar(0));
}

void GstAppSrcInfer::InferenceEngine(cv::Mat& frame){

    //processing
    std::vector<int64_t> InputTensorShape = {1, 3, -1, -1}; 
    float* blob; 

    Preprocessor(frame, blob, InputTensorShape); 
    size_t InputTensorSize = 1; 
    for(auto elem : InputTensorShape){
        InputTensorSize *= elem; 
    }
    std::cout << "Input Tensor Size:" << InputTensorSize << std::endl; 
}