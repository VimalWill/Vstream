//inference engine for Gstreamer Appsrc 
#include "inferEng.hpp"

struct Detection{
    cv::Rect box; 
    float confs{}; 
    int classId();
};

GstAppSrcInfer::GstAppSrcInfer(){

   //initalize the ONNX-Runtime  
   //https://github.com/microsoft/onnxruntime/issues/4245
   env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, instance.c_str()); 
   sessionOptions = Ort::SessionOptions();

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
        inputNamesString.push_back(input_name_ptr.get()); 
        InputNames.push_back(inputNamesString[i].c_str());
        std::cout << "Input Name:" << InputNames[i] << std::endl; 

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
        outputNamesString.push_back(output_name_ptr.get()); 
        OutputNames.push_back(outputNamesString[i].c_str());
        std::cout << "Output Name:" << OutputNames[i] << std::endl;

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

//get the best class
void GstAppSrcInfer::GetBestClass(std::vector<float>::iterator& it, const int& numClasses, float& bestConf, int& bestClassesId){
    bestClassesId = 5; 
    bestConf = 0; 

    for(int i=5; i<numClasses + 5; i++){
        if(it[i] > bestConf){
            bestConf = it[i]; 
            bestClassesId = i - 5; 
        }
    }
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

//post-processing the inferene results
void GstAppSrcInfer::Postprocessor(std::vector<Ort::Value>& OutputTensor, int Org_width, int Org_height){

    std::vector<cv::Rect> boxes; 
    std::vector<float> confs; 
    std::vector<int> classIds; 

    float confThershold = 0.5; 
    float IoUThershold = 0.5; 
    const int num_class = output_node_dims[0][2] - 5; 

    for(int layer = 0; layer < output_node_dims.size(); layer+=1){

        std::vector<int64_t> outputShape = output_node_dims[layer]; 
        const float* rawOutput = OutputTensor[layer].GetTensorData<float>(); 
        size_t count = OutputTensor[layer].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> output(rawOutput, rawOutput + count); 
        int elementsInBatch = (int)(output_node_dims[layer][1] *  output_node_dims[layer][2]); 

        for(auto it = output.begin(); it != output.begin() + elementsInBatch; it += output_node_dims[layer][2]){
            
            float clsconf = it[4]; 
            if(clsconf > confThershold){
                int left = (int) (it[0]); 
                int top = (int) (it[1]); 
                int right = (int) (it[2]);
                int bottom = (int) (it[3]); 
                int width = (int) right - left;
                int height = (int) bottom - top;

                float objConf; 
                int classId; 

                GetBestClass(it, num_class, objConf, classId); 
                float confidence = clsconf * objConf; 

                boxes.emplace_back(left, top, width, height); 
                confs.emplace_back(confidence); 
                classIds.emplace_back(classId); 
            }
        }
    }

    //non - maximum supression
    std::vector<int> indices; 
    cv::dnn::NMSBoxes(boxes, confs, confThershold, IoUThershold, indices); 
    std::cout << "No. of indices:" << indices.size() << std::endl;
}

void GstAppSrcInfer::InferenceEngine(cv::Mat& frame){

    int org_width = frame.size().width; 
    int org_height = frame.size().height; 

    //processing
    std::vector<int64_t> InputTensorShape = {1, 3, -1, -1}; 
    float* blob = nullptr;

    Preprocessor(frame, blob, InputTensorShape); 
    size_t InputTensorSize = 1; 
    for(auto elem : InputTensorShape){
        InputTensorSize *= elem; 
    }
    std::cout << "Input Tensor Size:" << InputTensorSize << std::endl; 

    std::vector<float> InputTensorValues(blob, blob + InputTensorSize); 
    std::vector<Ort::Value> InputTensor; 

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); 

    InputTensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info, InputTensorValues.data(), InputTensorSize, 
        InputTensorShape.data(), InputTensorShape.size())); 
    assert(InputTensor[0].IsTensor());

    //session run
    auto output = session.Run(Ort::RunOptions{nullptr}, 
                              InputNames.data(), 
                              InputTensor.data(), 
                              InputNames.size(), 
                              OutputNames.data(), 
                              OutputNames.size());

    //post-processing 
    Postprocessor(output, org_width, org_height);

    delete[] blob;
}