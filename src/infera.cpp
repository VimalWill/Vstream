#include "infera.hpp"


/*func@ constructor*/
neural_engine::neural_engine(std::string path){
    model_path = path;    
}

/*func@ destructor*/
neural_engine::~neural_engine(){}; 

/*func@  loads neural net & adaptive inference HW backend*/
bool neural_engine::load_model(){

    bool is_load = false; 
    bool is_gpu = false; 

    net = cv::dnn::readNetFromONNX(model_path);
    if (!net.empty()){
        std::cout << "[+]neural net successfully loaded" << std::endl; 
        is_load = true; 
    }
    else{
        perror("[-]failed to load the neural net"); 
        is_load = false; 
    }

    std::vector<cl::Platform> platforms; 
    std::vector<cl::Device> devices; 

    cl::Platform::get(&platforms); 
    if (platforms.size() == 0)
        is_gpu = false; 
    else
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    if (devices.size() != 0)
        is_gpu = true;
    
    if( (is_gpu)){
        std::cout << "[GPU]inference backend offloading to GPU" << std::endl; 
        net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA); 
        net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA); 
    }else{
        std::cout << "[CPU]inference backend offloading to CPU" << std::endl; 
        net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_OPENCV); 
        net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);
    }

    return is_load; 
}

/*func@ Square fit the input image*/
cv::Mat neural_engine::format2sq(cv::Mat& source){
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

cv::Mat neural_engine::detect(cv::Mat& img){

    cv::Mat modelInput = format2sq(img);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs[0]; 

}
