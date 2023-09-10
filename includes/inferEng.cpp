#include "inferEng.hpp"


/*func@ load NN model & enable Hardware backend support*/
bool Infera::infera_load_model(){

    net = cv::dnn::readNetFromONNX(model_path);

    if(!net.empty()){
        std::cout << "[+]neural net loaded successfully" << std::endl; 
        is_loaded = true; 
    }else{
        perror("[-]error in loading the neural net"); 
        is_loaded = false; 
    }

    cl::Platform::get(&platforms); 
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices); 

    if(devices.size() != 0){
        std::cout << "[GPU]offloading inference backend to GPU" << std::endl; 
        net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA); 
        net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA)
    }else{
        std::cout << "[CPU]offloading inference backend to CPU" << std::endl; 
        net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_OPENCV); 
        net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);
    }

    return is_loaded; 
}