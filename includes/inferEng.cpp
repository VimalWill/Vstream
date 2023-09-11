#include "inferEng.hpp"

/*func@  loads neural net & adaptive inference HW backend*/
bool neural_engine::load_model(){

    bool is_load = false; 
    bool is_gpu = false; 

    std::ifstream f("/home/vimal/Vstream/includes/config.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::string model_path = data["model_path"];

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