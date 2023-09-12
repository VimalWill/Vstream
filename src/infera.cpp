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

/*func@ performs forward prob on the input*/
cv::Mat neural_engine::detect(cv::Mat& img){

    cv::Mat modelInput = format2sq(img);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int row = outputs[0].size[2];
    int dims = outputs[0].size[1]; 

    outputs[0] = outputs[0].reshape(1, dims); 
    cv::transpose(outputs[0], outputs[0]); 
    float *data = (float *)outputs[0].data; 

    float x_factor =  modelInput.cols / 640.0; 
    float y_factor =  modelInput.rows / 640.0; 

    std::vector<int> class_ids; 
    std::vector<float> confs; 
    std::vector<cv::Rect> boxes; 

    for(int i = 0; i < row; ++i){
        
        float *classes_scores = data + 4; 
        cv::Mat scores(1, 80, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > 0.45)
            {
                confs.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }

        data += dims;
    }

    /*non-max supression*/
    std::vector<int> nms_results; 
    cv::dnn::NMSBoxes(boxes, confs, 0.45, 0.50, nms_results); 

    std::vector<Detection> detection{}; 
    for(unsigned int i = 0; i < nms_results.size(); ++i){

        int idx = nms_results[i]; 
        Detection results; 
        results.class_id = class_ids[i]; 
        results.confidence = confs[i];
        results.box = boxes[idx];

        detection.push_back(results);
    }

    cv::Mat infera_ouput = img; 
    /*draw the bounding box*/
    for(int j=0; j<detection.size(); ++j){
        Detection detector = detection[j]; 
        cv::rectangle(infera_ouput, detector.box, cv::Scalar(0, 0, 255), 2); 

    }

    return infera_ouput; 

}
