#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS

#include <includes/aisrc.hpp>
#include <includes/InferEng.hpp>
#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/app/gstappsrc.h>
#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

static void GstAISrc::NeedSource(GstAppSrc* appsrc, gpointer user_data){
    //initalize the gst variable
    GstBuffer* buffer; 
    GstMapInfo map;
    GstFlowReturn ret; 

    GstClockTime timestamp;
    GstInferaEng engine;

    //opencv source capture 
    cv::VideoCapture cap;
    cap.open(user_data);

    while(cap.isOpened()){
        //get the mat frames and push to Inference Engine 
        cv::Mat frame;
        cap.read(frame); 

        //AI Inference Engine
        cv::Mat preprocImg = engine.Preproc(frame);
        cv::Mat img = engine.InferenceEngine(preprocImg); 
         const int IMGwidth = 640; 
    const int IMGHeight = 480;

        //allocate the GStreamer Buffer 
        size_t bufsize = img.cols * img.rows * img.channels();
        buffer = gst_buffer_new_allocate(NULL, bufsize, NULL); 

    }
}

bool GstAISrc::HardwareDetect() {

    //get the opencl platform and context 
    std::vector<cl::Platform> platforms; 
    std::vector<cl::Device> platformDevices, ctxDevice;  
    bool HW_FLG = false;
    
    try{
        static cl_int err = cl::Platform(&platforms); 
        if(err != CL_SUCCESS)
            std::cerr << "no platform found"<< std::endl; 

        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &platformDevices); 
        cl::Context context(platformDevices); 
        ctxDevice = context.getInfo<CL_CONTEXT_DEVICES>(); 

        //get the device name 
        std::string deviceName = ctxDevice[0].getInfo<CL_DEVICE_NAME>();  
        if(deviceName.find("NVIDIA") != std::string::npos){
            std::cout << "Required Hardware Detetcted" << std::endl;
            HW_FLG = true; 
        }else{
            std::cout << "Warning: Required Hardware not found, offloading to CPU" << std::endl;
            HW_FLG = false; 
        }
    }

    //exceptation handle
    catch(cl::Error &err){
        std::cout << err.what() << std::endl; 
        HW_FLG = false; 
    }

    return HW_FLG; 
}

