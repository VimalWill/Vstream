#include <includes/aisrc.hpp>
#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/app/gstappsrc.h>

#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>

static void GstAISrc::NeedSource(GstAppSrc* appsrc, gpointer user_data){
    //initalize the gst variable
    GstBuffer* buffer; 
    GstMapInfo map;
    GstFlowReturn ret; 

    //opencv source capture 
    cv::VideoCapture cap;
    cap.open(user_data);

    while(cap.isOpened()){
        //get the mat frames and push to Inference Engine 
        cv::Mat frame;
        cap.read(frame); 

        //initalize the Inference Engine here 
    }
}