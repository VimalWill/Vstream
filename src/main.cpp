#include "infera.hpp"

#include <gst/app/gstappsrc.h>
#include <opencv2/opencv.hpp>
#include <gst/app/app.h>
#include <gst/gst.h>

#include <queue>
#include <iostream>

GMainLoop* loop; 
int width = 0, height = 0, channel = 0; 
std::queue<cv::Mat> AppSrcQueue; 

/*func@ infer frames & push to queue*/
void push_img(){

    neural_engine infera("../model/yolov8.onnx");
    while (true){
        cv::VideoCapture cap(0); 

        if(!cap.isOpened()){
            perror("[-]failed to open default camera");
            break; 
        }
         
        bool load = infera.load_model(); 
        if(!load)
            break; 

        cv::Mat frame; 
        cap.read(frame); 

        cv::Mat infera_output = infera.detect(frame); 

        width = infera_output.cols; 
        height = infera_output.rows; 
        channel = infera_output.channels(); 

        AppSrcQueue.push(infera_output); 
    }
}

/*func@ GStreamer Appsrc - takes images | push to GstBuffer*/
static void prepare_buffer(GstAppSrc* appsrc){

    static gboolean white = FALSE; 
    static GstClockTime timestamp = 0; 
    GstBuffer *buffer; 
    GstFlowReturn ret; 
    GstMapInfo map; 

    while(!AppSrcQueue.empty()){

        cv::Mat frame = AppSrcQueue.front(); 
        AppSrcQueue.pop(); 

        /*convert image -> GstBuffer*/
        guint buffer_size = width * height * channel; 
        buffer = gst_buffer_new_allocate(NULL, buffer_size, NULL); 
        
    }
}
