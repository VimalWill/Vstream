/*Vsrc - handles various source and converting to frames,
Pre - processing the frames and Push to AI Inference on it,
further push to GStreamer buffer*/

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/app.h>
#include <opencv4/opencv2/opencv.hpp>

#include <bits/stdc++.h>

struct userVar{
    std::string SourceName; 
    cv::VideoCapture cap; 
}vars; 

//analyse the source and return in the valid type
//initally the source is treated as string 
auto sourceAnalyzer(){
    
}

static void NeedSource(GstAppSrc *appsrc, gpointer userData){
    //initalize the gstreamer handles 
    GstBuffer *Buffer; 
    GstFlowReturn ret; 
    GstMapInfo map; 
    vars.cap()

}