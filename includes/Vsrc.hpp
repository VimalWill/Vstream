/*Vsrc - handles various source and converting to frames,
Pre - processing the frames and Push to AI Inference on it,
further push to GStreamer buffer*/

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/app.h>
#include <opencv4/opencv2/opencv.hpp>

#include <bits/stdc++.h>

class Vsrc{
    private:
        
        static void NeedSource(GstAppSrc* appsrc, gpointer userData){
            //get the source and convert to frames
            
        }

}