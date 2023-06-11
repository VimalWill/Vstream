/*Vstream is a end to end streaming pipeline for handling 
AI deployments. The based is constructed on the top of 
GStreamer pipeline and the AI models can be deployed into 
the pipeline which is based on Onnx runtime 

Author: Vimal W 
Date: 7th June 2023*/

#include <gstreamer-1.0/gst/app/gstappsrc.h>
#include <gstreamer-1.0/gst/rtsp-server/rtsp-server.h>