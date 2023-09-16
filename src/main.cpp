#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/app.h>

#include "infera.hpp"

#include <opencv2/opencv.hpp>

static GMainLoop *loop; 
neural_engine infera("../model/yolov8.onnx");
cv::VideoCapture cap("/dev/video0");

static void
prepare_buffer(GstAppSrc* appsrc){
    
    static GstClockTime timestamp = 0; 
    GstBuffer *buffer; 
    GstFlowReturn ret; 

    cv::Mat frame; 
    cap.read(frame); 

    cv::Mat img = infera.detect(frame);

    guint buffer_size = img.rows * img.cols * img.channels(); 
    buffer = gst_buffer_new_allocate(NULL, buffer_size, NULL); 
    gst_buffer_fill(buffer, 0, img.data, buffer_size); 

    GST_BUFFER_PTS (buffer) = timestamp;
    GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 2);
    timestamp += GST_BUFFER_DURATION (buffer);

    ret = gst_app_src_push_buffer(appsrc, buffer); 

    if(ret != GST_FLOW_OK){
        g_main_loop_quit(loop); 
    }
}

static void 
cb_need_data(GstElement *appsrc, guint unused_size, gpointer user_data){

    prepare_buffer((GstAppSrc*)appsrc); 
}

int main()
{
    
  infera.load_model();
  
  GstElement *pipeline, *appsrc, *conv, *videosink, *queue1;

  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH); 
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  /* init GStreamer */
  gst_init (NULL, NULL);
  loop = g_main_loop_new (NULL, FALSE);

  /* setup pipeline */
  pipeline = gst_pipeline_new ("pipeline");
  appsrc = gst_element_factory_make ("appsrc", "source");
  conv = gst_element_factory_make ("videoconvert", "conv");
  videosink = gst_element_factory_make ("autovideosink", "videosink");
  queue1 = gst_element_factory_make ("queue", "queue1");

  /* setup */
  g_object_set (G_OBJECT (appsrc), "caps",
  		gst_caps_new_simple ("video/x-raw",
                     "is-live", G_TYPE_BOOLEAN, TRUE,
                     "max-buffers", G_TYPE_UINT64,30, 
				     "format", G_TYPE_STRING, "BGR",
				     "width", G_TYPE_INT, width,
				     "height", G_TYPE_INT, height,
				     "framerate", GST_TYPE_FRACTION, 0, 1,
				     NULL), NULL);
  gst_bin_add_many (GST_BIN (pipeline), appsrc, queue1, conv, videosink, NULL);
  gst_element_link_many (appsrc, queue1, conv, videosink, NULL);

  /* setup appsrc */
  g_object_set (G_OBJECT (appsrc),
		"stream-type", 0,
		"format", GST_FORMAT_TIME, NULL);
  g_signal_connect (appsrc, "need-data", G_CALLBACK (cb_need_data), NULL);

  /* play */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_main_loop_run (loop);

  /* clean up */
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (pipeline));
  g_main_loop_unref (loop);

  cap.release();

  return 0;
  }