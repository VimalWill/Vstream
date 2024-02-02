#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/app.h>
#include <queue>
#include <thread>
#include <chrono>

#include "infera.hpp"

#include <opencv2/opencv.hpp>

static GMainLoop *loop; 
neural_engine infera("../model/yolov8.onnx");

struct _streamparams{
    cv::VideoCapture cap;
    std::queue<cv::Mat> comman_queue; 
    guint buffer_size = 0; 
}streamparams; 

void cv_producer(){

    while(true){
        cv::Mat frame; 
        streamparams.cap.read(frame); 
        streamparams.comman_queue.push(frame); 
    }

    //memory free 
    streamparams.cap.release();
}

static void
prepare_buffer(GstAppSrc* appsrc){
    
    static GstClockTime timestamp = 0; 
    GstFlowReturn ret; 
    GstMapInfo info; 
    cv::Mat frame; 
    gint idx = 0; 

    GstBuffer* buffer = gst_buffer_new_allocate(NULL, streamparams.buffer_size, NULL); 
    frame = streamparams.comman_queue.back(); 
    auto start = std::chrono::high_resolution_clock::now();
    frame = infera.detect(frame);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Inference Time: " << duration.count() << " seconds" << std::endl;

    if(streamparams.comman_queue.size() >= 30)
        streamparams.comman_queue.pop();

    // guint buffer_size = img.rows * img.cols * img.channels(); 
    // buffer = gst_buffer_new_allocate(NULL, buffer_size, NULL); 
    // gst_buffer_fill(streamparams.buffer, 0, streamparams.frame_inital.data, streamparams.buffer_size); 


    gst_buffer_map(buffer, &info, GST_MAP_WRITE); 
    unsigned char* buf = info.data; 
    memmove(buf, frame.data, streamparams.buffer_size); 
    gst_buffer_unmap(buffer, &info); 

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
  /*gstBuffer allocation*/
  streamparams.cap.open("/dev/video0");
  cv::Mat ref_frame; 
  streamparams.cap.read(ref_frame); 
  streamparams.buffer_size = ref_frame.rows * ref_frame.cols * ref_frame.channels(); 

  std::thread queue_t(cv_producer); 
  queue_t.detach(); 
  if(!infera.load_model())
        perror("[-]error in loading model");

  int width = streamparams.cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = streamparams.cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
  GstElement *pipeline, *appsrc, *conv, *videosink;

  /* init GStreamer */
  gst_init (NULL, NULL);
  loop = g_main_loop_new (NULL, FALSE);

  /* setup pipeline */
  pipeline = gst_pipeline_new ("pipeline");
  appsrc = gst_element_factory_make ("appsrc", "source");
  conv = gst_element_factory_make ("videoconvert", "conv");
  //enc = gst_element_factory_make ("x264enc", "enc");
  videosink = gst_element_factory_make ("fpsdisplaysink", "videosink");
  
  /* setup */
  g_object_set (G_OBJECT (appsrc), "caps",
  		gst_caps_new_simple ("video/x-raw",
				     "format", G_TYPE_STRING, "BGR",
				     "width", G_TYPE_INT, width,
				     "height", G_TYPE_INT, height,
				     "framerate", GST_TYPE_FRACTION, 0, 1,
				     NULL), NULL);
  gst_bin_add_many (GST_BIN (pipeline), appsrc, conv, videosink, NULL);
  gst_element_link_many (appsrc, conv, videosink, NULL);

  /* setup appsrc */
  g_object_set (G_OBJECT (appsrc),
		"stream-type", 0,
		"format", GST_FORMAT_TIME,
        "is-live", true, NULL);
  g_signal_connect (appsrc, "need-data", G_CALLBACK (cb_need_data), NULL);

  /* play */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_main_loop_run (loop);

  /* clean up */
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (pipeline));
  g_main_loop_unref (loop);

  return 0;
  }

