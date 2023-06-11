#include <CL/opencl.hpp>
#include <bits/stdc++.h>

#include <includes/clerrhandle.hpp>
#include <includes/Vinfra.hpp>
#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/app.h>

bool HWFlag = false; 

class HWDetector{
    private:
        std::vector<cl::Platform> platforms; 
        std::vector<cl::Device> devices, ctxDevices;

        std::string DeviceName;
        std::string HW = "NVIDIA";
        cl_int err;

        //bool HWFlag = false;

    public:
    
        void Detector(){

            //get the context and ctx device 
            cl::Platform(&platforms); 
            err = platforms[0].getInfo(CL_DEVICE_TYPE_GPU, &devices); 

            if (err != CL_SUCCESS){
                ErrorHandle handle(err); 
            }

            err = cl::Context context(devices); 
            if (err != CL_SUCCESS){
                ErrorHandle handle(err);
            }

            //get the device from the context
            ctxDevices = context.getInfo<CL_CONTEXT_DEVICES>(); 
            DeviceName = ctxDevices[0].getInfo<CL_DEVICE_NAME>(); 

            //search for NVIDIA GPU support
            if(DeviceName.find(HW) != std::string::npos){
                std::cout << "The NVIDIA GPU accelerator detected" << std::endl;
                HWFlag = true; 
            }else {
                std::cout << "Warning: The particular accelerator not detected" << std::endl; 
                HWFlag = false;
            }
        }
};

class SrcHandle{
    private:
        GstBuffer* buffer; 
        GstMapInfo map; 
        GstFlowReturn ret; 
        GMainLoop* loop;

        //source handle params 
        std::string Srcloc; 
        char SrcType;
    
    public:
        SrcHandle(std::string SrcDir, char SrcCl){
            Srcloc = SrcDir; 
            SrcType = SrcCl;

            //initalize the detector 
            HWDetector().Detector();

        }

        static void NeedSource(GstAppSrc *appsrc, gpointer user_data){
            
            //get the appsrc and push to buffer
            if(SrcType == "i"){
                auto src_int = int(Srcloc); 

                cv::VideoCapture cap(src_int); 
                while(cap.isOpened()){
                    cv::Mat frame; 
                    cap.read(frame); 

                    //AI Inference and Overlay function 
                    if (HWFlag){
                        //use NVIDIA GPU
                        InferenceEngine engine("/includes/models", "GPU"); 
                        cv::Mat frame = engine.Vinference(); 

                    }

                    
                    size_t size = frame.cols * frame.rows * frame.channels(); 
                    buffer = gst_buffer_new_allocate(NULL, size, NULL); 
                    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
                    memcpy(map.data, frame.data, size); 
                }
            }  
        }
};