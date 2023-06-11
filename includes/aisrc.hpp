#ifndef AISRC_HPP 
#define AISRC_HPP

#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/app/gstappsrc.h>

class GstAISrc{
    public:
        static void NeedSource(GstAppSrc*, gpointer); 
        bool HardwareDetect();

};

#endif /*AISRC_HPP*/