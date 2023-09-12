#include <gtest/gtest.h>
#include <iostream>

#include "infera.hpp"

/*func@ testsuit for loading model*/
TEST(Vstream_global_testsuit, model_load_test){
    neural_engine infera("../model/yolov5s.onnx"); 
    bool load = infera.load_model(); 

    EXPECT_TRUE(load); 
}

/*func@ testsuit for fomat image*/
TEST(Vstream_global_testsuit, format_test){
    neural_engine infera("../model/yolov5s.onnx"); 
    cv::Mat img = cv::imread("../images/test.jpeg"); 
    
    ASSERT_FALSE(img.empty()); 
    ASSERT_EQ(3, img.channels()); 
}


int main(){
    testing::InitGoogleTest(); 
    return RUN_ALL_TESTS(); 
}