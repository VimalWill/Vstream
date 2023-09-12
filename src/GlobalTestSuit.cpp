#include <gtest/gtest.h>
#include <iostream>

#include "infera.hpp"

/*func@ testsuit for loading model*/
TEST(Vstream_global_testsuit, model_load_test){
    neural_engine infera("../model/yolov8.onnx"); 
    bool load = infera.load_model(); 

    EXPECT_TRUE(load); 
}

/*func@ testsuit for fomat image*/
TEST(Vstream_global_testsuit, format_test){
    neural_engine infera("../model/yolov8.onnx"); 
    cv::Mat img = cv::imread("../images/test.jpeg"); 

    cv::Mat result = infera.format2sq(img); 
    
    ASSERT_FALSE(result.empty()); 
    ASSERT_EQ(3, result.channels()); 
}

TEST(Vstream_global_testsuit, detect_test){
    neural_engine infera("../model/yolov8.onnx"); 
    bool load = infera.load_model(); 
    cv::Mat img = cv::imread("../images/test.jpeg"); 

    cv::Mat result = infera.detect(img); 
    ASSERT_FALSE(result.empty()); 
    ASSERT_EQ(3, result.channels());
    EXPECT_TRUE(load); 

}

int main(){
    testing::InitGoogleTest(); 
    return RUN_ALL_TESTS(); 
}