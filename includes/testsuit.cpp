#include <gtest/gtest.h>
#include "infera.hpp"

/*func@ G-test for loading the neural net*/
TEST(Infera_Test_Suit, model_load_test){
    neural_engine infera; 
    bool load = infera.load_model(); 

    EXPECT_TRUE(load);
}

/*func@ G-tets for squarefit the incomming image*/
TEST(Infera_Test_Suit, format2sq_test){
    neural_engine infera; 
    cv::Mat test_img = cv::imread("/home/vimal/Edge_ai/Vstream/includes/test.jpeg", cv::IMREAD_COLOR);
    cv::Mat result = infera.format2sq(test_img); 

    ASSERT_FALSE(result.empty()); 
    ASSERT_EQ(3, result.channels());
}

TEST(Infera_Test_Suit, detect_test){
    neural_engine infera; 
    cv::Mat test_img = cv::imread("/home/vimal/Edge_ai/Vstream/includes/test.jpeg", cv::IMREAD_COLOR);
    cv::Mat result = infera.detect(test_img); 

    ASSERT_FALSE(result.empty()); 
    ASSERT_EQ(640, result.rows); 
    ASSERT_EQ(640, result.cols); 
}



int main(){
    testing::InitGoogleTest(); 
    return RUN_ALL_TESTS(); 
}