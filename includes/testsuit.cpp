#include <gtest/gtest.h>
#include "inferEng.hpp"

/*func@ G-test for loading the neural net*/
TEST(Infera_Test_Suit, model_load_test){
    neural_engine infera; 
    bool load = infera.load_model(); 

    EXPECT_TRUE(load);
}

int main(){
    testing::InitGoogleTest(); 
    return RUN_ALL_TESTS(); 
}