#include <CL/opencl.hpp>
#include <bits/stdc++.h>

class ErrorHandle{
    private:
        cl_int err; 
    
    public:
        ErrorHandle(cl_int error){
            err = error; 
            Handler();
        }

        void Handler(){
            switch(err){
                case CL_INVALID_VALUE:
                    std::cerr << "Error: The Platform is Null"; 
                
                default:
                    std::cerr << "Un-Identified Error Flag" << std::endl;
            }
        }
}