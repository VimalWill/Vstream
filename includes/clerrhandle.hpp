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

                case CL_INVALID_PROPERTY:
                    std::cerr << "Error: The value specified for supported property 
                    is not valid" << std::endl;

                case CL_INVALID_DEVICE_TYPE:
                    std::cerr << "Error: Device type is not valid" << std::endl;
                
                case CL_DEVICE_NOT_AVAILABLE:
                    std::cerr << "Error: No specified devices are not available" << std::endl;
                
                case CL_DEVICE_NOT_FOUND:
                    std::cerr << "Error; no devices which specified are found" << std::endl;

                default:
                    std::cerr << "Error: Un-Identified Error Flag" << std::endl;
            }
        }
}