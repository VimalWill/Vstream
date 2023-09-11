<h2> Infera - Neural Inference Engine for Vstream </h2>

The Inference Engine, built upon the OpenCV framework, is designed for neural networks. Presently, it is configured for YoLov5 as the foundational object detection model using the Coco dataset. The engine possesses the capability of Adaptive Hardware backend support for inference, which automatically detects the underlying hardware and configures the backend accordingly.

<h3> Requirements </h3>

- OpenCL 
- OpenCV 
- G-test (optional)
- CMake 

<h3> Inference Engine Unit Test </h3>

Unit Tests are written for verifying the functions of Inference Engine and which can be executed by the following instructions, 

```
$ mkdir build && cd build
$ cmake ..
$ make -j8
$ ./Infera
```
