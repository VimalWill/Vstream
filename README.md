<h1> Vstream - Video Analytics Pipeline </h1>

The pipeline is constructed on the top of Neural-Net Inference Engine called "Infera" which currently supports YoloV8 object detection model and Streaming pipeline buiild on the base of GStreamer framework. The Infera supports google-test for the functionality verification of each functions in the system. 

<h2> Requirements </h2>

- OpenCV (above 4.8-dev)
- OpenCL 
- GStreamer 
- Google-Test for C++
- Standard C++ 17

<h2> Build Procedure for Vstream </h2>
The building process for Vstream handles building both Test-suit for Infera and Final Executable for the Vstream

```
$ mkdir build && cd build
$ cmake ..
$ make -j64
```
 - The exectuables will be created for both Global Test-suit and Vstream

<h2> Features of Vstream </h2>

- Adapative Hardware Offloading of Hardware for Inference 
- Hardware Accelerated Video Compression for network stream (on-progress)

<h3> To-Dos</h3>

- Hardware acceleration for video encoding
- Network Streaming 
- Object Tracking


