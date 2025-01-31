# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo
2. Get media data
```shell
git lfs pull
```
3. Make a build directory in the top level project directory: `mkdir build && cd build`
4. Compile: `cmake .. && make`
5. Run it: `./3D_object_tracking`.

## Run
```shell
./3D_object_tracking -det detectorType -des descriptorType -sel selectorType -mat matcherType -dir save_dir -vis 0|1
```
- detectorType: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT (default **SHITOMASI**)
- descriptorType: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT (default **BRISK**)
- matcherType: MAT_BF, MAT_FLANN (default **MAT_BF**)
- selectorType: SEL_NN, SEL_KNN (default **SEL_NN**)
- save_dir: directory to save images. Is used only if -DSAVE is set (default **./out**)

## Build to save images without visualizing
```shell
cmake -DSAVE=on ..
```
## FP.0 Final Report
Assignment|Function|Call|Declared|Defined|Return|What is it|
----------|--------|----|--------|-------|------|----------|
FP1|matchBoundingBoxes|FinalProject_Camera.cpp:304|camFusion.hpp:14|camFusion_Student.cpp:354|matches|ids pairs of the matched regions of interest|
FP2|computeTTCLidar|FinalProject_Camera.cpp:342|camFusion.hpp:20|camFusion_Student.cpp:305|TTC|time-to-collision based on Lidar data|
|FP3|clusterKptMatchesWithROI|FinalProject_Camera.cpp:349|camFusion.hpp:13|camFusion_Student.cpp:150|boundingBox.kptMatches|Keypoint matches which correspond to a given bounding box|
|FP4|computeTTCCamera|FinalProject_Camera.cpp:350|camFusion.hpp:18|camFusion_Student.cpp:228|TTC|time-to-collision based on keypoint correspondences in successive images|
|FP5|Done in FP56.pdf and fp5_6.xlsx|
|FP6|Done in FP56.pdf and fp5_6.xlsx|

