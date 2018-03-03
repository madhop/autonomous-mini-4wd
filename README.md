# OpenCV-based Lanes Detection

This algorithm detects the two more likely lanes in a video footage.
The class "LanesDetection" provides two functions
### Image Coordinates  
The function "detectLanesImage(Mat src)" takes as input a frame and returns a vector of either zero (if not detected) or two lanes. In turn each lane is a vector of "Point"s. Each point is a 2D coordinate of a given pixel in the original frame. The origin is the top-left corner of the image.
### World coordiantes  
The function "detectLanesWorld(Mat src)" takes as input a frame and returns a vector of either zero (if not detected) or two lanes. In turn each lane is a vector of "Point3f"s. Each point is a 3D coordinate of a given point in the real world, expressed in cm. The origin is the position of the camera at ground level.  

## Prerequisites
- cmake
  ```
  sudo apt-get install cmake
  ```
  or, if already installed
  ```
  sudo apt-get upgrade
  ```

- OpenCV 3.3.0: installation guide http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

### Example video data
Try the algorithm with some toy examples.
http://www.mediafire.com/file/xsv2qv5cpnzo6as/autonomous-mini-4wd-examples.zip

## Running test

- clone/download master
- from terminal, go to the folder Lanes/build
- execute the commands
```
cmake .
make
 ```

You can try an example provided by us by running the command
```
./launch_lanes_detection [video_path]
```

Here the results on three different video (click to play):  
[![Alt text for your video](https://img.youtube.com/vi/zvCGRYlw3hM/0.jpg)](https://youtu.be/zvCGRYlw3hM)

### Parameters
Given different videos, in order to get a better performance in each different case, you may need to change some parameters:
- "rectWidthRatio": to change the width of the windows (Figure 4);
- "nRect": to change the number of windows;
- "order": the order of the polynomial to fit;
- "horizonOffsetRatio": to move up/down the smaller base of the trapezoid (four yellow dots in Figure 1) upon which the perspective transform is computed;
- "perspAnchorOffsetRatio": to move up/down the bigger base of the trapezoid;
- "rectOffsetRatio": to avoid the dashboard;
- "camera": to change the camera calibration parameters (fxRatio, cxRatio, fyRatio, cyRatio, dist1, dist2, dist5).  
    e.g.,
    ```
    lanesDetection.camera.setFxRatio(0.45)
    ```
    In order to find these parameters you need to run a camera calibration program, provided by OpenCV (by default the parameters are computed for GoPro4).

## Algorithm
- **Camera calibration**: each frame is undistorted.
- **Vanishing point** (green dot in the image): computed as moving average in the first "vanishingPointWindow" frames of the video, after "vanishingPointWindowOffset" frames.
![alt text](https://image.ibb.co/j8JF8S/2_vanish_point.jpg)  
(Figure 1)  
Only after the vanishing point is computed, the algorithm start looking for the lanes:
- **Perspective transform** to have a bird view of the image.
- **Binary thresholding** of the perspective transform to distinguish the lanes from the road.
![alt text](https://image.ibb.co/djExoS/threshold.jpg)  
(Figure 2)
- **Histogram** to compute the distribution of white pixels along the vertical lines. The two highest peaks are taken.
![alt text](https://image.ibb.co/k9O42n/3_hist.jpg)  
(Figure 3)
- **Sliding window method** to classify the two lanes. For each window (red rectangles in Figure 4), if a connected component is found within it, the centroid of the connected component is computed (red dots in Figure 4).  
"nRects" windows are placed upon the perspective transformed image by fitting partial curves, starting from the peaks found by the histogram, at bottom level.
- **Polynomial fitting** of the centroids of the windows to find the two lanes (green lines in Figure 4).
![alt text](https://image.ibb.co/focyTS/rectangles.jpg)  
(Figure 4)  

## Possible improvements
- World coordinates are just an esteem: we took n measurements from a suitably crafted video and computed the scaling factor. However, the goodness of the esteem depends on how precise is the vanishing point. A better estimation could be obtained by computing the roto-transalition matrix of the camera.
- Histogram could be improved by fitting a mixture of Gaussian models or implementing a derivative-based model.
- The algorithm returns only two lanes. It could be extended to multiple ones.
- The vanishing point could be computed with better computer vision techniques and tracked with an algorithm such as Kalman filter.

## Authors
**Luca Fucci** and **Umberto Fazio**  
from Politecnico di Milano  

***See you soon..***
