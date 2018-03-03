# autonomous-mini-4wd

OpenCV installation guide: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

Example video data:
https://mega.nz/#!BuoF0bqT!y6kVfObzDOj2Tt8uet7h_UjsCH8HSNY571BSNvWxdQQ

- go to folder Lanes/build
- execute the command "cmake ."
- execute the command "make"

The class "LanesDetection" provides the position of the two detected lanes in two different ways:
- The function "detectLanesImage(Mat src)" returns each point of both lanes as pixel coordiantes, where the origin is the top-left corner of the image.
- The function "detectLanesWorld(Mat src)" returns each point of both lanes as cm coordinates, where the origin is the position of the camera.

You can try an example provided by us by running the command "./launch_lanes_detection [video_path]"

Given different videos, in order to get a better performance in each different case, you may need to change some parameters:
- "rectWidthRatio": to change the width of the windows (Figure 3);
- "nRect": to change the number of windows;
- "brightnessModelB0" and "brightnessModelB1": to tune the thresholding;
- "order": the order of the curve to fit;
- "perspAnchorOffsetRatio": to move up/down the bigger base of the trapezoid (four yellow dots in Figure 1) upon which the perspective transform is computed.

##The algorithm:
- **Camera calibration**: each frame is undistorted.
- **Vanishing point** (green dot in the image): computed as moving average in the first "vanishingPointWindow" frames of the video.
![alt text](https://image.ibb.co/j8JF8S/2_vanish_point.jpg)  
(Figure 1)  
Only after the computation of the vanishing point, the algorithm start looking for the lanes:
- **Perspective transform** to have a bird view of the image
- **Binary thresholding** of the perspective transform to distinguish the lanes from the road
![alt text](https://image.ibb.co/djExoS/threshold.jpg)  
(Figure 2)  
- **Sliding window method** to classify the two lanes. For each window (red rectangles in Figure 3) centroid (red dots in Figure 3) of the connected component within it is computed. The windows are placed starting from the bottom. The first one is place differently given two different cases:
  - when there was not a curve in the previous frame, it is placed as result of the histogram;
  - when there was a curve in the previous frame, it is placed at the same position of the first window of the previous lane.
  Then the first "partialFittingOrder" windows are places one on top of each other.
  After "partialFittingOrder" windows, every window is place as result of the fitting of the centroids of the previous windows.
- **Polyfit** of the centroids of the windows to find the two lanes (green lines in Figure 3).
![alt text](https://image.ibb.co/focyTS/rectangles.jpg)  
(Figure 3)  

##Authors
**Luca Fucci** and **Umberto Fazio**
from Politecnico di Milano
