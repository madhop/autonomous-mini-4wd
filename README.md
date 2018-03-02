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

Algorithm:
- Camera calibration: each frame is undistorted.
- Vanishing point (green dot in the image): computed as moving average in the first "vanishingPointWindow" frames of the video.
![alt text](https://image.ibb.co/j8JF8S/2_vanish_point.jpg)
  Only after the computation of the vanishing point, the algorithm start looking for the lanes:
- Perspective transform to have a bird view of the image
- Binary thresholding of the perspective transform to distinguish the lanes from the road
![alt text](https://image.ibb.co/djExoS/threshold.jpg)

**VANISHING POINT**

Up to now the algorithm provides:
- Gaussian Blur
- Perspective Transform:
  For the time being the perspective transform points are hard coded with respect to the video "challenge.mp4",
  later on we will try to find and track the vanishing point and extract the four points from it.
- Adaptive binary thresholding:
  We tried different solutions: the method 'threshold' with OTSU threshold, or 'adaptiveThreshold' with
  ADAPTIVE_THRESH_GAUSSIAN_C threshold, the latter seems to work better.
- Lanes histogram:
  Considering a smaller area of the frame, we compute the accumulation of the white pixels in the x-axis
- Sliding windows:
  When we have no previous curves, starting from the maximum of the histogram we're putting some windows
  on top of each one, computing the barycenter (if they have one) and center the window on it.
  When we have already a curve, we start placing the window as the previous and then center them on their
  actual barycenters (if they have one)
- Lane classification
  We're trying different solutions:
  - Computing the RMSE between the current curve and the last one; if for n frame the RMSE is acceptable,
    the lane may be good
  - Counting the number of changes in direction that a curve shows (within certain tolerances): if a curve
    in one frame shows more than n changes, the lane is bad
- Adaptive curve mask
  When a good lane is detected, we compute a mask shaped on it and use it to refine the area in which
  to look for the next lanes, and we keep the same until a bad curve is detected; then, we reset.

Given different videos, you may need to change some parameters (the width/height of the rectangles, the thresholding, the order of the curve to fit ecc.) in order to get a better performance in each case.
