# autonomous-mini-4wd

OpenCV installation guide: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

Example video data:
https://mega.nz/#!Nm4kCBzZ!y6kVfObzDOj2Tt8uet7h_UjsCH8HSNY571BSNvWxdQQ

-go to folder Lanes/build
-execute the command "cmake ."
-execute the command "make"

-to run curve_fitting execute the command "./curve_fitting ../data/vid/challenge.MP4"

Up to now the algorithm provides:
- Gaussian Blur
- Perspective Transform*
- Adaptive binary thresholding


* For the time being the perspective transform points are calibrated with respect to the video "challenge.mp4"
