#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

/// Global variables

int edgeThresh = 1;
int lowThreshold = 50;
int ratio = 3;
int canny_kernel = 3;
int blur_kernel = 5;
char* window_name = "Edge Map";


float movingAverage(float avg, float new_sample){
  int N = 20;
  if(avg == 0.0){
    return new_sample;
  }
  avg -= avg / N;
  avg += new_sample / N;
  return avg;
}

Vec4i rightLaneAvg = Vec4i(0,0,0,0);
Vec4i leftLaneAvg = Vec4i(0,0,0,0);

/** @function main */
int main( int argc, char** argv ){
  //Load video
  VideoCapture cap( argv[1]); // open the default camera
  if(!cap.isOpened()){  // check if we succeeded
    return -1;
  }
  VideoWriter outputVideo;
  outputVideo.open("out.avi", VideoWriter::fourcc('P','I','M','1'), cap.get(CV_CAP_PROP_FPS), Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
  if (!outputVideo.isOpened())
  {
    cout  << "Could not open the output video" << endl;
    return -1;
  }
  /*
  //Load image
  src = imread( argv[1] ); /// Load an image
  if( !src.data ){
  return -1;
}*/
for(;;){
  Mat src, detected_lines;
  //Capture frame
  cap >> src;
  int width = src.size().width;
  int height = src.size().height;

  Mat src_gray;
  cvtColor(src,src_gray,CV_RGB2GRAY);
  Mat wip = Mat::zeros(height,width, CV_8UC1);
  //Crea mask


  #if 0
      Canny(image, image, 50, 200, 3); // Apply canny edge
  #endif
      // Create and LSD detector with standard or no refinement.
  #if 1
      Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
  #else
      Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
  #endif
      vector<Vec4f> lines_std;
      // Detect the lines
      ls->detect(src_gray, lines_std);

      // Show found lines


      //ls->drawSegments(wip, lines_std);
      int x1,x2,y1,y2;
      for( size_t i = 0; i < lines_std.size(); i++ ){
        Vec4i l = lines_std[i];
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];
        line( wip, Point(x1, y1), Point(x2, y2), 255, 3, CV_AA);
      }

      Point points[1][4];
      points[0][0] = Point( width/9,height);
      points[0][1] = Point( width-width/9, height);
      points[0][2] = Point( width/2+width/8, height/2+height/10);
      points[0][3] = Point( width/2-width/8, height/2+height/10);
      const Point* ppt[1] = { points[0] };
      int npt[] = { 4 };
      Mat mask = Mat::zeros(height,width, CV_8UC1);
      fillPoly( mask, ppt, npt, 1, 255 ,8);
      //Applica mask a immagine
      bitwise_and(wip, mask, wip);

      vector<Vec4i> lines;
      HoughLinesP(wip, lines, 1, CV_PI/180, 50, 50, 200 ); //original: HoughLinesP(detected_edges, lines, 1, CV_PI/180, 10, 50, 200 ); http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
      //Find longest right and left lines
      float longestRightLen, longestLeftLen;
      longestRightLen = 0.0;
      longestLeftLen = 0.0;
      Vec4i rightLane, leftLane;
      for( size_t i = 0; i < lines.size(); i++ ){
        Vec4i l = lines[i];
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];
        float slope = (float)(y2-y1)/(x2-x1);
        float len = sqrt(pow(y2-y1,2)+pow(x2-x1,2));
        if(slope < -0.5){
          if(len > longestLeftLen){
            leftLane = l;
            longestLeftLen = len;
          }
        }else if(slope > 0.5){
          if(len > longestRightLen){
            rightLane = l;
            longestRightLen = len;
          }
        }
      }
      //if no lane is detected, substitute with avearge
      if(leftLane[0]==0 && leftLane[1]==0 && leftLane[2]==0 &&  leftLane[3]==0){
        leftLane = leftLaneAvg;
        cout << "left sostituita" << endl;
      }
      if(rightLane[0]==0 && rightLane[1]==0 && rightLane[2]==0 && rightLane[3]==0){
        rightLane = rightLaneAvg;
        cout << "right sostituita" << endl;
      }
      //regularize lines
      int xUp1 = 0; int yUp1 = (height - height/3);
      int xUp2 = width; int yUp2 = (height - height/3);
      int xDown1 = 0; int yDown1 = height;
      int xDown2 = width; int yDown2 = height;

      float m_up = (float)(yUp2-yUp1)/(xUp2-xUp1);
      float m_down = (float)(yDown2-yDown1)/(xDown2-xDown1);
      float q_up = yUp1-m_up*xUp1;
      float q_down = yDown1-m_up*xDown1;


      //find right line; y = m*x+q
      x1 = rightLane[0];
      y1 = rightLane[1];
      x2 = rightLane[2];
      y2 = rightLane[3];
      float m_right = (float)(y2-y1)/(x2-x1);
      float q_right = y1-m_right*x1;
      //right intersection points
      int xIntRight1 = (q_up - q_right)/(m_right - m_up);
      int yIntRight1 = m_right*xIntRight1 + q_right;
      int xIntRight2 = (q_down - q_right)/(m_right - m_down);
      int yIntRight2 = m_right*xIntRight2 + q_right;

      //find left line; y = m*x+q
      x1 = leftLane[0];
      y1 = leftLane[1];
      x2 = leftLane[2];
      y2 = leftLane[3];
      float m_left = (float)(y2-y1)/(x2-x1);
      float q_left = y1-m_left*x1;
      //left intersection points
      int xIntLeft1 = (q_up - q_left)/(m_left - m_up);
      int yIntLeft1 = m_left *xIntLeft1 + q_left;
      int xIntLeft2 = (q_down - q_left)/(m_left - m_down);
      int yIntLeft2 = m_left*xIntLeft2 + q_left;

      //moving average lines
      //right
      rightLaneAvg[0] = movingAverage(rightLaneAvg[0], xIntRight1);
      rightLaneAvg[1] = movingAverage(rightLaneAvg[1], yIntRight1);
      rightLaneAvg[2] = movingAverage(rightLaneAvg[2], xIntRight2);
      rightLaneAvg[3] = movingAverage(rightLaneAvg[3], yIntRight2);
      //left
      leftLaneAvg[0] = movingAverage(leftLaneAvg[0], xIntLeft1);
      leftLaneAvg[1] = movingAverage(leftLaneAvg[1], yIntLeft1);
      leftLaneAvg[2] = movingAverage(leftLaneAvg[2], xIntLeft2);
      leftLaneAvg[3] = movingAverage(leftLaneAvg[3], yIntLeft2);

      detected_lines = Mat::zeros(height,width, src.type());
      //draw lines
      line( detected_lines, Point(rightLane[0], rightLane[1]), Point(rightLane[2], rightLane[3]), Scalar(0,0,255), 3, CV_AA); //red line
      line( detected_lines, Point(leftLane[0], leftLane[1]), Point(leftLane[2], leftLane[3]), Scalar(0,0,255), 3, CV_AA); // red line
      //line( detected_lines, Point(xIntRight1, yIntRight1), Point(xIntRight2, yIntRight2), Scalar(255,255,255), 12); //regularized right line (white)
      //line( detected_lines, Point(xIntLeft1, yIntLeft1), Point(xIntLeft2, yIntLeft2), Scalar(255,255,255), 12); //regularized left line (white)
      line( detected_lines, Point(rightLaneAvg[0], rightLaneAvg[1]), Point(rightLaneAvg[2], rightLaneAvg[3]), Scalar(255,255,255), 12); //regularized right line (white)
      line( detected_lines, Point(leftLaneAvg[0], leftLaneAvg[1]), Point(leftLaneAvg[2], leftLaneAvg[3]), Scalar(255,255,255), 12); //regularized left line (white)


  //Overlap images
  double alpha = 1;
  double beta = 1;
  addWeighted( src, alpha, detected_lines, beta, 0.0, src);

  //cout << leftLaneAvg[0] << " " << leftLaneAvg[1] << " " << leftLaneAvg[2] << " " << leftLaneAvg[3] << endl;


  //Display Image
  namedWindow( window_name, WINDOW_NORMAL );
  cvResizeWindow(window_name, 800, 500);
  imshow( window_name, src );
  //waitKey(0);
  if(waitKey(30) >= 0) break;
  //outputVideo << src;

}
return 0;
}
