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
float min_slope = 0.3;
Vec4i rightLaneAvg = Vec4i(0,0,0,0);
Vec4i leftLaneAvg = Vec4i(0,0,0,0);





float movingAverage(float avg, float new_sample){
  int N = 20;
  if(avg == 0.0){
    return new_sample;
  }
  avg -= avg / N;
  avg += new_sample / N;
  return avg;
}

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
  Mat src, detected_lines, wip;
  int x1,x2,y1,y2;
  vector<Vec4i> lines;
  //Capture frame
  cap >> src;
  int width = src.size().width;
  int height = src.size().height;
  detected_lines = Mat::zeros(height,width, CV_8UC1);
  wip = src;
  cvtColor( wip, wip, CV_BGR2GRAY );

  for ( int i = 1; i < blur_kernel; i = i + 2 ){
    GaussianBlur( wip, wip, Size( i, i ), 0, 0 );
  }

  //Canny( wip, wip, lowThreshold, lowThreshold*ratio, canny_kernel );


  //Perspective Transform
  Point2f inPoints[4];
  inPoints[0] = Point2f( 0, height );
  inPoints[1] = Point2f( width/2-width/8, height/2+height/6);
  inPoints[2] = Point2f( width/2+width/8, height/2+height/6);
  inPoints[3] = Point2f( width, height);

  Point2f outPoints[4];
  outPoints[0] = Point2f( 0,height);
  outPoints[1] = Point2f( 0, 0);
  outPoints[2] = Point2f( width, 0);
  outPoints[3] = Point2f( width, height);

  // Set the lambda matrix the same type and size as input
  Mat lambda = Mat::zeros( height, width , src.type() );

  // Get the Perspective Transform Matrix i.e. lambda
  lambda = getPerspectiveTransform( inPoints, outPoints );
  // Apply the Perspective Transform just found to the src image
  warpPerspective(wip,wip,lambda,wip.size() );

  //inverse
  /*
  // Get the Perspective Transform Matrix i.e. lambda
  lambda = getPerspectiveTransform( outPoints, inPoints );
  // Apply the Perspective Transform just found to the src image
  warpPerspective(wip,wip,lambda,wip.size() );
  */
  //Color Filtering
  //White Filter
  inRange(wip, Scalar(150, 150, 150), Scalar(255, 255, 255), wip);
  //cvtColor( wip, wip, CV_BGR2GRAY );

  adaptiveThreshold(wip,wip,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,55,-20);
  //threshold(wip,wip,0,255,THRESH_BINARY | THRESH_OTSU);
  //threshold(wip,wip,THRESH_OTSU,255,THRESH_OTSU);


  //Compute Histogram
  int histogram[width];
  int max = 0;
  for(int i = 0; i<width; i++){
    int sum = 0;
    for(int j = 0; j<height; j++){
      Scalar intensity = wip.at<uchar>(j, i);
      //cout << intensity.val[0] << endl;
      if(intensity.val[0] == 255){
        sum++;
      }
      histogram[i] = sum;
      if(sum > max){
        max = sum;
      }
    }
  }

  //Find max left and min left
  //the loop could be included in the computation of the
  //histogram but anyway later on we will change this with Gaussian Fitting
  //First half
  int leftMax = 0;
  int leftMaxPos = 0;
  for(int i = 0; i < width/2; i++){
    if(histogram[i] > leftMax){
      leftMax++;
      leftMaxPos = i;
    }
  }
  int rightMax = 0;
  int rightMaxPos = 0;
  //Second half
  for(int i = width/2; i < width; i++){
    if(histogram[i] > leftMax){
      rightMax++;
      rightMaxPos = i;
    }
  }


  /*
  //Display Histogram
  Mat hist =  Mat::zeros( max, width, wip.type() );

  for(int i = 0; i<width;i++){
  hist.at<uchar>(max - histogram[i] - 1, i) = 255;
  //cout << max- histogram[i] << endl;
}

char* window_2 = "Histogram";
namedWindow( window_2, WINDOW_NORMAL );
cvResizeWindow(window_2, 800, 500);
imshow( window_2, hist );
*/

//Barycenter computation
//Mat rectangles = Mat::zeros( height, width, src.type() );
Mat rectangles = wip;
cvtColor( rectangles, rectangles, CV_GRAY2BGR );
int rect_width = width/10;
int rect_offset = 0;
int n_rect = 10;
int rect_height = (height - rect_offset)/n_rect;
Scalar rect_color = Scalar(0,0,255);
int rect_thickness = 2;


int leftBarX = leftMaxPos;
int leftBarY = height - rect_offset;
int rightBarX = rightMaxPos;
int rightBarY = height - rect_offset;
for(int i=0;i<n_rect;i++){
  cout << leftBarX << endl;
  //Compute left rectangle ... per adesso non usiamo la y del baricentro ma costruiamo il rettangolo a partire dal tetto di quello sotto
  Point lr1 = Point(leftBarX - rect_width/2, height - rect_offset - i*rect_height);
  Point lr2 = Point(leftBarX - rect_width/2, height - rect_offset - (i+1)*rect_height);
  Point lr3 = Point(leftBarX + rect_width/2, height - rect_offset - (i+1)*rect_height);
  Point lr4 = Point(leftBarX + rect_width/2, height - rect_offset - i*rect_height);
  //Compute right rectangle
  Point rr1 = Point(rightBarX - rect_width/2, height - rect_offset - i*rect_height);
  Point rr2 = Point(rightBarX - rect_width/2, height - rect_offset - (i+1)*rect_height);
  Point rr3 = Point(rightBarX + rect_width/2, height - rect_offset - (i+1)*rect_height);
  Point rr4 = Point(rightBarX + rect_width/2, height - rect_offset - i*rect_height);


  //Draw left rectangle
  line( rectangles, lr1, lr2, rect_color, rect_thickness, CV_AA);
  line( rectangles, lr2, lr3, rect_color, rect_thickness, CV_AA);
  line( rectangles, lr3, lr4, rect_color, rect_thickness, CV_AA);
  line( rectangles, lr4, lr1, rect_color, rect_thickness, CV_AA);
  //Draw right rectangle
  line( rectangles, rr1, rr2, rect_color, rect_thickness, CV_AA);
  line( rectangles, rr2, rr3, rect_color, rect_thickness, CV_AA);
  line( rectangles, rr3, rr4, rect_color, rect_thickness, CV_AA);
  line( rectangles, rr4, rr1, rect_color, rect_thickness, CV_AA);



  //Compute left barycenter
  int yWeight = 0;
  for(int j = lr1.y; j > lr2.y; j--){
    int weight = 0;
    for(int k = lr1.x; k < lr4.x; k++){
      int intensity = wip.at<uchar>(j, k);
      if(intensity == 255){
        weight ++;
        yWeight++;
      }
    }
    leftBarY += j*weight;
  }
  if(yWeight!=0){
    leftBarY /= yWeight;
  }
  int xWeight = 0;
  for(int j = lr1.x; j < lr4.x; j++){
    int weight = 0;
    for(int k = lr1.y; k > lr2.y; k--){
      int intensity = wip.at<uchar>(k, j);
      if(intensity == 255){
        weight ++;
        xWeight++;
      }
    }
    leftBarX += j*weight;
  }
  if(xWeight!=0){
    leftBarX /= xWeight;
  }

  //Draw left barycenter
  circle( rectangles, Point(leftBarX,leftBarY), 5, Scalar( 0, 0, 255 ),  3, 3 );

  //Compute right barycenter
  yWeight = 0;
  for(int j = rr1.y; j > rr2.y; j--){
    int weight = 0;
    for(int k = rr1.x; k < rr4.x; k++){
      int intensity = wip.at<uchar>(j, k);
      if(intensity == 255){
        weight ++;
        yWeight++;
      }
    }
    rightBarY += j*weight;
  }
  if(yWeight!=0){
    rightBarY /= yWeight;
  }
  xWeight = 0;
  for(int j = rr1.x; j < rr4.x; j++){
    int weight = 0;
    for(int k = rr1.y; k > rr2.y; k--){
      int intensity = wip.at<uchar>(k, j);
      if(intensity == 255){
        weight ++;
        xWeight++;
      }
    }
    rightBarX += j*weight;
  }
  if(xWeight!=0){
    rightBarX /= xWeight;
  }
  circle( rectangles, Point(rightBarX,rightBarY), 5, Scalar( 0, 0, 255 ),  3, 3 );

}

//cvtColor( wip, wip, CV_GRAY2BGR );
//addWeighted( rectangles, 1, wip, 0, 0.0, rectangles);
//bitwise_or(wip,rectangles,rectangles);

//Display Image
char* window_1 = "Result";
namedWindow( window_1, WINDOW_NORMAL );
cvResizeWindow(window_1, 800, 500);
imshow( window_1, wip );
char* window_2 = "Rectangles";
namedWindow( window_2, WINDOW_NORMAL );
cvResizeWindow(window_2, 800, 500);
imshow( window_2, rectangles );

waitKey(0);
//if(waitKey(30) >= 0) break;
//outputVideo << src;

}
return 0;
}
