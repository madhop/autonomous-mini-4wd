#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

/// Global variables
#define canny_low_threshold 50
#define canny_high_threshold_ratio 3
#define canny_kernel 3
#define blur_kernel 5
#define mask_offset 300
#define rect_width_ratio 10
#define rect_offset_ratio 20
#define n_rect 10
#define rect_thickness 2
#define weight_threshold 10

const Scalar rect_color = Scalar(0,0,255);

vector<Point> computeRect(Point center, int rect_width,int rect_height){ //given the center of the rectangle compute the 4 vertex
  vector<Point> points;
  points.push_back(Point(center.x - rect_width/2, center.y + rect_height/2));
  points.push_back(Point(center.x - rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y + rect_height/2 ));
  return points;
}

void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles){ //draw the rectangles
  line( rectangles, rect_points[0], rect_points[1], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[1], rect_points[2], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[2], rect_points[3], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[3], rect_points[0], rect_color, thickness, CV_AA);
}

void displayImg(const char* window_name,Mat mat){
  namedWindow( window_name, WINDOW_NORMAL );
  cvResizeWindow(window_name, 800, 500);
  imshow( window_name, mat );
}

Mat perspectiveTransform(Mat mat){
  //Perspective Transform
  int width = mat.size().width;
  int height = mat.size().height;
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
  Mat lambda = Mat::zeros( height, width , mat.type() );

  // Get the Perspective Transform Matrix i.e. lambda
  lambda = getPerspectiveTransform( inPoints, outPoints );
  // Apply the Perspective Transform just found to the src image
  warpPerspective(mat,mat,lambda,mat.size() );
  return mat;
}

float movingAverage(float avg, float new_sample){
  int N = 20;
  if(avg == 0.0){
    return new_sample;
  }
  avg -= avg / N;
  avg += new_sample / N;
  return avg;
}

Point computeBarycenter(vector<Point> points, Mat mat){
  int totWeight = 0;
  Point bar;
  bar.y = 0;
  bar.x = 0;
  for(int j = points[0].y; j > points[1].y; j--){
    int weight = 0;
    for(int k = points[0].x; k < points[3].x; k++){
      int intensity = mat.at<uchar>(j, k);
      if(intensity == 255){
        weight ++;
        totWeight++;
      }
    }
    bar.y += j*weight;
  }
  totWeight=0;
  for(int j = points[0].x; j < points[3].x; j++){
    int weight = 0;
    for(int k = points[0].y; k > points[1].y; k--){
      int intensity = mat.at<uchar>(k, j);
      if(intensity == 255){
        weight ++;
        totWeight++;
      }
    }
    bar.x += j*weight;
  }
  if(totWeight>weight_threshold){//xWeight!=0 && yWeight!=0){ //if no line is detected no barycenter is added or even if it's just a random bunch of pixels
    bar.y /= totWeight;
    bar.x /= totWeight;
  }else{
    bar.x = -1;
    bar.y = -1;
  }
  return bar;
}

vector<Point> polyFit(vector<Point> points,Mat mat){
  vector<Point> fittedPoints;
  int width = mat.size().width;
  int height = mat.size().height;
  if(points.size() >= 3){
    Mat X = Mat::zeros( points.size(), 1 , CV_32F );
    Mat y = Mat::zeros( points.size(), 3 , CV_32F );
    Mat beta; //= Mat::zeros( 3, 1 , CV_32F );
    //matrix Y
    for(int i = 0; i < y.rows; i++){
      for(int j = 0; j < y.cols; j++){
        y.at<float>(i,j) = pow(points[i].y,j);
      }
    }
    //matrix x
    for(int i = 0; i < X.rows; i++){
      X.at<float>(i,0) = points[i].x;
    }
    beta = y.inv(DECOMP_SVD)*X;//leftBeta = ((leftX.t()*leftX).inv()*leftX.t())*leftY;
    fittedPoints = vector<Point>();

    for(int i = 0; i<height; i++){
      float fittedX = beta.at<float>(2,0)*pow(i,2)+beta.at<float>(1,0)*i+beta.at<float>(0,0);
      Point fp = Point(fittedX,i);
      //circle( rectangles, fp, 5, Scalar( 0, 255, 0 ),  3, 3 );
      fittedPoints.push_back(fp);
    }
  }
  return fittedPoints;
}

vector<int> findHistAcc(Mat mat){
  int width = mat.size().width;
  int height = mat.size().height;
  //Compute Histogram
  int histogram[width];
  int max = 0;
  for(int i = 0; i<width; i++){
    int sum = 0;
    for(int j = height/2; j < height; j ++){
      Scalar intensity = mat.at<uchar>(j, i);
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
      leftMax = histogram[i];
      leftMaxPos = i;
    }
  }
  int rightMax = 0;
  int rightMaxPos = 0;
  //Second half
  for(int i = width/2; i < width; i++){
    if(histogram[i] > rightMax){
      rightMax = histogram[i];
      rightMaxPos = i;
    }
  }
  vector<int> acc;
  acc.push_back(leftMaxPos);
  acc.push_back(rightMaxPos);
  return acc;
}

Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset){
  int height = mat.size().height;
  int width = mat.size().width;
  Mat mask = Mat::zeros(height,width, CV_8UC1);
  polylines( mask, curve1, 0, 255, offset, 0);
  polylines( mask, curve2, 0, 255, offset, 0);
  bitwise_and(mat,mask,mat);
  displayImg("Mask",mask);
  return mat;
}

float computeRmse(vector<Point> curve1, vector<Point> curve2){
  float rmse = 0;
  if( curve1.size()>0){
    //RMSE
    for(int i=0; i<curve1.size();i++){
      rmse+=pow(curve1[i].x-curve2[i].x,2)/curve1.size();
    }
    rmse = sqrt(rmse);
  }
  return rmse;
}

/** @function main */
int main( int argc, char** argv ){
  //Load video
  VideoCapture cap( argv[1]); // open the default camera
  if(!cap.isOpened()){  // check if we succeeded
    return -1;
  }
  /*
  //Write video
  VideoWriter outputVideo;
  outputVideo.open("out.avi", VideoWriter::fourcc('P','I','M','1'), cap.get(CV_CAP_PROP_FPS), Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
  if (!outputVideo.isOpened())
  {
    cout  << "Could not open the output video" << endl;
    return -1;
  }
  */
  /*
  //Load image
  src = imread( argv[1] ); /// Load an image
  if( !src.data ){
  return -1;
}
*/
vector<Point> mask_curve_left;
vector<Point> mask_curve_right;
vector<Point> lastOkFittedRight;
vector<Point> lastOkFittedLeft;
vector<Point> rightBarycenters;
vector<Point> leftBarycenters;
vector<Point> lastOkRightRectCenters;
vector<Point> lastOkLeftRectCenters;
bool left_ok = false;  //serve per non fare la maschera al primo ciclo quando non ho ancora le linee
bool right_ok = false;
for(;;){
  Mat src, wip;
  //Capture frame
  cap >> src;
  int width = src.size().width;
  int height = src.size().height;
  const int rect_width = width/rect_width_ratio;
  const int rect_offset = height/rect_offset_ratio;
  const int rect_height = (height - rect_offset)/n_rect;
  wip = src;

  cvtColor( wip, wip, CV_BGR2GRAY );

  for ( int i = 1; i < blur_kernel ; i = i + 2 ){
    GaussianBlur( wip, wip, Size( i, i ), 0, 0 );
  }

  //Canny( wip, wip, canny_low_threshold, canny_low_threshold*canny_high_threshold_ratio, canny_kernel );

  wip = perspectiveTransform(wip);


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



  //Curve Mask
  if(left_ok && right_ok){
    wip = curve_mask(mask_curve_right,mask_curve_left,wip,mask_offset);
  }



  Mat rectangles = wip;
  cvtColor( rectangles, rectangles, CV_GRAY2BGR );

  rightBarycenters = vector<Point>();
  leftBarycenters = vector<Point>();
  vector<Point> leftRectCenters;
  vector<Point> rightRectCenters;
  //Initialize rectangles
  if(left_ok==false){//Se non ho left
    leftRectCenters = vector<Point>();
    //First rectangle
    vector<int> acc = findHistAcc(wip);
    int leftFirstX = acc[0];
    leftRectCenters.push_back(Point(leftFirstX, height - rect_offset - rect_height/2));
    //Other rectangles
    for(int i=0;i<n_rect;i++){
      //Compute left rectangle
      vector<Point> left_rect = computeRect(leftRectCenters[i], rect_width, rect_height);
      //Draw left rectangle
      drawRect(left_rect, rect_color, rect_thickness, rectangles);
      //Compute barycenters and rectangle centers
      Point nextLeftCenter = Point();

      Point leftBar = computeBarycenter(left_rect ,wip);
      if(leftBar.x!=-1 && leftBar.y!=-1){ //if no line is detected no barycenter is added
        leftBarycenters.push_back(leftBar);
        circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        nextLeftCenter.x = leftBar.x;
      }
      else{
        nextLeftCenter.x = rightRectCenters[i].x;
      }
      nextLeftCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;


      if(i<n_rect-1){ //aggiunge il quadrato successivo tranne all'ultimo giro
        leftRectCenters.push_back(nextLeftCenter);
      }
    }
  }
  else { //Se ho left
    leftRectCenters = lastOkLeftRectCenters;
    for(int i=0;i<n_rect;i++){
      //Compute left rectangle
      vector<Point> left_rect = computeRect(leftRectCenters[i], rect_width, rect_height);
      //Draw left rectangle
      drawRect(left_rect, rect_color, rect_thickness, rectangles);

      Point leftBar = computeBarycenter(left_rect ,wip);
      if(leftBar.x!=-1 && leftBar.y!=-1){ //if no line is detected no barycenter is added
          leftRectCenters[i].x = leftBar.x; //comment for fixed rectangles
          circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
          leftBarycenters.push_back(leftBar);
      }
  }
}
  if(right_ok==false){//Se non ho right
    rightRectCenters = vector<Point>();
    //First rectangle
    vector<int> acc = findHistAcc(wip);
    int rightFirstX = acc[1];
    rightRectCenters.push_back(Point(rightFirstX, height - rect_offset - rect_height/2));
    //Other rectangles
    for(int i=0;i<n_rect;i++){
      //Compute right rectangle
      vector<Point> right_rect = computeRect(rightRectCenters[i], rect_width, rect_height);

      //Draw right rectangle
      drawRect(right_rect, rect_color, rect_thickness, rectangles);

      //Compute barycenters and rectangle centers
      Point nextRightCenter = Point();

      Point rightBar = computeBarycenter(right_rect ,wip);
      if(rightBar.x!=-1 && rightBar.y!=-1){
        rightBarycenters.push_back(rightBar);
        circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        nextRightCenter.x = rightBar.x;
      }else{
        nextRightCenter.x = rightRectCenters[i].x;
      }
      nextRightCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;

      if(i<n_rect-1){
        rightRectCenters.push_back(nextRightCenter);
      }
    }

  }
else {//Se ho right
   rightRectCenters = lastOkRightRectCenters;
    for(int i=0;i<n_rect;i++){
      //Compute right rectangle
      vector<Point> right_rect = computeRect(rightRectCenters[i], rect_width, rect_height);
      //Draw right rectangle
      drawRect(right_rect, rect_color, rect_thickness, rectangles);

      Point rightBar = computeBarycenter(right_rect ,wip);
      if(rightBar.x!=-1 && rightBar.y!=-1){
        rightRectCenters[i].x = rightBar.x; //comment for fixed rectangles
        circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        rightBarycenters.push_back(rightBar);
      }

    }
  }



  //LEAST SQUARES SECOND ORDER POLYNOMIAL FITTING
  // x = beta_2*y^2 + beta_1*y + beta_0
  vector<Point> fittedLeft = polyFit(leftBarycenters,wip);
  vector<Point> fittedRight = polyFit(rightBarycenters,wip);

  polylines( rectangles, lastOkFittedRight, 0, Scalar(255,0,0) ,8,0);
  polylines( rectangles, lastOkFittedLeft, 0, Scalar(255,0,0) ,8,0);
  polylines( rectangles, fittedLeft, 0, Scalar(0,255,0) ,8,0);
  polylines( rectangles, fittedRight, 0, Scalar(0,255,0) ,8,0);



  //If there's at least one sound curve and the curve at the current frame is sound as well
  if( mask_curve_right.size()>0 && fittedRight.size()>0){
    float rightRmse;
    rightRmse = computeRmse(fittedRight,lastOkFittedRight);
    if(rightRmse < 10){
      lastOkFittedRight = fittedRight;
      lastOkRightRectCenters = rightRectCenters;
    }
  }

  if( mask_curve_left.size()>0 && fittedLeft.size()>0){
    float leftRmse;
    leftRmse = computeRmse(fittedLeft,lastOkFittedLeft);
    if(leftRmse < 10){
      lastOkFittedLeft = fittedLeft;
      lastOkLeftRectCenters = leftRectCenters;
    }
  }


  //implement condition to switch from false to true
  if(left_ok==false){
    mask_curve_left = fittedLeft;
    lastOkFittedLeft = fittedLeft;
    lastOkLeftRectCenters = leftRectCenters;
  }
  if(right_ok==false){
    mask_curve_right = fittedRight;
    lastOkFittedRight = fittedRight;
    lastOkRightRectCenters = rightRectCenters;
  }
  left_ok=true;
  right_ok=true;



  //Display Image
  displayImg("Wip",wip);
  displayImg("Rectangles",rectangles);

  waitKey(0);
  //if(waitKey(30) >= 0) break;
  //outputVideo << src;
}
return 0;
}