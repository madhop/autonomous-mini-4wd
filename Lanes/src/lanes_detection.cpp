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

const Scalar rect_color = Scalar(0,0,255);


void displayImg(char* window_name,Mat mat){
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

Point computeBarycenter(Point p1, Point p2, Point p3, Point p4, Mat mat){
  int yWeight = 0;
  int xWeight = 0;
  Point bar;
  bar.y = 0;
  bar.x = 0;
  for(int j = p1.y; j > p2.y; j--){
    int weight = 0;
    for(int k = p1.x; k < p4.x; k++){
      int intensity = mat.at<uchar>(j, k);
      if(intensity == 255){
        weight ++;
        yWeight++;
      }
    }
    bar.y += j*weight;
  }
  for(int j = p1.x; j < p4.x; j++){
    int weight = 0;
    for(int k = p1.y; k > p2.y; k--){
      int intensity = mat.at<uchar>(k, j);
      if(intensity == 255){
        weight ++;
        xWeight++;
      }
    }
    bar.x += j*weight;
  }
  if(xWeight!=0 && yWeight!=0){ //if no line is detected no barycenter is added
    bar.y /= yWeight;
    bar.x /= xWeight;
    return bar;
  }else{
    return Point(NULL,NULL);
  }
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
  char* window_3 = "Mask";
  namedWindow( window_3, WINDOW_NORMAL );
  cvResizeWindow(window_3, 800, 500);
  imshow( window_3, mask );
  bitwise_and(mat,mask,mat);
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
vector<Point> fittedLeft;
vector<Point> fittedRight;
vector<Point> rightBarycenters;
vector<Point> leftBarycenters;
vector<Point> rightRectCenters;
vector<Point> leftRectCenters;
int counter = 0;  //serve per non fare la maschera al primo ciclo quando non ho ancora le linee
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

  Mat rectangles = wip;
  /*
  if(counter>0){
    Mat mask = Mat::zeros(height,width, CV_8UC1);
    polylines( mask, fittedLeft, 0, 255, mask_offset, 0);
    polylines( mask, fittedRight, 0, 255, mask_offset, 0);
    char* window_3 = "Mask";
    namedWindow( window_3, WINDOW_NORMAL );
    cvResizeWindow(window_3, 800, 500);
    imshow( window_3, mask );
    bitwise_and(wip,mask,wip);
  }*/



  cvtColor( rectangles, rectangles, CV_GRAY2BGR );



  rightBarycenters = vector<Point>();
  leftBarycenters = vector<Point>();
  //Initialize rectangles
  if(counter==0){//Se non ho le linee
    rightRectCenters = vector<Point>();
    leftRectCenters = vector<Point>();
    //First rectangle
    vector<int> acc = findHistAcc(wip);
    int leftFirstX = acc[0];
    int rightFirstX = acc[1];
    leftRectCenters.push_back(Point(leftFirstX, height - rect_offset - rect_height/2));
    rightRectCenters.push_back(Point(rightFirstX, height - rect_offset - rect_height/2));
    //Other rectangles
    for(int i=0;i<n_rect;i++){
      //Compute left rectangle
      Point lr1 = Point(leftRectCenters[i].x - rect_width/2, leftRectCenters[i].y + rect_height/2);
      Point lr2 = Point(leftRectCenters[i].x - rect_width/2, leftRectCenters[i].y - rect_height/2);
      Point lr3 = Point(leftRectCenters[i].x + rect_width/2, leftRectCenters[i].y - rect_height/2);
      Point lr4 = Point(leftRectCenters[i].x + rect_width/2, leftRectCenters[i].y + rect_height/2);
      //Compute right rectangle
      Point rr1 = Point(rightRectCenters[i].x - rect_width/2, rightRectCenters[i].y + rect_height/2);
      Point rr2 = Point(rightRectCenters[i].x - rect_width/2, rightRectCenters[i].y - rect_height/2);
      Point rr3 = Point(rightRectCenters[i].x + rect_width/2, rightRectCenters[i].y - rect_height/2);
      Point rr4 = Point(rightRectCenters[i].x + rect_width/2, rightRectCenters[i].y + rect_height/2);

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

      //Compute barycenters and rectangle centers
      Point nextLeftCenter = Point();
      Point nextRightCenter = Point();

      Point leftBar = computeBarycenter(lr1,lr2,lr3,lr4,wip);
      if(leftBar.x!=NULL && leftBar.y!=NULL){ //if no line is detected no barycenter is added
        leftBarycenters.push_back(leftBar);
        circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        nextLeftCenter.x = leftBar.x;
      }
      else{
        nextLeftCenter.x = rightRectCenters[i].x;
      }
      nextLeftCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;

      Point rightBar = computeBarycenter(rr1,rr2,rr3,rr4,wip);
      if(rightBar.x!=NULL && rightBar.y!=NULL){
        rightBarycenters.push_back(rightBar);
        circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        nextRightCenter.x = rightBar.x;
      }else{
        nextRightCenter.x = rightRectCenters[i].x;
      }
      nextRightCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;

      if(i<n_rect-1){
        rightRectCenters.push_back(nextRightCenter);
        leftRectCenters.push_back(nextLeftCenter);
      }


    }
    cout << "Left centers" << leftRectCenters << endl;
    cout << "Right centers" << rightRectCenters << endl;

  }else{ //Se ho le linee

    //Create mask based on previous curves
    //wip = curve_mask(fittedRight,fittedLeft,wip,mask_offset);

    for(int i=0;i<n_rect;i++){

      //Compute left rectangle
      Point lr1 = Point(leftRectCenters[i].x - rect_width/2, leftRectCenters[i].y + rect_height/2);
      Point lr2 = Point(leftRectCenters[i].x - rect_width/2, leftRectCenters[i].y - rect_height/2);
      Point lr3 = Point(leftRectCenters[i].x + rect_width/2, leftRectCenters[i].y - rect_height/2);
      Point lr4 = Point(leftRectCenters[i].x + rect_width/2, leftRectCenters[i].y + rect_height/2);
      //Compute right rectangle
      Point rr1 = Point(rightRectCenters[i].x - rect_width/2, rightRectCenters[i].y + rect_height/2);
      Point rr2 = Point(rightRectCenters[i].x - rect_width/2, rightRectCenters[i].y - rect_height/2);
      Point rr3 = Point(rightRectCenters[i].x + rect_width/2, rightRectCenters[i].y - rect_height/2);
      Point rr4 = Point(rightRectCenters[i].x + rect_width/2, rightRectCenters[i].y + rect_height/2);

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


      Point leftBar = computeBarycenter(lr1,lr2,lr3,lr4,wip);
      if(leftBar.x!=NULL && leftBar.y!=NULL){ //if no line is detected no barycenter is added
        //leftRectCenters[i].x = leftBar.x; //comment for fixed rectangles
        circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        leftBarycenters.push_back(leftBar);
      }
      Point rightBar = computeBarycenter(rr1,rr2,rr3,rr4,wip);
      if(rightBar.x!=NULL && rightBar.y!=NULL){
        //rightRectCenters[i].x = rightBar.x; //comment for fixed rectangles
        circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        rightBarycenters.push_back(rightBar);
      }

  }
}

  //LEAST SQUARES SECOND ORDER POLYNOMIAL FITTING
  // x = beta_2*y^2 + beta_1*y + beta_0
  fittedLeft = polyFit(leftBarycenters,wip);
  fittedRight = polyFit(rightBarycenters,wip);

  polylines( rectangles, fittedLeft, 0, Scalar(0,255,0) ,8,0);
  polylines( rectangles, fittedRight, 0, Scalar(0,255,0) ,8,0);


  //Display Image
  displayImg("Wip",wip);
  displayImg("Rectangles",rectangles);

  waitKey(0);
  //if(waitKey(30) >= 0) break;
  //outputVideo << src;
  counter++;
}
return 0;
}
