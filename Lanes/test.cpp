#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

//* Global variables *
#define canny_low_threshold 50
#define canny_high_threshold_ratio 3
#define canny_kernel 3
#define blur_kernel 5
#define mask_offset_ratio 3
#define rect_width_ratio 10
#define rect_offset_ratio 20
#define n_rect 10
#define rect_thickness_ratio 200
#define tot_min_weight 10
#define max_dir_changes 5
#define straight_tolerance_ratio 80
#define max_rmse_ratio 70
#define max_bad_curves 3
#define min_good_curves 1
#define min_barycenters 5 //in realtà andrebbe messo come ratio e diviso per n_rect
#define next_bary_max_distance 50 //anche qui va messo ratio
#define rmse_tolerance 20
#define min_similar_curves 3
#define adj_rmse_threshold 30
#define n_long_lines 20 //number of lines for vanishing point
#define max_slope 10
#define min_slope 0.1
#define window_width 800
#define window_height 500
#define horizon_offset_ratio 5
#define straight_range 30 //cambiare con ratio

//* Colors *
const Scalar rect_color = Scalar(0,0,255);
const Scalar last_ok_fitted_color = Scalar(255,0,0);
const Scalar avg_curve_avg = Scalar(0,255,255);
const Scalar cur_fitted_color = Scalar(0,255,0);
const Scalar white_filtering_threshold = Scalar(120, 120, 120);

//* Prototypes *
vector<Point> computeRect(Point center, int rect_width,int rect_height);
void drawRect(vector<Point> rect_points, Scalar rect_color, int height, Mat rectangles);
void displayImg(const char* window_name,Mat mat);
Mat perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints);
float movingAverage(float avg, float new_sample);
Point computeBarycenter(vector<Point> points, Mat mat);
vector<float> polyFit(vector<Point> points,Mat mat, int order);
int findHistAcc(Mat mat, int pos);
Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset);
float computeRmse(vector<Point> curve1, vector<Point> curve2);
int dirChanges(vector<Point> points, int tolerance);
bool classifyCurve(vector<Point> &fittedCurve, bool &some_curve, int &curve_similar_series, int &curve_bad_series, int &curve_ok_series, vector<Point> &lastFittedCurve, vector<Point> &lastOkFittedCurve, vector<Point> &lastOkCurveRectCenters, vector<Point> &curveRectCenters, vector<float> beta, vector<float> &lastOkBeta);
int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, vector<Point> &barycenters, int pos, Mat wip, int width, int height, int rect_offset, int rect_height, int rect_width, Mat rectangles, vector<Point> &lastOkRectCenters); //pos: 0=left, 1=right
vector<Point2f> findPerspectiveInPoints(Mat src);
vector<Point> computePoly(vector<float> beta, int n_points);
int computeDirection(float actualPos, float desiredPos);

/** @function main */
int main( int argc, char** argv ){
  //* Load video *
  VideoCapture cap( argv[1]); // open the default camera
  if(!cap.isOpened()){  // check if we succeeded
    return -1;
  }

  //* Open video to write *
  /*
  VideoWriter outputVideo;
  outputVideo.open("out.avi", VideoWriter::fourcc('P','I','M','1'), cap.get(CV_CAP_PROP_FPS), Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
  if (!outputVideo.isOpened())
  {
  cout  << "Could not open the output video" << endl;
  return -1;
}
*/


vector<Point> lastOkFittedRight;
vector<Point> lastOkFittedLeft;
vector<Point> lastOkRightRectCenters;
vector<Point> lastOkLeftRectCenters;
vector<Point> lastFittedRight;
vector<Point> lastFittedLeft;
vector<Point2f> perspTransfInPoints;
vector<float> lastOkBetaLeft;
vector<float> lastOkBetaRight;

bool some_left = false;
bool some_right = false;
int left_bad_series = 0;
int right_bad_series = 0;
int right_ok_series = 0;
int left_ok_series = 0;
int right_similar_series = 0;
int left_similar_series = 0;

int counter = 0;
for(;;){
  cout << "* frame *" << endl;

  //* Capture frame *
  Mat src, wip;
  cap >> src;
  int width = src.size().width;
  int height = src.size().height;
  const int rect_width = width/rect_width_ratio;
  const int rect_offset = height/rect_offset_ratio;
  const int rect_height = (height - rect_offset)/n_rect;
  const int straight_tolerance = width/straight_tolerance_ratio;
  const int max_rmse = height/max_rmse_ratio; //height perchè la parabola orizzontale è calcolata da x a y

  //compute binary image/
  Mat vanishingPointMap = src.clone();
  Mat lightnessMat, saturationMat;
  Mat grayMat = src.clone();

  for ( int i = 1; i < blur_kernel ; i = i + 2 ){
    GaussianBlur( vanishingPointMap, vanishingPointMap, Size( i, i ), 0, 0, BORDER_DEFAULT );
    GaussianBlur( grayMat, grayMat, Size( i, i ), 0, 0, BORDER_DEFAULT );
  }

  cvtColor( vanishingPointMap, vanishingPointMap, CV_BGR2HLS );
  Mat planes[3];
  split(vanishingPointMap,planes);
  vanishingPointMap = planes[2];
  saturationMat = planes[2];
  lightnessMat = planes[1];
  //displayImg("lightnessMat", lightnessMat);
  //displayImg("planes[0]", planes[0]);
  displayImg("planes[2]", planes[2]);

  //compute avg lightness
  int sum = 0;
  int n_pixel = 0;
  for(int i = 0; i < height; i++){
    for(int j = 0; j< width; j++){
      n_pixel++;
      sum += lightnessMat.at<uchar>(i,j);
    }
  }
  float lightnessAvg = sum/n_pixel;
  cout << "lightness: " << lightnessAvg << endl;

  //change s_channel based on l_channel
  for(int i = 0; i < height; i++){
    for(int j = 0; j< width; j++){
      if(lightnessMat.at<uchar>(i,j) < lightnessAvg){
        vanishingPointMap.at<uchar>(i,j) = 0;
      }else{
        vanishingPointMap.at<uchar>(i,j)
      }
    }
  }
  displayImg("vanishingPointMapThres", vanishingPointMap);

  //sobelx
  cvtColor( grayMat, grayMat, CV_BGR2GRAY );
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y, combined_binary;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;
  double min, max;
  Sobel( grayMat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  //Sobel( grayMat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  //convertScaleAbs( grad_y, abs_grad_y );
  //bitwise_and(abs_grad_x, abs_grad_y, abs_grad_x);

  inRange(vanishingPointMap, 50,255, vanishingPointMap); //Scalar(150, 150, 150)
  adaptiveThreshold(vanishingPointMap,vanishingPointMap,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,105,0);
  //threshold(vanishingPointMap,vanishingPointMap,0,255,THRESH_BINARY | THRESH_OTSU);

  inRange(abs_grad_x, 50, 255 , abs_grad_x); //Scalar(255, 255, 255)
  adaptiveThreshold(abs_grad_x,abs_grad_x,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,33,0);
  //threshold(abs_grad_x,abs_grad_x,0,255,THRESH_BINARY | THRESH_OTSU);
  //threshold(abs_grad_x,abs_grad_x,THRESH_OTSU,255,THRESH_OTSU);

  displayImg("abs_grad_x", abs_grad_x);
  //displayImg("vanishingPointMapThres", vanishingPointMap);

  bitwise_or(abs_grad_x, vanishingPointMap, combined_binary);

  displayImg("combined_binary", combined_binary);

  //end compute binary image/


  wip = combined_binary;//src;

  //* perspective Transform *
  vector<Point2f> perspTransfOutPoints;
  if(counter==0){
    perspTransfInPoints = findPerspectiveInPoints(src);
  }
  if(perspTransfInPoints.size()>0){ //If vanishing point has been found
    perspTransfOutPoints.push_back(Point2f( 0,height));
    perspTransfOutPoints.push_back(Point2f( 0, 0));
    perspTransfOutPoints.push_back(Point2f( width, 0));
    perspTransfOutPoints.push_back(Point2f( width, height));
    wip = perspectiveTransform(wip, perspTransfInPoints, perspTransfOutPoints);
  }

  //* Curve Mask *
  if(some_right && some_left){
    int mask_offset = height/mask_offset_ratio;
    Mat mask = curve_mask(lastOkFittedRight,lastOkFittedLeft,wip,mask_offset);
    bitwise_and(wip,mask,wip);
    //displayImg("Mask",mask);
  }

  //* Find curve points *
  Mat rectangles = wip;
  cvtColor( rectangles, rectangles, CV_GRAY2BGR );
  vector<Point> leftRectCenters; //filled by function findCurvePoints
  vector<Point> rightRectCenters;
  vector<Point> leftBarycenters;
  vector<Point> rightBarycenters;
  findCurvePoints(some_left, leftRectCenters, leftBarycenters, 0, wip, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkLeftRectCenters);
  findCurvePoints(some_right, rightRectCenters, rightBarycenters, 1, wip, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkRightRectCenters);

  //* Fit curves *
  //* Least squares 2nd order polynomial fitting    x = beta_2*y^2 + beta_1*y + beta_0 *
  vector<float> leftBeta = polyFit(leftBarycenters,wip, 2);
  vector<Point> fittedRight;
  vector<Point> fittedLeft;
  if(leftBeta.size() > 0){
    fittedLeft = computePoly(leftBeta, height);
  }
  vector<float> rightBeta = polyFit(rightBarycenters,wip, 2);
  if(rightBeta.size() > 0){
    fittedRight = computePoly(rightBeta, height);
  }

  //* Draw curves *
  polylines( rectangles, lastOkFittedRight, 0, last_ok_fitted_color, 8, 0);
  polylines( rectangles, lastOkFittedLeft, 0, last_ok_fitted_color, 8, 0);
  polylines( rectangles, fittedLeft, 0, cur_fitted_color, 8, 0);
  polylines( rectangles, fittedRight, 0, cur_fitted_color, 8, 0);

  //* Classify Curves *
  bool right_ok = classifyCurve(fittedRight, some_right, right_similar_series, right_bad_series, right_ok_series, lastFittedRight, lastOkFittedRight, lastOkRightRectCenters, rightRectCenters, rightBeta, lastOkBetaRight);
  bool left_ok = classifyCurve(fittedLeft, some_left, left_similar_series, left_bad_series, left_ok_series, lastFittedLeft, lastOkFittedLeft, lastOkLeftRectCenters, leftRectCenters, leftBeta, lastOkBetaLeft);

  //* Find average curve *
  vector<float> avgBeta = vector<float>();
  vector<Point> avgCurve;
  if(some_right && some_left){
    for(int i=0; i<lastOkBetaLeft.size(); i++){
      avgBeta.push_back((lastOkBetaLeft[i]+lastOkBetaRight[i])/2);
    }
    avgCurve = computePoly(avgBeta, height);
  }
  polylines( rectangles, avgCurve, 0, avg_curve_avg, 8, 0);


  //*** Inverse perspective transform ***
  /*
  if(perspTransfInPoints.size() > 0){
    rectangles = perspectiveTransform(rectangles,perspTransfOutPoints,perspTransfInPoints);
  }*/

  //Find direction
  float dir = 0;
  float u = 0;
  float p = 0.9;
  for(int i=0; i<avgCurve.size(); i++){
    //dir+=avgCurve[i].x;
    u = p*u + (1-p);
    dir+=u*avgCurve[i].x;
  }
  dir/=avgCurve.size();
  circle( rectangles, Point(dir,height), 5, Scalar( 0, 255, 0 ),  3, 3 );
  circle( rectangles, Point(width/2,height), 5, Scalar( 0, 100, 255 ),  3, 3 );


  int turn = computeDirection(dir, width/2);
  if(turn == 1){
    cout << "turn right" << endl;
  }else if(turn == -1){
    cout << "turn left" << endl;
  }else{
    cout << "go straight" << endl;
  }
  //* Display Images *
  displayImg("Rectangles",rectangles);
  //displayImg("Wip",wip);
  //displayImg("Src",src);

  //* Kill frame *
  waitKey(0);
  //if(waitKey(30) >= 0) break;

  //* Write to video *
  //outputVideo << src;
  counter++;
}
return 0;
} //END MAIN

//* Functions *
void drawRect(vector<Point> rect_points, Scalar rect_color, int height, Mat rectangles){ //draw the rectangles
  const float thickness = height/rect_thickness_ratio;
  line( rectangles, rect_points[0], rect_points[1], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[1], rect_points[2], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[2], rect_points[3], rect_color, thickness, CV_AA);
  line( rectangles, rect_points[3], rect_points[0], rect_color, thickness, CV_AA);
}

vector<Point> computeRect(Point center, int rect_width,int rect_height){ //given the center of the rectangle compute the 4 vertex
  vector<Point> points;
  points.push_back(Point(center.x - rect_width/2, center.y + rect_height/2));
  points.push_back(Point(center.x - rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y + rect_height/2 ));
  return points;
}

void displayImg(const char* window_name,Mat mat){
  namedWindow( window_name, WINDOW_NORMAL );
  cvResizeWindow(window_name, window_width, window_height);
  imshow( window_name, mat );
}

Mat perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints){
  //Perspective Transform
  int width = mat.size().width;
  int height = mat.size().height;
  Point2f inPoints[4];
  /*
  inPoints[0] = Point2f( 0, height );
  inPoints[1] = Point2f( width/2-width/8, height/2+height/6);
  inPoints[2] = Point2f( width/2+width/8, height/2+height/6);
  inPoints[3] = Point2f( width, height);
  */
  inPoints[0] = perspTransfInPoints[0];
  inPoints[1] = perspTransfInPoints[1];
  inPoints[2] = perspTransfInPoints[2];
  inPoints[3] = perspTransfInPoints[3];

  Point2f outPoints[4];
  outPoints[0] = perspTransfOutPoints[0];
  outPoints[1] = perspTransfOutPoints[1];
  outPoints[2] = perspTransfOutPoints[2];
  outPoints[3] = perspTransfOutPoints[3];

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
  if(totWeight>tot_min_weight){//xWeight!=0 && yWeight!=0){ //if no line is detected no barycenter is added or even if it's just a random bunch of pixels
  bar.y /= totWeight;
  bar.x /= totWeight;
}else{
  bar.x = -1;
  bar.y = -1;
}
return bar;
}

vector<float> polyFit(vector<Point> points,Mat mat, int order){
  vector<float> beta = vector<float>();
  int width = mat.size().width;
  int height = mat.size().height;
  if(points.size() > order){
    Mat X = Mat::zeros( points.size(), 1 , CV_32F );
    Mat y = Mat::zeros( points.size(), order+1 , CV_32F );
    Mat betaMat; //= Mat::zeros( 3, 1 , CV_32F );
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

    //Least squares
    betaMat = y.inv(DECOMP_SVD)*X; //leftBeta = ((leftX.t()*leftX).inv()*leftX.t())*leftY;
    beta = vector<float>();
    for(int i=0; i < betaMat.size().height; i++){
      beta.push_back(betaMat.at<float>(i,0));
    }

  }
  return beta;
}

vector<Point> computePoly(vector<float> beta, int n_points){
  vector<Point> fittedPoints = vector<Point>();
  for(int i = 0; i<n_points; i++){
    float fittedX = 0;
    for(int j = 0; j < beta.size(); j++){
      fittedX += beta[j]*pow(i,j);
    }
    Point fp = Point(fittedX,i);
    //circle( rectangles, fp, 5, Scalar( 0, 255, 0 ),  3, 3 );
    fittedPoints.push_back(fp);
  }
  return fittedPoints;
}

int findHistAcc(Mat mat, int pos){
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
  int leftMaxPos = -1;
  for(int i = 0; i < width/2; i++){
    if(histogram[i] > leftMax){
      leftMax = histogram[i];
      leftMaxPos = i;
    }
  }
  int rightMax = 0;
  int rightMaxPos = -1;
  //Second half
  for(int i = width/2; i < width; i++){
    if(histogram[i] > rightMax){
      rightMax = histogram[i];
      rightMaxPos = i;
    }
  }
  if(pos == 0){
    return leftMaxPos;
  }else{
    return rightMaxPos;
  }
}

Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset){
  int height = mat.size().height;
  int width = mat.size().width;
  Mat mask = Mat::zeros(height,width, CV_8UC1);
  polylines( mask, curve1, 0, 255, offset, 0);
  polylines( mask, curve2, 0, 255, offset, 0);
  return mask;
}

float computeRmse(vector<Point> curve1, vector<Point> curve2){
  float rmse = rmse_tolerance+1;
  if( curve1.size() > 0 && curve2.size() > 0){
    //RMSE
    rmse = 0;
    for(int i=0; i<curve1.size();i++){
      rmse+=pow(curve1[i].x-curve2[i].x,2)/curve1.size();
    }
    rmse = sqrt(rmse);
  }
  return rmse;
}

int dirChanges(vector<Point> points, int tolerance){
  int changes = 0;
  int direction; // -1 = left; 0 = straight; 1 = right
  if(points.size() > 1){
    changes = 0;
    for(int i = 0; i < points.size()-1; i++){
      int curCenterX = points[i].x;
      int nextCenterX = points[i+1].x;
      if(abs(curCenterX - nextCenterX) < tolerance){ //going straight
        if(direction != 0){
          direction = 0;
          changes++;
        }
      }else if(curCenterX - nextCenterX > tolerance){ //going left
        if(direction != -1){
          direction = -1;
          changes++;
        }
      }else{  //going right
        if(direction != 1){
          direction = 1;
          changes++;
        }
      }
    }
  }
  return changes;
}


bool classifyCurve(vector<Point> &fittedCurve, bool &some_curve, int &curve_similar_series, int &curve_bad_series, int &curve_ok_series, vector<Point> &lastFittedCurve, vector<Point> &lastOkFittedCurve, vector<Point> &lastOkCurveRectCenters, vector<Point> &curveRectCenters, vector<float> beta, vector<float> &lastOkBeta){
  //Classify
  bool curve_ok = false; //0 bad  1 good
  vector<Point> lower_rects = curveRectCenters;
  vector<Point> upper_rects = curveRectCenters;

  lower_rects.erase(lower_rects.begin());
  upper_rects.erase(upper_rects.begin()+curveRectCenters.size()-1);

  float adj_rmse = computeRmse(lower_rects,upper_rects);

  //Absolute classification
  if(adj_rmse > adj_rmse_threshold){ //trash curve if its barycenters are too far away one from each other with respect to the x
    curve_ok = false;
  }
  //Relative classification
  else{
    if(some_curve){ //If there's a good curve
    if(computeRmse(fittedCurve,lastOkFittedCurve) < rmse_tolerance){//If there's a good curve and current curve is similar to the good one
    curve_ok = true;
  }else{ //If there's a good curve and current curve is different from the good one
  if(computeRmse(fittedCurve,lastFittedCurve) < rmse_tolerance){ //If there's good curve, the current curve is different from the good one but similar to the previous
  curve_similar_series++;
  if(curve_similar_series >= min_similar_curves){
    curve_ok = true;
  }else{
    curve_ok = false;
  }
}else{ //If there's a good curve, the current curve is different from the good and the last curve
curve_similar_series = 0;
curve_ok = false;
}
}
}
else{ //If there's not a good curve
cout << "computeRmse(fittedCurve,lastFittedCurve): " << computeRmse(fittedCurve,lastFittedCurve) << endl;
if(computeRmse(fittedCurve,lastFittedCurve) < rmse_tolerance){ // If there's no good curve and the current is similar to the previous
cout << "no good curve, last 2 similar" << endl;
curve_similar_series++;
if(curve_similar_series >= min_similar_curves){
  curve_ok = true;
}else{
  curve_ok = false;
}
}
else{ //If there's no good curve and the current is different from the previous
curve_similar_series = 0;
curve_ok = false;
}
}
}


//Update states
if(curve_ok == false){ //Current curve is bad
  curve_ok_series = 0;
  curve_bad_series++;
}else if(curve_ok == true){ //Current curve is good
  curve_ok_series++;
  curve_bad_series = 0;
}


if(curve_ok_series >= min_good_curves){
  cout << "curve_ok_series: " << curve_ok_series << endl;
  some_curve = true;
  lastOkFittedCurve = fittedCurve;
  lastOkCurveRectCenters = curveRectCenters;
  lastOkBeta = beta;
}
if(curve_bad_series > max_bad_curves){
  some_curve = false;
  lastOkFittedCurve = vector<Point>();
  lastOkCurveRectCenters = vector<Point>();
  lastOkBeta = vector<float>();
}

lastFittedCurve = fittedCurve;
return curve_ok;
}

int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, vector<Point> & barycenters, int pos, Mat wip, int width, int height, int rect_offset, int rect_height, int rect_width, Mat rectangles, vector<Point> &lastOkRectCenters){ //pos: 0=left, 1=right
  if(some_curve == false){
    rectCenters = vector<Point>();
    //First rectangle
    int firstX = findHistAcc(wip, pos); //0 means left
    if(firstX == -1){  //in caso non trovi il massimo
      firstX = width/4;
    }

    rectCenters.push_back(Point(firstX, height - rect_offset - rect_height/2));
    //Other rectangles
    for(int i=0;i<n_rect;i++){
      //Compute left rectangle
      vector<Point> rect = computeRect(rectCenters[i], rect_width, rect_height);
      //Compute barycenters and rectangle centers
      Point nextCenter = Point();
      Point bar = computeBarycenter(rect ,wip);
      if(bar.x!=-1 && bar.y!=-1 ){ //if no line is detected no barycenter is added  && abs(bar.x - rectCenters[i].x)< next_bary_max_distance
        //move rectangle
        rect = computeRect(Point(bar.x, rectCenters[i].y), rect_width, rect_height);
        rectCenters[i].x = bar.x;

        barycenters.push_back(bar);
        circle( rectangles, bar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        nextCenter.x = bar.x;
      }
      else{
        nextCenter.x = rectCenters[i].x;
      }
      nextCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;


      if(i<n_rect-1){ // if we are in the last rectangle, we don't push the next rectangle
      rectCenters.push_back(nextCenter);
    }
    //Draw left rectangle
    drawRect(rect, rect_color, height, rectangles);
  }
}
else {

  rectCenters = lastOkRectCenters;
  for(int i=0;i<n_rect;i++){
    //Compute left rectangle
    vector<Point> rect = computeRect(rectCenters[i], rect_width, rect_height);

    Point bar = computeBarycenter(rect ,wip);
    if(bar.x!=-1 && bar.y!=-1 ){ //if no line is detected no barycenter is added
      //and if the barycenters are way too far from each other   && abs(bar.x - rectCenters[i].x)< next_bary_max_distance
      //move rectangle
      rect = computeRect(Point(bar.x, rectCenters[i].y), rect_width, rect_height);

      rectCenters[i].x = bar.x; //comment for fixed rectangles
      circle( rectangles, bar, 5, Scalar( 0, 0, 255 ),  3, 3 );
      barycenters.push_back(bar);
      /*if(i<n_rect-1){ // update for next rectangle as well
      rectCenters[i+1].x = rectCenters[i].x;
    }*/
  }
  /*if(i<n_rect-1){ // update for next rectangle as well
  rectCenters[i+1].x = rectCenters[i].x;
}*/
//Draw left rectangle
drawRect(rect, rect_color, height, rectangles);
}
}
return 0;
}


vector<Point2f> findPerspectiveInPoints(Mat src){
  Mat vanishingPointMap = src.clone();
  int height = src.size().height;
  int width = src.size().width;
  const int horizon_offset = height/horizon_offset_ratio;
  cout << "horizon offset " << horizon_offset << endl;
  vector<Point2f> perspTransfInPoints;

  cvtColor( vanishingPointMap, vanishingPointMap, CV_BGR2GRAY );

  for ( int i = 1; i < blur_kernel ; i = i + 2 ){
    GaussianBlur( vanishingPointMap, vanishingPointMap, Size( i, i ), 0, 0 );
  }

  Canny( vanishingPointMap, vanishingPointMap, canny_low_threshold, canny_low_threshold*canny_high_threshold_ratio, canny_kernel );


  //create mask
  //create mask
  Point mask_points[1][4];
  /*mask_points[0][0] = Point( 0, height - height/10);
  mask_points[0][1] = Point( width, height - height/10);
  mask_points[0][2] = Point( width, height/2);
  mask_points[0][3] = Point( 0, height/2);*/
  mask_points[0][0] = Point( 0, height - height/10);
  mask_points[0][1] = Point( width, height - height/10);
  mask_points[0][2] = Point( width, height/10);
  mask_points[0][3] = Point( 0, height/10);
  const Point* ppt[1] = { mask_points[0] };
  int npt[] = { 4 };
  Mat mask = Mat::zeros(height,width, CV_8UC1);
  fillPoly( mask, ppt, npt, 1, 255 ,8);
  //apply mask to image
  bitwise_and(vanishingPointMap, mask, vanishingPointMap);


  //Hugh Transform for Line detection, needed for looging for the vanishing point
  vector<Vec4i> hough_lines;
  vector<Vec4i> hough_longest_lines;
  vector<Vec4i> horizontal_lines = vector<Vec4i>();
  HoughLinesP(vanishingPointMap, hough_lines, 1, CV_PI/180, 120, 50, 100 ); //100 50 200
  vanishingPointMap = Mat::zeros(height,width, src.type());
  //keep only the longest lines
  float longestLen;
  if(hough_lines.size() > n_long_lines){  //if there are more lines than the number of lines that we want
    hough_longest_lines = vector<Vec4i>();
    for(int j = 0; j < n_long_lines; j++){
      longestLen = 0.0;
      Vec4i longestLine = Vec4i();
      int longest_index;
      for( int i = 0; i < hough_lines.size(); i++ ){
        Vec4i l = hough_lines[i];
        int x1 = l[0];
        int y1 = l[1];
        int x2 = l[2];
        int y2 = l[3];
        float len = sqrt(pow(y2-y1,2)+pow(x2-x1,2));
        float slope = (float)(y2-y1)/(x2-x1);
        //if(len > longestLen && abs(slope) < 10 && abs(slope) > 0.1){
        if(len > longestLen && abs(slope) > min_slope && abs(slope) < max_slope ){
          longestLine = l;
          longestLen = len;
          longest_index = i;
        }else if(abs(slope)==0){ //save horizon line for computing trapezium later
          horizontal_lines.push_back(l);
        }
      }
      //if(longestLine[0]!=0 && longestLine[1]!=0 && longestLine[2]!=0 && longestLine[3]!=0){
      hough_longest_lines.push_back(longestLine);
      hough_lines.erase(hough_lines.begin() + longest_index);

    }
  }else{
    hough_longest_lines = vector<Vec4i>();
    for(int i=0; i<hough_lines.size();i++){
      Vec4i l = hough_lines[i];
      int x1 = l[0];
      int y1 = l[1];
      int x2 = l[2];
      int y2 = l[3];
      float len = sqrt(pow(y2-y1,2)+pow(x2-x1,2));
      float slope = (float)(y2-y1)/(x2-x1);
      if(abs(slope) < max_slope && abs(slope) > min_slope){
        hough_longest_lines.push_back(l);

      }
    }
  }

  //*** Show Hough ***
  for(int i = 0; i<hough_longest_lines.size(); i++){
    cout << "hough_longest_lines: " << hough_longest_lines[i] << endl;
  }
  cout << hough_lines.size() << endl;
  cout << hough_longest_lines.size() << endl;
  Mat houghmap = Mat::zeros(height,width, src.type());
  /*for(int i = 0; i < hough_longest_lines.size(); i++){
      line( houghmap, Point(hough_longest_lines[i][0], hough_longest_lines[i][1]), Point(hough_longest_lines[i][2], hough_longest_lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  }*/
  for(int i = 0; i < hough_lines.size(); i++){
      line( houghmap, Point(hough_lines[i][0], hough_lines[i][1]), Point(hough_lines[i][2], hough_lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  }
  //displayImg("hough",houghmap);



  //* Compute all line equations *
  vector<Vec2f> m_and_q = vector<Vec2f>();
  for(int i = 0; i < hough_longest_lines.size() ; i++){
    Vec2f mq = Vec2f();
    Vec4i l = hough_longest_lines[i];
    int x1 = l[0];
    int y1 = l[1];
    int x2 = l[2];
    int y2 = l[3];
    float m = (float)(y2-y1)/(x2-x1);
    float q = y1-m*x1;
    mq[0] = m; mq[1] = q;
    m_and_q.push_back(mq);
  }
  //draw lines
  for(int i = 0; i < m_and_q.size(); i++){
    Vec2f r = m_and_q[i];
    float m = r[0];
    float q = r[1];
    //retrieve 2 point of the rect given m and q
    int x0 = 0;
    int x_width = width;
    float y0 = m * x0 + q;
    float y_width = m * x_width + q;
    line( vanishingPointMap, Point(x0, y0), Point(x_width, y_width), Scalar(0,0,255), 3, CV_AA); //red lines
  }
  /*
  line( vanishingPointMap, Point( width, height/2), Point( 0, height/2), Scalar(255,0,255), 3, CV_AA); //horizontal line
  line( vanishingPointMap, Point( width, height*9/10), Point( 0, height*9/10), Scalar(255,0,255), 3, CV_AA); //horizontal line
  */
  //Overlap images
  bitwise_or(src, vanishingPointMap, vanishingPointMap);


  //* Compute all intersection points *
  vector<Point> intersectionPoints = vector<Point>();
  for(int i = 0; i < m_and_q.size(); i++){
    Vec2f mq1 = m_and_q[i];
    for(int k = i; k < m_and_q.size(); k++){
      //intersection points
      Vec2f mq2 = m_and_q[k];
      int x_int = (mq1[1] - mq2[1])/(mq2[0] - mq1[0]);
      int y_int = mq1[0]*x_int + mq1[1];
      Point intersection_point = Point(x_int, y_int);
      if(x_int > 0 && x_int < width && y_int > height/3 && y_int < 2*height/3){ //y_int > 0 && y_int < height
        intersectionPoints.push_back(intersection_point);
        //draw intersection
        circle( vanishingPointMap, intersection_point, 5, Scalar( 255, 255, 255),  2, 2 ); //white dots
      }
    }
  }

  /*
  //*** Cluster points and get the biggest one ***
  vector<Point> cluster_centroids;
  //find k nearest points to a point
  for(int i = 0; i < height; i++){  //i row
    for(int j = 0; j < width; j++){ //j column

    }
  }

  //find nearest point to a point
  for(int )
*/

  //* Find vanishing point as the average of the intersection points *
  int x_sum = 0;
  int y_sum = 0;
  for(int i = 0; i < intersectionPoints.size(); i++){
    x_sum += intersectionPoints[i].x;
    y_sum += intersectionPoints[i].y;
  }
  if(intersectionPoints.size() > 0){
    float x_van_point = x_sum/intersectionPoints.size(); //media
    float y_van_point = y_sum/intersectionPoints.size(); //media
    //float x_van_point = intersectionPoints[intersectionPoints.size()/2].x; //mediana
    //float y_van_point = intersectionPoints[intersectionPoints.size()/2].y; //mediana
    Point vanishing_point = Point(x_van_point, y_van_point);
    circle( vanishingPointMap, vanishing_point, 5, Scalar( 0, 255, 0),  4, 4 ); //green dot

    //* Build 2 lines from the vanishing point to the bottom corners *
    float m_left = (float)(height - height/6 - vanishing_point.y)/(0 - vanishing_point.x);
    float q_left = vanishing_point.y-m_left*vanishing_point.x;
    float m_right = (float)(height - height/6 - vanishing_point.y)/(width - vanishing_point.x);
    float q_right = vanishing_point.y-m_right*vanishing_point.x;
    //draw
    for(int i = 0; i<2; i++){
      float m,q;
      if(i==0){
        m = m_right;
        q = q_right;
      }else{
        m = m_left;
        q = q_left;
      }
      int x0 = 0;
      int x_width = width;
      float y0 = m * x0 + q;
      float y_width = m * x_width + q;
      line( vanishingPointMap, Point(x0, y0), Point(x_width, y_width), Scalar(255,0,0), 3, CV_AA); //blue lines
    }

    //* Find trapezium points *
    //Find horizon line
    int horizon = 0;
    for(int i=0; i<horizontal_lines.size(); i++){
      Vec4i l = horizontal_lines[i];
      int v = vanishing_point.y;
      if(l[1]>v && (abs(l[1]-v) < height/6) && l[1]>horizon){
        horizon = l[1];
      }
    }
    cout << "horizon ************* " << horizon << endl;
    cout << "van ************* " << vanishing_point << endl;
    if(horizon < vanishing_point.y){
      horizon = vanishing_point.y;
    }
    //horizontal lines
    int xUp1 = 0;
    int yUp1 = horizon + 50;//horizon_offset; //height - height/3;
    int xUp2 = width;
    int yUp2 = yUp1;
    int xDown1 = 0;
    int yDown1 = height; //- height/6;  //height*9/10;
    int xDown2 = width;
    int yDown2 = yDown1;
    float m_up = (float)(yUp2-yUp1)/(xUp2-xUp1);
    float m_down = (float)(yDown2-yDown1)/(xDown2-xDown1);
    float q_up = yUp1-m_up*xUp1;
    float q_down = yDown1-m_up*xDown1;
    //left intersection points
    int xIntLeft1 = (q_down - q_left)/(m_left - m_down);
    int yIntLeft1 = m_left*xIntLeft1 + q_left;
    int xIntLeft2 = (q_up - q_left)/(m_left - m_up);
    int yIntLeft2 = m_left*xIntLeft2 + q_left;
    //right intersection points
    int xIntRight1 = (q_up - q_right)/(m_right - m_up);
    int yIntRight1 = m_right*xIntRight1 + q_right;
    int xIntRight2 = (q_down - q_right)/(m_right - m_down);
    int yIntRight2 = m_right*xIntRight2 + q_right;
    circle( vanishingPointMap, Point(xIntRight1, yIntRight1), 5, Scalar( 0, 255, 255),  4, 2 ); //yellow dots
    circle( vanishingPointMap, Point(xIntRight2, yIntRight2), 5, Scalar( 0, 255, 255),  4, 2 );
    circle( vanishingPointMap, Point(xIntLeft1, yIntLeft1), 5, Scalar( 0, 255, 255),  4, 2 );
    circle( vanishingPointMap, Point(xIntLeft2, yIntLeft2), 5, Scalar( 0, 255, 255),  4, 2 );

    //* Return perspective transform points *
    perspTransfInPoints = vector<Point2f>();
    perspTransfInPoints.push_back(Point(xIntLeft1, yIntLeft1));
    perspTransfInPoints.push_back(Point(xIntLeft2, yIntLeft2));
    perspTransfInPoints.push_back(Point(xIntRight1, yIntRight1));
    perspTransfInPoints.push_back(Point(xIntRight2, yIntRight2));

  }

  //displayImg("vanishingPointMap",vanishingPointMap);
  return perspTransfInPoints;

}

int computeDirection(float actualPos, float desiredPos){ // 1 turn right, 0 don't turn, -1 turn left
if(desiredPos + straight_range - actualPos <  0){
  return -1;
}else if(desiredPos - straight_range - actualPos > 0){
  return 1;
}
return 0;
}
