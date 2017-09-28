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
#define tot_min_weight 10
#define max_dir_changes 5
#define straight_tolerance_ratio 80
#define max_rmse_ratio 70
#define max_bad_curves 5
#define min_good_curves 3
#define min_barycenters 5 //in realtà andrebbe messo come ratio e diviso per n_rect
#define next_bary_max_distance 50 //anche qui va messo ratio
#define rmse_tolerance 20
#define min_similar_curves 3
#define adj_rmse_threshold 30

const Scalar rect_color = Scalar(0,0,255);

//PROTOTYPES
vector<Point> computeRect(Point center, int rect_width,int rect_height);
void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles);
void displayImg(const char* window_name,Mat mat);
Mat perspectiveTransform(Mat mat);
Mat reversePerspectiveTransform(Mat mat);
float movingAverage(float avg, float new_sample);
Point computeBarycenter(vector<Point> points, Mat mat);
vector<Point> polyFit(vector<Point> points,Mat mat);
int findHistAcc(Mat mat, int pos);
Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset);
float computeRmse(vector<Point> curve1, vector<Point> curve2);
int dirChanges(vector<Point> points, int tolerance);
int classifyCurve(vector<Point> &fittedCurve, bool &some_curve, int &similar_series, int &curve_bad_series, int &curve_ok_series, vector<Point> &lastFittedCurve, vector<Point> &lastOkFittedCurve, vector<Point> &lastOkRectCenters, vector<Point> &rectCenters);
int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, int pos, Mat wip, int width, int height, int rect_offset, int rect_height, int rect_width, vector<Point> &barycenters, Mat rectangles, vector<Point> &lastOkRectCenters); //pos: 0=left, 1=right

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
vector<Point> rightBarycenters; //servono fuori per fare il fitting
vector<Point> leftBarycenters;
vector<Point> lastOkRightRectCenters;
vector<Point> lastOkLeftRectCenters;
//umbi
vector<Point> lastFittedRight;
vector<Point> lastFittedLeft;
//vector<Point> goodFittedRight;
//vector<Point> goodFittedLeft;
int right_similar_series = 0;
int left_similar_series = 0;

bool left_ok = false;  //serve per non fare la maschera al primo ciclo quando non ho ancora le linee
bool right_ok = false;
bool some_left = false;
bool some_right = false;
int left_bad_series = 0;
int right_bad_series = 0;
int right_ok_series = 0;
int left_ok_series = 0;

for(;;){
  Mat src, wip;
  //Capture frame
  cap >> src;
  int width = src.size().width;
  int height = src.size().height;
  const int rect_width = width/rect_width_ratio;
  const int rect_offset = height/rect_offset_ratio;
  const int rect_height = (height - rect_offset)/n_rect;
  const int straight_tolerance = width/straight_tolerance_ratio;
  const int max_rmse = height/max_rmse_ratio; //height perchè la parabola orizzontale è calcolata da x a y
  wip = src;

  cvtColor( wip, wip, CV_BGR2GRAY );

  for ( int i = 1; i < blur_kernel ; i = i + 2 ){
    GaussianBlur( wip, wip, Size( i, i ), 0, 0 );
  }

  //Canny( wip, wip, canny_low_threshold, canny_low_threshold*canny_high_threshold_ratio, canny_kernel );

  wip = perspectiveTransform(wip);

  //Color Filtering
  //White Filter
  inRange(wip, Scalar(150, 150, 150), Scalar(255, 255, 255), wip);
  //cvtColor( wip, wip, CV_BGR2GRAY );

  adaptiveThreshold(wip,wip,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,55,-20);
  //threshold(wip,wip,0,255,THRESH_BINARY | THRESH_OTSU);
  //threshold(wip,wip,THRESH_OTSU,255,THRESH_OTSU);


/*
  //Curve Mask
  if(some_right && some_left){
    wip = curve_mask(lastOkFittedRight,lastOkFittedLeft,wip,mask_offset);
  }*/


  Mat rectangles = wip;
  cvtColor( rectangles, rectangles, CV_GRAY2BGR );

  rightBarycenters = vector<Point>();
  leftBarycenters = vector<Point>();
  vector<Point> leftRectCenters;
  vector<Point> rightRectCenters;
  //Initialize rectangles
  int findLeftCurve =  findCurvePoints(some_left, leftRectCenters, 0, wip, width, height, rect_offset, rect_height, rect_width, leftBarycenters, rectangles, lastOkLeftRectCenters);
  int findRightCurve =  findCurvePoints(some_right, rightRectCenters, 1, wip, width, height, rect_offset, rect_height, rect_width, rightBarycenters, rectangles, lastOkRightRectCenters);
  //LEAST SQUARES SECOND ORDER POLYNOMIAL FITTING
  // x = beta_2*y^2 + beta_1*y + beta_0
  vector<Point> fittedLeft = polyFit(leftBarycenters,wip);
  vector<Point> fittedRight = polyFit(rightBarycenters,wip);



  polylines( rectangles, lastOkFittedRight, 0, Scalar(255,0,0) ,8,0);
  polylines( rectangles, lastOkFittedLeft, 0, Scalar(255,0,0) ,8,0);
  polylines( rectangles, fittedLeft, 0, Scalar(0,255,0) ,8,0);
  polylines( rectangles, fittedRight, 0, Scalar(0,255,0) ,8,0);


  //Classification parameters
  //Compute changes in direction
  //int leftChanges = dirChanges(leftBarycenters,straight_tolerance);
  //int rightChanges = dirChanges(rightBarycenters,straight_tolerance);
  //Compute rmse between current curve and last one
  //int leftRmse = computeRmse(fittedLeft,lastOkFittedLeft);
  //int rightRmse = computeRmse(fittedRight,lastOkFittedRight);


  //if there is NOT a good curve look for a minimum number of similar curve in a row
  //if there is a good curve compare the current curve with the good one
  //right
  cout << "***** frame" << endl;
  cout << "* Right" << endl;
  int classifyRight = classifyCurve(fittedRight, some_right, right_similar_series, right_bad_series, right_ok_series, lastFittedRight, lastOkFittedRight, lastOkRightRectCenters, rightRectCenters);
  //left
  cout << "* Left" << endl;
  int classifyLeft = classifyCurve(fittedLeft, some_left, left_similar_series, left_bad_series, left_ok_series, lastFittedLeft, lastOkFittedLeft, lastOkLeftRectCenters, leftRectCenters);

  cout << "some left: " << some_left << endl;
  cout << "left bad series: " << left_bad_series << endl;
  cout << "some right: " << some_right << endl;
  cout << "right bad series: " << right_bad_series << endl;

//rectangles = reversePerspectiveTransform(rectangles);

//Display Image<
//displayImg("Wip",wip);
displayImg("Rectangles",rectangles);
displayImg("Wip",wip);

waitKey(0);
//if(waitKey(30) >= 0) break;
//outputVideo << src;
}
return 0;
}

//FUNCTIONS
void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles){ //draw the rectangles
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

Mat reversePerspectiveTransform(Mat mat){
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
  lambda = getPerspectiveTransform( outPoints, inPoints );
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
  bitwise_and(mat,mask,mat);
  displayImg("Mask",mask);
  return mat;
}

float computeRmse(vector<Point> curve1, vector<Point> curve2){
  float rmse = -1;
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


int classifyCurve(vector<Point> &fittedCurve, bool &some_curve, int &curve_similar_series, int &curve_bad_series, int &curve_ok_series, vector<Point> &lastFittedCurve, vector<Point> &lastOkFittedCurve, vector<Point> &lastOkCurveRectCenters, vector<Point> &curveRectCenters){
  //Classify
  int curve_ok = 0; //0 bad  1 good
  vector<Point> lower_rects = curveRectCenters;
  vector<Point> upper_rects = curveRectCenters;

  lower_rects.erase(lower_rects.begin());
  upper_rects.erase(upper_rects.begin()+curveRectCenters.size()-1);

  float adj_rmse = computeRmse(lower_rects,upper_rects);

  //Absolute classification
  if(adj_rmse > adj_rmse_threshold){ //trash curve if its barycenters are too far away one from each other with respect to the x
    curve_ok = 0;
  }
  //Relative classification
  else{
    if(some_curve){ //If there's a good curve
      if(computeRmse(fittedCurve,lastOkFittedCurve) < rmse_tolerance){//If there's a good curve and current curve is similar to the good one
        curve_ok = 1;
      }else{ //If there's a good curve and current curve is different from the good one
        if(computeRmse(fittedCurve,lastFittedCurve) < rmse_tolerance){ //If there's good curve, the current curve is different from the good one but similar to the previous
            curve_similar_series++;
            if(curve_similar_series >= min_similar_curves){
              curve_ok = 1;
            }else{
              curve_ok = 0;
            }
          }
          else{ //If there's a good curve, the current curve is different from the good and the last curve
            curve_similar_series = 0;
            curve_ok = 0;
          }
        }
      }
    else{ //If there's not a good curve
      if(computeRmse(fittedCurve,lastFittedCurve) < rmse_tolerance){ // If there's no good curve and the current is similar to the previous
      curve_similar_series++;
        if(curve_similar_series >= min_similar_curves){
          curve_ok = 1;
        }else{
          curve_ok = 0;
        }
      }
      else{ //If there's no good curve and the current is different from the previous
        curve_similar_series = 0;
        curve_ok = 0;
      }
    }
  }


  //Update states
  if(curve_ok == 0){ //Current curve is bad
    curve_ok_series = 0;
    curve_bad_series++;
  }else if(curve_ok == 1){ //Current curve is good
    curve_ok_series++;
    curve_bad_series = 0;
  }


  if(curve_ok_series >= min_good_curves){
    some_curve = true;
    lastOkFittedCurve = fittedCurve;
    lastOkCurveRectCenters = curveRectCenters;
  }
  if(curve_bad_series > max_bad_curves){
    some_curve = false;
    lastOkFittedCurve = vector<Point>();
    lastOkCurveRectCenters = vector<Point>();
  }

  lastFittedCurve = fittedCurve;
  return curve_ok;
}

int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, int pos, Mat wip, int width, int height, int rect_offset, int rect_height, int rect_width, vector<Point> &barycenters, Mat rectangles, vector<Point> &lastOkRectCenters){ //pos: 0=left, 1=right
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
    drawRect(rect, rect_color, rect_thickness, rectangles);
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
drawRect(rect, rect_color, rect_thickness, rectangles);
}
}
return 0;
}
