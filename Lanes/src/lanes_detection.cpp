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
#define max_bad_curves 2
#define min_good_curves 1
#define min_barycenters 5 //in realtà andrebbe messo come ratio e diviso per n_rect
#define next_bary_max_distance 50 //anche qui va messo ratio
#define rmse_tolerance 20
#define min_similar_curves 3

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
  if(some_right && some_left){
    wip = curve_mask(mask_curve_right,mask_curve_left,wip,mask_offset);
  }

  Mat rectangles = wip;
  cvtColor( rectangles, rectangles, CV_GRAY2BGR );

  rightBarycenters = vector<Point>();
  leftBarycenters = vector<Point>();
  vector<Point> leftRectCenters;
  vector<Point> rightRectCenters;
  //Initialize rectangles
  if(some_left == false){//Se non ho left
    mask_curve_left = vector<Point>();
    leftRectCenters = vector<Point>();
    //First rectangle
    int leftFirstX = findHistAcc(wip,0); //0 indica sinistra
    if(leftFirstX == -1){  //in caso non trovi il massimo
      leftFirstX = width/4;
    }

      leftRectCenters.push_back(Point(leftFirstX, height - rect_offset - rect_height/2));
      //Other rectangles
      for(int i=0;i<n_rect;i++){
        //Compute left rectangle
        vector<Point> left_rect = computeRect(leftRectCenters[i], rect_width, rect_height);
        //Compute barycenters and rectangle centers
        Point nextLeftCenter = Point();
        Point leftBar = computeBarycenter(left_rect ,wip);
        if(leftBar.x!=-1 && leftBar.y!=-1 ){ //if no line is detected no barycenter is added  && abs(leftBar.x - leftRectCenters[i].x)< next_bary_max_distance
          //move rectangle
          left_rect = computeRect(Point(leftBar.x, leftRectCenters[i].y), rect_width, rect_height);
          leftRectCenters[i].x = leftBar.x;

          leftBarycenters.push_back(leftBar);
          circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
          nextLeftCenter.x = leftBar.x;
        }
        else{
          nextLeftCenter.x = leftRectCenters[i].x;
        }
        nextLeftCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;


        if(i<n_rect-1){ // if we are in the last rectangle, we don't push the next rectangle
          leftRectCenters.push_back(nextLeftCenter);
        }

        //Draw left rectangle
        drawRect(left_rect, rect_color, rect_thickness, rectangles);
      }

  }
  else { //Se ho left

    leftRectCenters = lastOkLeftRectCenters;
    for(int i=0;i<n_rect;i++){
      //Compute left rectangle
      vector<Point> left_rect = computeRect(leftRectCenters[i], rect_width, rect_height);

      Point leftBar = computeBarycenter(left_rect ,wip);
      if(leftBar.x!=-1 && leftBar.y!=-1 ){ //if no line is detected no barycenter is added
                                          //and if the barycenters are way too far from each other   && abs(leftBar.x - leftRectCenters[i].x)< next_bary_max_distance
        //move rectangle
        left_rect = computeRect(Point(leftBar.x, leftRectCenters[i].y), rect_width, rect_height);

        leftRectCenters[i].x = leftBar.x; //comment for fixed rectangles
        circle( rectangles, leftBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        leftBarycenters.push_back(leftBar);
        /*if(i<n_rect-1){ // update for next rectangle as well
          leftRectCenters[i+1].x = leftRectCenters[i].x;
        }*/
      }
      /*if(i<n_rect-1){ // update for next rectangle as well
        leftRectCenters[i+1].x = leftRectCenters[i].x;
      }*/
      //Draw left rectangle
      drawRect(left_rect, rect_color, rect_thickness, rectangles);
  }
}
  if(some_right == false){//Se non ho right
    mask_curve_right = vector<Point>();
    rightRectCenters = vector<Point>();
    //First rectangle
    int rightFirstX = findHistAcc(wip,1); // 1 indica destra
    if(rightFirstX == -1){
      rightFirstX = width*3/4;
    }
      rightRectCenters.push_back(Point(rightFirstX, height - rect_offset - rect_height/2));
      //Other rectangles
      for(int i=0;i<n_rect;i++){
        //Compute right rectangle
        vector<Point> right_rect = computeRect(rightRectCenters[i], rect_width, rect_height);
        //Compute barycenters and rectangle centers
        Point nextRightCenter = Point();
        Point rightBar = computeBarycenter(right_rect ,wip);
        if(rightBar.x!=-1 && rightBar.y!=-1 ){ //&& abs(rightBar.x - rightRectCenters[i].x)< next_bary_max_distance
          //move rectangle
          right_rect = computeRect(Point(rightBar.x, rightRectCenters[i].y), rect_width, rect_height);
          rightRectCenters[i].x = rightBar.x;

          rightBarycenters.push_back(rightBar);
          circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
          nextRightCenter.x = rightBar.x;
        }else{
          nextRightCenter.x = rightRectCenters[i].x;
        }
        nextRightCenter.y = height - rect_offset - rect_height/2 - (i+1)*rect_height;

        if(i<n_rect-1){ // if we are in the last rectangle, we don't push the next rectangle
          rightRectCenters.push_back(nextRightCenter);
        }
        //Draw right rectangle
        drawRect(right_rect, rect_color, rect_thickness, rectangles);
      }
  }
else {//Se ho right

   rightRectCenters = lastOkRightRectCenters;
    for(int i=0;i<n_rect;i++){
      //Compute right rectangle
      vector<Point> right_rect = computeRect(rightRectCenters[i], rect_width, rect_height);

      Point rightBar = computeBarycenter(right_rect ,wip);
      if(rightBar.x!=-1 && rightBar.y!=-1){ // && abs(rightBar.x - rightRectCenters[i].x)< next_bary_max_distance
        //move rectangle
        right_rect = computeRect(Point(rightBar.x, rightRectCenters[i].y), rect_width, rect_height);
        rightRectCenters[i].x = rightBar.x; //comment for fixed rectangles
        circle( rectangles, rightBar, 5, Scalar( 0, 0, 255 ),  3, 3 );
        rightBarycenters.push_back(rightBar);
        /*
        if(i<n_rect-1){ // uncomment for updating above only if found barycenter
          rightRectCenters[i+1].x = rightRectCenters[i].x;
        }*/
      }
      /*if(i<n_rect-1){ // update next rectangle
        rightRectCenters[i+1].x = rightRectCenters[i].x;
      }*/
      //Draw right rectangle
      drawRect(right_rect, rect_color, rect_thickness, rectangles);

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


  //Classification parameters
  //Compute changes in direction
  int leftChanges = dirChanges(leftBarycenters,straight_tolerance);
  int rightChanges = dirChanges(rightBarycenters,straight_tolerance);
  //Compute rmse between current curve and last one
  int leftRmse = computeRmse(fittedLeft,lastOkFittedLeft);
  //int rightRmse = computeRmse(fittedRight,lastOkFittedRight);


  //if there is NOT a good curve look for a minimum number of similar curve in a row
  //if there is a good curve compare the current curve with the good one
  //right
  cout << "*** frame" << endl;
  if(!some_right){//if there is not a good curve
    cout << "no good line" << endl;
    if(lastFittedRight.size() > 0 && fittedRight.size() > 0){ // check if there is a curve in the last frame
      cout << "there are curves in the last frame and in the current" << endl;
      int rmse_last_frames = computeRmse(fittedRight, lastFittedRight); //difference between last frame and current frame curves
      cout << "rmse_last_frames: " << rmse_last_frames << endl;
      if(rmse_last_frames < rmse_tolerance){
        right_similar_series++;
        cout << "right_similar_series: " << right_similar_series << endl;
      }else{
        right_similar_series = 0;
      }
      lastFittedRight = fittedRight;

      if(right_similar_series >= min_similar_curves){ //check how many similar curves in a row
        cout << "there are a min number of similar curves in a row" << endl;
        lastOkFittedRight = fittedRight;
        lastOkRightRectCenters = rightRectCenters;
        some_right = true;
        right_similar_series = 0;
      }
    }else{//if there is NOT curve in the last frame
      if(fittedRight.size() > 0){
        lastFittedRight = fittedRight;
      }
    }
  }else{ //if there is a good curve
    cout << "there is a good curve" << endl;
    int right_rmse = computeRmse(fittedRight, lastOkFittedRight); //compare current curve with the good one
    cout << "right_rmse: " << right_rmse << endl;
    if(right_rmse < rmse_tolerance){
      cout << "current curve is similar to good curve" << endl;
      lastOkFittedRight = fittedRight;
      lastOkRightRectCenters = rightRectCenters;
      right_bad_series = 0;
      lastFittedRight = lastOkFittedRight;
    }else{
      right_bad_series++;
      cout << "current curve is not similar to good curve - right_bad_series: " << right_bad_series << endl;
      right_rmse = computeRmse(fittedRight, lastFittedRight); //compare current curve with last frame curve
      if(right_rmse < rmse_tolerance){
        right_similar_series++;
        cout << "current curve is similar to last curve - right_similar_series: " << right_similar_series << endl;
      }else{
        right_similar_series = 0;
        cout << "current curve is not even similar to last curve" << endl;
      }
      lastFittedRight = fittedRight;
      if(right_similar_series >= min_similar_curves){ //check how many similar curves in a row
        cout << "there are a min number of similar curves in a row" << endl;
        lastOkFittedRight = fittedRight;
        lastOkRightRectCenters = rightRectCenters;
        right_similar_series = 0;
      }
    }
  }
  //left


  //Reset reference states
  if(left_bad_series > max_bad_curves){
    some_left = false;
  }
  if(right_bad_series > max_bad_curves){
    some_right = false;
  }



  //Classify sound and bad curves
  /*
  if(leftChanges > max_dir_changes || leftBarycenters.size() < min_barycenters){ //Se ho troppi cambi di direzione o ho troppi pochi punti è bad
    left_ok = false;
  }else{
    if(some_left){
      if(leftRmse > 20){ // Se ho pochi cambi di direzione ma ho rmse alto allora è bad
        left_ok = false;
      }else{  //Se ho pochi cambi e rmse basso allora ok
        left_ok = true;
      }
    }else{ //Se ho pochi cambi e nessuna curva di riferimento allora ok
      left_ok = true;
    }
  }
  if(rightChanges > max_dir_changes || rightBarycenters.size() < min_barycenters){ //Se ho troppi cambi di direzione o ho troppi pochi punti è bad
    right_ok = false;
  }else{
    if(some_right){
      if(rightRmse > 20){ // Se ho pochi cambi di direzione ma ho rmse alto allora è bad
        right_ok = false;
      }else{  //Se ho pochi cambi e rmse basso allora ok
        right_ok = true;
      }
    }else{ //Se ho pochi cambi e nessuna curva di riferimento allora ok
      right_ok = true;
    }
  }


  //Update trace curves
  if(left_ok){
    left_ok_series++;
    left_bad_series = 0;
    lastOkFittedLeft = fittedLeft;
    lastOkLeftRectCenters = leftRectCenters;
  }else{
    left_ok_series = 0;
    left_bad_series++;
  }
  if(right_ok){
    right_ok_series++;
    right_bad_series = 0;
    lastOkFittedRight = fittedRight;
    lastOkRightRectCenters = rightRectCenters;
  }else{
    right_ok_series = 0;
    right_bad_series++;
  }

  //Update reference states and curves
  if(right_ok_series > min_good_curves){
    if(!some_right){
      some_right = true;
      mask_curve_right = fittedRight;
    }
  }
  if(left_ok_series > min_good_curves){
    if(!some_left){
      some_left = true;
      mask_curve_left = fittedLeft;
    }
  }


  //Reset reference states
  if(left_bad_series > max_bad_curves){
    some_left = false;
  }
  if(right_bad_series > max_bad_curves){
    some_right = false;
  }
  */




  //rectangles = reversePerspectiveTransform(rectangles);

  //Display Image
  //displayImg("Wip",wip);
  displayImg("Rectangles",rectangles);

  waitKey(0);
  //if(waitKey(30) >= 0) break;
  //outputVideo << src;
}
return 0;
}
