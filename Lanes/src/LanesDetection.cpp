#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include "LanesDetection.h"
#include "tinysplinecpp.h"
#include "Camera_Params.h"

using namespace std;
using namespace cv;

//* Global variables *
const int canny_low_threshold = 50;
const int canny_high_threshold_ratio = 3;
const int canny_kernel = 3;
const int blur_kernel = 5;
const int mask_offset_ratio = 10;
const int rect_width_ratio = 9;
const int rect_offset_ratio = 200;
const int n_rect = 20;
const int rect_thickness_ratio = 200;
const int tot_min_weight = 10;
const int max_dir_changes = 5;
const int straight_tolerance_ratio = 80;
const int max_rmse_ratio = 70;
const int max_bad_curves = 3;
const int min_good_curves = 1;
const int min_barycenters = 2; //in realtà andrebbe messo come ratio e diviso per n_rect
const int next_bary_max_distance = 50; //anche qui va messo ratio
const int rmse_tolerance = 20;
const int min_similar_curves = 3;
const int adj_rmse_threshold = 30;
const int n_long_lines = 20; //number of lines for vanishing point
const float max_slope = 10;
const float min_slope = 0.1;
const int window_width = 800;
const int window_height = 500;
const int horizon_offset_ratio = 5;
const int straight_range = 3; //cambiare con ratio
const int vanishing_point_window = 10;
const int vanishing_point_window_offset = 1;
const int fit_order = 3;
const int n_barycenters_window = 3;
const int partial_fitting_order = 1;
const bool profile_param = false;
const bool display_param = true;
const int interpolartion_type = 0; //0: polynomial, 1: b-spline
const int camera_type = 0; //0:GoPro hero 4
//colors
const Scalar rect_color = Scalar(0,0,255);
const Scalar last_ok_fitted_color = Scalar(255,0,0);
const Scalar avg_curve_avg = Scalar(0,255,255);
const Scalar cur_fitted_color = Scalar(0,255,0);
const Scalar white_filtering_threshold = Scalar(120, 120, 120);



//Function definition
//Constructor
LanesDetection::LanesDetection(){
    this->cannyLowThreshold = canny_low_threshold;
    this->cannyHighThresholdRatio = canny_high_threshold_ratio;
    this->cannyKernel = canny_kernel;
    this->blurKernel = blur_kernel;
    this->maskOffsetRatio = mask_offset_ratio;
    this->rectWidthRatio = rect_width_ratio;
    this->rectOffsetRatio = rect_offset_ratio;
    this->nRect = n_rect;
    this->rectThicknessRatio = rect_thickness_ratio;
    this->totMinWeight = tot_min_weight;
    this->maxDirChanges = max_dir_changes;
    this->straightToleranceRatio = straight_tolerance_ratio;
    this->maxRmseRatio = max_rmse_ratio;
    this->maxBadCurves = max_bad_curves;
    this->minGoodCurves = min_good_curves;
    this->minBarycenters  = min_barycenters;
    this->nextBaryMaxDistance = next_bary_max_distance;
    this->rmseTolerance = rmse_tolerance;
    this->minSimilarCurves = min_similar_curves;
    this->adjRmseThreshold = adj_rmse_threshold;
    this->nLongLines = n_long_lines;
    this->maxSlope = max_slope;
    this->minSlope = min_slope;
    this->windowWidth = window_width;
    this->windowHeight = window_height;
    this->horizonOffsetRatio = horizon_offset_ratio;
    this->straightRange = straight_range;
    this->vanishingPointWindow = vanishing_point_window;
    this->vanishingPointWindowOffset = vanishing_point_window_offset;
    this->order = fit_order;
    this->nBarycentersWindow = n_barycenters_window;
    this->partialFittingOrder = partial_fitting_order;
    this->profile = profile_param;
    this->display = display_param;
    this->interpolationType = interpolartion_type;
    this->camera = Camera_Params(camera_type);
    //colors
    this->rectColor = rect_color;
    this->lastOkFittedColor = last_ok_fitted_color;
    this->avgCurveAvg = avg_curve_avg;
    this->curFittedColor = cur_fitted_color;
    this->whiteFilteringThreshold = white_filtering_threshold;
    //dynamic attributes
    this->someLeft = false;
    this->someRight = false;
    this->leftBadSeries = 0;
    this->rightBadSeries = 0;
    this->rightOkSeries = 0;
    this->leftOkSeries = 0;
    this->rightSimilarSeries = 0;
    this->leftSimilarSeries = 0;
    this->vanishingPointAvg = Point(0,0);
    this->counter = 0;



};

int LanesDetection::getCannyLowThreshold(){
  return cannyLowThreshold;
}
int LanesDetection::getCannyHighThresholdRatio(){
  return cannyHighThresholdRatio;
}
int LanesDetection::getCannyKernel(){
  return cannyKernel;
}
int LanesDetection::getBlurKernel(){
  return blurKernel;
}
int LanesDetection::getMaskOffsetRatio(){
  return maskOffsetRatio;
}
int LanesDetection::getRectWidthRatio(){
  return rectWidthRatio;
}
int LanesDetection::getRectOffsetRatio(){
  return rectOffsetRatio;
}
int LanesDetection::getNRect(){
  return nRect;
}
int LanesDetection::getRectThicknessRatio(){
  return rectThicknessRatio;
}
int LanesDetection::getTotMinWeight(){
  return totMinWeight;
}
int LanesDetection::getMaxDirChanges(){
  return maxDirChanges;
}
int LanesDetection::getStraightToleranceRatio(){
  return straightToleranceRatio;
}
int LanesDetection::getMaxRmseRatio(){
  return maxRmseRatio;
}
int LanesDetection::getMaxBadCurves(){
  return maxBadCurves;
}
int LanesDetection::getMinGoodCurves(){
  return minGoodCurves;
}
int LanesDetection::getMinBarycenters(){
  return minBarycenters;
}
int LanesDetection::getNextBaryMaxDistance(){
  return nextBaryMaxDistance;
}
int LanesDetection::getRmseTolerance(){
  return rmseTolerance;
}
int LanesDetection::getMinSimilarCurves(){
  return minSimilarCurves;
}
int LanesDetection::getAdjRmseThreshold(){
  return adjRmseThreshold;
}
int LanesDetection::getNLongLines(){
  return nLongLines;
}
float LanesDetection::getMaxSlope(){
  return maxSlope;
}
float LanesDetection::getMinSlope(){
  return minSlope;
}
int LanesDetection::getWindowWidth(){
  return windowWidth;
}
int LanesDetection::getWindowHeight(){
  return windowHeight;
}
int LanesDetection::getHorizonOffsetRatio(){
  horizonOffsetRatio;
}
int LanesDetection::getStraightRange(){
  return straightRange;
}
int LanesDetection::getVanishingPointWindow(){
  return vanishingPointWindow;
}
int LanesDetection::getVanishingPointWindowOffset(){
  return vanishingPointWindowOffset;
}
int LanesDetection::getOrder(){
  return order;
}
int LanesDetection::getNBarycentersWindow(){
  return nBarycentersWindow;
}
Scalar LanesDetection::getRectColor(){
  return rectColor;
}
Scalar LanesDetection::getLastOkFittedColor(){
  return lastOkFittedColor;
}
Scalar LanesDetection::getAvgCurveAvg(){
  return avgCurveAvg;
}
Scalar LanesDetection::getCurFittedColor(){
  return curFittedColor;
}
Scalar LanesDetection::getWhiteFilteringThreshold(){
  return whiteFilteringThreshold;
}
int LanesDetection::getPartialFittingOrder(){
  return partialFittingOrder;
}
bool LanesDetection::getProfile(){
  return profile;
}
int LanesDetection::getInterpolationType(){
  return interpolationType;
}
Camera_Params LanesDetection::getCamera(){
  return camera;
}


void LanesDetection::setCannyLowThreshold(int cannyLowThreshold){
  this->cannyLowThreshold = cannyLowThreshold;
}
void LanesDetection::setCannyHighThresholdRatio(int cannyHighThresholdRatio){
  this->cannyHighThresholdRatio = cannyHighThresholdRatio;
}
void LanesDetection::setCannyKernel(int cannyKernel){
  this->cannyKernel = cannyKernel;
}
void LanesDetection::setBlurKernel(int blurKernel){
  this->blurKernel = blurKernel;
}
void LanesDetection::setMaskOffsetRatio(int maskOffsetRatio){
  this->maskOffsetRatio = maskOffsetRatio;
}
void LanesDetection::setRectWidthRatio(int rectWidthRatio){
  this->rectWidthRatio = rectWidthRatio;
}
void LanesDetection::setRectOffsetRatio(int rectOffsetRatio){
  this->rectOffsetRatio = rectOffsetRatio;
}
void LanesDetection::setNRect(int nRect){
  this->nRect = nRect;
}
void LanesDetection::setRectThicknessRatio(int rectThicknessRatio){
  this->rectThicknessRatio = rectThicknessRatio;
}
void LanesDetection::setTotMinWeight(int totMinWeight){
  this->totMinWeight = totMinWeight;
}
void LanesDetection::setMaxDirChanges(int maxDirChanges){
  this->maxDirChanges = maxDirChanges;
}
void LanesDetection::setStraightToleranceRatio(int straightToleranceRatio){
  this->straightToleranceRatio = straightToleranceRatio;
}
void LanesDetection::setMaxRmseRatio(int maxRmseRatio){
  this->maxRmseRatio = maxRmseRatio;
}
void LanesDetection::setMaxBadCurves(int maxBadCurves){
  this->maxBadCurves = maxBadCurves;
}
void LanesDetection::setMinGoodCurves(int minGoodCurves){
  this->minGoodCurves = minGoodCurves;
}
void LanesDetection::setMinBarycenters(int minBarycenters){
  this->minBarycenters = minBarycenters;
}
void LanesDetection::setNextBaryMaxDistance(int nextBaryMaxDistance){
  this->nextBaryMaxDistance = nextBaryMaxDistance;
}
void LanesDetection::setRmseTolerance(int rmseTolerance){
  this->rmseTolerance = rmseTolerance;
}
void LanesDetection::setMinSimilarCurves(int minSimilarCurves){
  this->minSimilarCurves = minSimilarCurves;
}
void LanesDetection::setAdjRmseThreshold(int adjRmseThreshold){
  this->adjRmseThreshold = adjRmseThreshold;
}
void LanesDetection::setNLongLines(int nLongLines){
  this->nLongLines = nLongLines;
}
void LanesDetection::setMaxSlope(float maxSlope){
  this->maxSlope = maxSlope;
}
void LanesDetection::setMinSlope(float minSlope){
  this->minSlope = minSlope;
}
void LanesDetection::setWindowWidth(int windowWidth){
  this->windowWidth = windowWidth;
}
void LanesDetection::setWindowHeight(int windowHeight){
  this->windowHeight = windowHeight;
}
void LanesDetection::setHorizonOffsetRatio(int horizonOffsetRatio){
  this->horizonOffsetRatio = horizonOffsetRatio;
}
void LanesDetection::setStraightRange(int straightRange){
  this->straightRange = straightRange;
}
void LanesDetection::setVanishingPointWindow(int vanishingPointWindow){
  this->vanishingPointWindow = vanishingPointWindow;
}
void LanesDetection::setVanishingPointWindowOffset(int vanishingPointWindowOffset){
  this->vanishingPointWindowOffset = vanishingPointWindowOffset;
}
void LanesDetection::setOrder(int order){
  this->order = order;
}
void LanesDetection::setNBarycentersWindow(int nBarycentersWindow){
  this->nBarycentersWindow = nBarycentersWindow;
}
void LanesDetection::setRectColor(Scalar rectColor){
  this->rectColor = rectColor;
}
void LanesDetection::setLastOkFittedColor(Scalar lastOkFittedColor){
  this->lastOkFittedColor = lastOkFittedColor;
}
void LanesDetection::setAvgCurveAvg(Scalar avgCurveAvg){
  this->avgCurveAvg = avgCurveAvg;
}
void LanesDetection::setCurFittedColor(Scalar curFittedColor){
  this->curFittedColor = curFittedColor;
}
void LanesDetection::setWhiteFilteringThreshold(Scalar whiteFilteringThreshold){
  this->whiteFilteringThreshold = whiteFilteringThreshold;
}
void LanesDetection::setPartialFittingOrder(int partialFittingOrder){
  this->partialFittingOrder = partialFittingOrder;
}
void LanesDetection::setProfile(bool profile){
  this->profile = profile;
}
void LanesDetection::setInterpolationType(int interpolationType){
  this->interpolationType = interpolationType;
}
void LanesDetection::setCamera(int cameraType){
  this->camera = Camera_Params(cameraType);
}


void LanesDetection::drawRect(vector<Point> rect_points, Scalar rectColor, int height, Mat rectangles){ //draw the rectangles
  const float thickness = height/rectThicknessRatio;
  line( rectangles, rect_points[0], rect_points[1], rectColor, thickness, CV_AA);
  line( rectangles, rect_points[1], rect_points[2], rectColor, thickness, CV_AA);
  line( rectangles, rect_points[2], rect_points[3], rectColor, thickness, CV_AA);
  line( rectangles, rect_points[3], rect_points[0], rectColor, thickness, CV_AA);
}

vector<Point> LanesDetection::computeRect(Point center, int rect_width,int rect_height){ //given the center of the rectangle compute the 4 vertex
  vector<Point> points;
  points.push_back(Point(center.x - rect_width/2, center.y + rect_height/2));
  points.push_back(Point(center.x - rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y - rect_height/2));
  points.push_back(Point(center.x + rect_width/2, center.y + rect_height/2 ));
  return points;
}

void LanesDetection::displayImg(const char* window_name,Mat mat){
  namedWindow( window_name, WINDOW_NORMAL );
  cvResizeWindow(window_name, windowWidth, windowHeight);
  imshow( window_name, mat );
}

Mat LanesDetection::perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints){
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

float LanesDetection::movingAverage(float avg, float new_sample){
  int N = 20;
  if(avg == 0.0){
    return new_sample;
  }
  avg -= avg / N;
  avg += new_sample / N;
  return avg;
}

Mat LanesDetection::calibrateCamera(Mat in){
  Mat out = in.clone();
  double width = in.size().width;
  double height = in.size().height;

  Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
  cameraMatrix.at<double>(0,0) = width*camera.getFxRatio();
  cameraMatrix.at<double>(0,2) = width*camera.getCxRatio();
  cameraMatrix.at<double>(1,1) = height*camera.getFyRatio();
  cameraMatrix.at<double>(1,2) = height*camera.getCyRatio();

  Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
  distCoeffs.at<double>(0,0) = camera.getDist1();
  distCoeffs.at<double>(0,1) = camera.getDist2();
  distCoeffs.at<double>(0,4) = camera.getDist5();
  undistort(in, out, cameraMatrix, distCoeffs);
  return out;
}

vector<Point> LanesDetection::laneConnectedComponent(Mat mat){
  vector<Point> relativeCentroids = vector<Point>();
  Mat labels;
  Mat stats;
  Mat centroids;
  connectedComponentsWithStats(mat, labels, stats, centroids);
  for(int i = 1; i < centroids.size().height; i++){
    relativeCentroids.push_back(Point(centroids.at<double>(i,0), centroids.at<double>(i,1)));
    //circle( mat, relativeCentroids[relativeCentroids.size()-1], 5, Scalar( 0, 255, 0 ),  3, 3 );
  }
  //displayImg("ROI",mat);
  return relativeCentroids;
}

int LanesDetection::distPointToLine(Point P1, Point P2, Point point){
  return ( abs( (P2.x-P1.x)*point.y - (P2.y-P1.y)*point.x + (P2.y)*(P1.x) - (P2.x)*(P1.y) ) ) / ( sqrt( pow(P2.y-P1.y, 2) + pow(P2.x-P1.x, 2) ) );
}

int LanesDetection::distPointToPoint(Point P1, Point P2){
  return sqrt(  pow(P1.x-P2.x, 2) + pow(P1.y-P2.y, 2)  );
}

Point LanesDetection::computeBarycenter(vector<Point> points, Mat mat, vector<Point> &lastOkRectCenters, vector<Point> &rectCenters, vector<Point> barycenters, bool some_curve, vector<Point> lastOkFitted, vector<float> &beta){
  timeval start, end;
  long startMillis, endMillis;
  Point barycenter = Point(-1,-1);
  Point bottomLeft = points[0];
  Point topRight = points[2];

  if(topRight.x > 0 && bottomLeft.x < mat.cols){
    //keep the ROI inside the matrix
    if(bottomLeft.x < 0){
      bottomLeft.x = 0;
    }
    if(topRight.x > mat.cols){
      topRight.x = mat.cols;
    }
    //***** Compute centroids inside the ROI with connected component *****
    Rect rectROI = Rect(bottomLeft, topRight);
    Mat ROI = mat(rectROI);
    vector<Point> centroids = laneConnectedComponent(ROI);


    if (!some_curve) {
      Point relativeBarycenter;

      if(profile){
        gettimeofday(&start, NULL);
        startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
      }
      if(centroids.size() > 0){
        vector<float> beta;
        Point P1;
        Point P2;
        Point absoluteCentroid;
        float minDist;
        int dist;

        if(barycenters.size() < 1){ //first rectangle  //TODO what if, for example, the first is missing?
          barycenter = Point(bottomLeft.x + centroids[0].x, points[1].y + centroids[0].y); //TODO non prendere il primo, decidiamo cosa prendere
        }else if(barycenters.size() == 1) {  //second rectangle
          barycenter = Point(bottomLeft.x + centroids[0].x, points[1].y + centroids[0].y); //TODO non prendere il primo, decidiamo cosa prendere
        }else{  //if more than 2 barycenters found
          if(barycenters.size() > 2){
            beta = polyFit(barycenters, mat, 2);
          }else if(barycenters.size() > 1){
            beta = polyFit(barycenters, mat, 1);
          }
          //compute distance between the curve and each centroid -> chose the one closer to the curve
          float x1 = 0;
          float x2 = 0;
          for(int i = 0; i<beta.size(); i++){
            x1 += beta[i]*pow(bottomLeft.y,i);
            x2 += beta[i]*pow(topRight.y,i);
          }
          P1 = Point(x1, bottomLeft.y);
          P2 = Point(x2, topRight.y);

          barycenter = Point(bottomLeft.x + centroids[0].x, points[1].y + centroids[0].y);
          minDist = distPointToLine(P1, P2, barycenter);
          for(int i = 1; i < centroids.size(); i++){
            absoluteCentroid = Point(bottomLeft.x + centroids[i].x, points[1].y + centroids[i].y);
            dist = distPointToLine(P1, P2, absoluteCentroid);
            //cout << "DIST " << i+1 << ": " << dist << endl;
            if(dist < minDist){
              minDist = dist;
              barycenter = absoluteCentroid;
            }
          }
        }
        if(display){
          circle( mat, barycenter, 5, Scalar( 0, 255, 0 ),  3, 3 );
          //circle( mat, P1, 5, Scalar( 200, 0, 0 ),  2, 2 );
          //circle( mat, P2, 5, Scalar( 200, 0, 0 ),  2, 2 );
          displayImg("barycenter",mat);
        }
      }
      if(profile){
        gettimeofday(&end, NULL);
        endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
        cout << "Component distance analysis: " << endMillis - startMillis << endl;
      }

      //cout << "BARYCENTER: " << barycenter << endl;
    }else{// some_curve


      if(centroids.size() > 0){
        Point P1;
        Point P2;
        Point absoluteCentroid;
        float minDist;
        int dist;

        //compute distance between the curve and each centroid -> chose the one closer to the curve
        float x1 = 0;
        float x2 = 0;
        for(int i = 0; i<beta.size(); i++){
          x1 += beta[i]*pow(bottomLeft.y,i);
          x2 += beta[i]*pow(topRight.y,i);
        }
        P1 = Point(x1, bottomLeft.y);
        P2 = Point(x2, topRight.y);

        barycenter = Point(bottomLeft.x + centroids[0].x, points[1].y + centroids[0].y);
        minDist = distPointToLine(P1, P2, barycenter);
        for(int i = 1; i < centroids.size(); i++){
          absoluteCentroid = Point(bottomLeft.x + centroids[i].x, points[1].y + centroids[i].y);
          dist = distPointToLine(P1, P2, absoluteCentroid);
          //cout << "DIST " << i+1 << ": " << dist << endl;
          if(dist < minDist){
            minDist = dist;
            barycenter = absoluteCentroid;
          }
        }
      }
    }
  }

  return barycenter;
}

vector<float> LanesDetection::polyFit(vector<Point> points, Mat mat, int fitOrder){
  vector<float> beta = vector<float>();
  int width = mat.size().width;
  int height = mat.size().height;
  if(points.size() > fitOrder){
    Mat X = Mat::zeros( points.size(), 1 , CV_32F );
    Mat y = Mat::zeros( points.size(), fitOrder+1 , CV_32F );
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

vector<Point> LanesDetection::computePoly(vector<float> beta, int n_points){
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

int LanesDetection::findHistAcc(Mat mat, int pos, int rect_offset){
  int width = mat.size().width;
  int height = mat.size().height;
  //Compute Histogram
  int histogram[width];
  for(int i = 0; i<width; i++){
    int sum = 0;
    for(int j = height-(height/10); j < (height-rect_offset); j ++){//for(int j = height/2; j < height; j ++){
      Scalar intensity = mat.at<uchar>(j, i);
      if(intensity.val[0] == 255){
        sum++;
      }
      histogram[i] = sum;
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

Mat LanesDetection::curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset){
  int height = mat.size().height;
  int width = mat.size().width;
  Mat mask = Mat::zeros(height,width, CV_8UC1);
  if(display){
    polylines( mask, curve1, 0, 255, offset, 0);
    polylines( mask, curve2, 0, 255, offset, 0);
  }

  return mask;
}

float LanesDetection::computeRmse(vector<Point> curve1, vector<Point> curve2){
  float rmse = rmseTolerance+1;
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

int LanesDetection::dirChanges(vector<Point> points, int tolerance){
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


void LanesDetection::classifyCurve(bool &some_curve, int &curve_bad_series, int &curve_ok_series, vector<Point> barycenters){
  //Classify
  bool curve_ok = false; //0 bad  1 good

  if(barycenters.size() >= minBarycenters){
    curve_ok = true;
  }
  //Update states
  if(curve_ok == false){ //Current curve is bad
    curve_ok_series = 0;
    curve_bad_series++;
  }else if(curve_ok == true){ //Current curve is good
    curve_ok_series++;
    curve_bad_series = 0;
  }

  if(curve_ok_series >= minGoodCurves){
    some_curve = true;
  }
  if(curve_bad_series > maxBadCurves){
    some_curve = false;
  }
}

//find next rect center
Point LanesDetection::nextRectCenter(int y, vector<Point> points, Mat mat, int fitOrder){
  vector<float> beta = polyFit(points, mat, fitOrder);

  int x = 0;
  for(int i = 0; i<beta.size(); i++){
    x += beta[i]*pow(y,i);
  }
  Point p = Point(x , y);
  return p;
}


int LanesDetection::findCurvePoints(bool &some_curve, vector<Point> &rectCenters, vector<Point> & barycenters, int pos, Mat wip, int width, int height,
   int rect_offset, int rect_height, int rect_width, Mat rectangles, vector<Point> &lastOkRectCenters, vector<float> &beta, int offset, vector<Point> lastOkFitted){ //pos: 0=left, 1=right
  timeval start, end;
  long startMillis, endMillis;
  if(some_curve == false){
    rectCenters = vector<Point>();
    //**** First rectangle: histogram *****
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    int firstX = findHistAcc(wip, pos, rect_offset); //0 means left
    if(firstX == -1){  //in caso non trovi il massimo
      firstX = width/4;
    }
    rectCenters.push_back(Point(firstX, height - rect_offset - rect_height/2));


      //**** Other rectangles ****
      for(int i=0;i<nRect-1;i++){//for(int i=0;i<nRect;i++){
        //**** Compute current rectangle ****
        vector<Point> rect = computeRect(rectCenters[i], rect_width, rect_height);

        //*** Compute current barycenter ***
        if(profile){
          gettimeofday(&start, NULL);
          startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
        }

        Point bar = computeBarycenter(rect ,wip, lastOkRectCenters, rectCenters, barycenters, some_curve, lastOkFitted, beta);
        if(profile){
            gettimeofday(&end, NULL);
            endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
            cout << "Barycenter computation: " << endMillis - startMillis << endl;
          }

        //**** Re-compute rectangle ****
        Point nextCenter = Point();
        if(bar.x!=-1 && bar.y!=-1 ){
          rect = computeRect(Point(bar.x, rectCenters[i].y), rect_width, rect_height);
          barycenters.push_back(bar);
          rectCenters[i].x = bar.x;
          if(display){
            circle( rectangles, bar, 5, Scalar( 0, 0, 255 ),  3, 3 ); //draw barycenter
          }
        }

        //**** Compute next rectangle center *****
        if(barycenters.size() > partialFittingOrder){
          vector<Point> lastNBar = vector<Point>();
          for(int j = 0; (j<nBarycentersWindow && j<barycenters.size()); j++){
            lastNBar.push_back(barycenters[barycenters.size()-1-j]);
          }
          nextCenter = nextRectCenter(height - rect_offset - rect_height/2 - (i+1)*rect_height, lastNBar, wip, partialFittingOrder);
        }else{
          nextCenter = Point(rectCenters[i].x, height - rect_offset - rect_height/2 - (i+1)*rect_height);
        }
        rectCenters.push_back(nextCenter);
        if(display){
          circle( rectangles, nextCenter, 5, Scalar( 255, 0, 0 ),  10, 3 );
        }
        //**** Draw updated rectangle ****
        drawRect(rect, rectColor, height, rectangles);

      }
    if(profile){
        gettimeofday(&end, NULL);
        endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
        cout << "Histogram computation: " << endMillis - startMillis << endl;
      }

  }else{// some_curve == true
    //**** rectangles from previous frame *****
    rectCenters = lastOkRectCenters;
    for (int i = 0; i < rectCenters.size(); i++) {
      vector<Point> rect = computeRect(rectCenters[i], rect_width, rect_height);
      Point rCenter = Point(lastOkFitted[lastOkRectCenters[i].y].x, lastOkRectCenters[i].y);
      rectCenters[i] = rCenter;
      Point barycenter = computeBarycenter(rect ,wip, lastOkRectCenters, rectCenters, barycenters, some_curve, lastOkFitted, beta);
      if(barycenters.size()>0 && abs(barycenter.y - barycenters[barycenters.size()-1].y) > rect_height*100){
        barycenter.x = -1;
        barycenter.y = -1;
      }
      if(barycenter.x!=-1 && barycenter.y!=-1 ){
        barycenters.push_back(barycenter);
        rectCenters[i].x = barycenter.x;
      }
      if(display){
        circle( rectangles, barycenter, 5, Scalar( 0, 0, 255 ),  3, 3 ); //draw barycenter
        //**** Draw updated rectangle ****
        drawRect(rect, rectColor, height, rectangles);
      }
    }
  }

  lastOkRectCenters = rectCenters;

  return 0;
}


vector<Point2f> LanesDetection::findPerspectiveInPoints(Mat src, Point &vanishingPointAvg){
  Mat vanishingPointMap = src.clone();
  int height = src.size().height;
  int width = src.size().width;
  const int horizon_offset = height/horizonOffsetRatio;
  //cout << "horizon offset " << horizon_offset << endl;
  vector<Point2f> perspTransfInPoints;

  cvtColor( vanishingPointMap, vanishingPointMap, CV_BGR2GRAY );

  for ( int i = 1; i < blurKernel ; i = i + 2 ){
    GaussianBlur( vanishingPointMap, vanishingPointMap, Size( i, i ), 0, 0 );
  }

  Canny( vanishingPointMap, vanishingPointMap, cannyLowThreshold, cannyLowThreshold*cannyHighThresholdRatio, cannyKernel );


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
  if(hough_lines.size() > nLongLines){  //if there are more lines than the number of lines that we want
    hough_longest_lines = vector<Vec4i>();
    for(int j = 0; j < nLongLines; j++){
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
        if(len > longestLen && abs(slope) > minSlope && abs(slope) < maxSlope ){
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
      if(abs(slope) < maxSlope && abs(slope) > minSlope){
        hough_longest_lines.push_back(l);

      }
    }
  }

  //*** Show Hough ***
  /*for(int i = 0; i<hough_longest_lines.size(); i++){
  cout << "hough_longest_lines: " << hough_longest_lines[i] << endl;
}
cout << hough_lines.size() << endl;
cout << hough_longest_lines.size() << endl;*/
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
    if(x_int > 0 && x_int < width && y_int > 0 && y_int < height){ // y_int > height/3 && y_int < 2*height/3
      intersectionPoints.push_back(intersection_point);
      //draw intersection
      if(display){
        circle( vanishingPointMap, intersection_point, 5, Scalar( 255, 255, 255),  2, 2 ); //white dots
      }
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
  Point new_vanishing_point = Point(x_van_point, y_van_point);
  //cout << "new_vanishing_point: " << new_vanishing_point << endl;
  if(display){
    circle( vanishingPointMap, new_vanishing_point, 5, Scalar( 0, 255, 0),  4, 4 ); //green dot
  }
  if(vanishingPointAvg.x == 0 && vanishingPointAvg.y == 0 ){
    //cout << "vanishingPointAvg: " << vanishingPointAvg << endl;
    vanishingPointAvg = new_vanishing_point;
  }else{
    vanishingPointAvg.x -= vanishingPointAvg.x / vanishingPointWindow;
    vanishingPointAvg.y -= vanishingPointAvg.y / vanishingPointWindow;
    vanishingPointAvg.x += new_vanishing_point.x / vanishingPointWindow;
    vanishingPointAvg.y += new_vanishing_point.y / vanishingPointWindow;
    //cout << "vanishingPointAvg: " << vanishingPointAvg << endl;
  }
  if(display){
    circle( vanishingPointMap, vanishingPointAvg, 5, Scalar( 255, 0, 0),  4, 4 ); //blue dot
  }
  Point vanishing_point = vanishingPointAvg;
  //* Build 2 lines from the vanishing point to the bottom corners *
  float m_left = (float)(height - ((height - vanishing_point.y) - (height - vanishing_point.y)/4) - vanishing_point.y)/(0 - vanishing_point.x); //cout << "m left " << m_left << endl;
  float q_left = vanishing_point.y-m_left*vanishing_point.x;
  float m_right = (float)(height - ((height - vanishing_point.y) - (height - vanishing_point.y)/4) - vanishing_point.y)/(width - vanishing_point.x); //cout << "m right " << m_right << endl;
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
  //cout << "horizon ************* " << horizon << endl;
  //cout << "van ************* " << vanishing_point << endl;
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
  if(display){
    circle( vanishingPointMap, Point(xIntRight1, yIntRight1), 5, Scalar( 0, 255, 255),  4, 2 ); //yellow dots
    circle( vanishingPointMap, Point(xIntRight2, yIntRight2), 5, Scalar( 0, 255, 255),  4, 2 );
    circle( vanishingPointMap, Point(xIntLeft1, yIntLeft1), 5, Scalar( 0, 255, 255),  4, 2 );
    circle( vanishingPointMap, Point(xIntLeft2, yIntLeft2), 5, Scalar( 0, 255, 255),  4, 2 );
  }


  //* Return perspective transform points *
  perspTransfInPoints = vector<Point2f>();
  perspTransfInPoints.push_back(Point(xIntLeft1, yIntLeft1));
  perspTransfInPoints.push_back(Point(xIntLeft2, yIntLeft2));
  perspTransfInPoints.push_back(Point(xIntRight1, yIntRight1));
  perspTransfInPoints.push_back(Point(xIntRight2, yIntRight2));

}

  if(display){
    displayImg("vanishingPointMap",vanishingPointMap);
  }

  return perspTransfInPoints;
}

int LanesDetection::computeDirection(float actualPos, float desiredPos){ // 1 turn right, 0 don't turn, -1 turn left
  if(desiredPos + straightRange - actualPos <  0){
    return 1;
  }else if(desiredPos - straightRange - actualPos > 0){
    return -1;
  }
  return 0;
}


Mat LanesDetection::computeBinaryThresholding(Mat src){ //thresholding with just adaptive threshold on gray scale image
  int height = src.size().height;
  int width = src.size().width;
  //compute binary image/
  Mat wip  = src.clone();
  cvtColor( wip, wip, CV_BGR2GRAY );

  for ( int i = 1; i < blurKernel ; i = i + 2 ){
    GaussianBlur( wip, wip, Size( i, i ), 0, 0, BORDER_DEFAULT );
  }

  inRange(wip, 120,255, wip); //Scalar(150, 150, 150)
  //adaptiveThreshold(wip,wip,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,55,-20);
  threshold(wip,wip,0,255,THRESH_BINARY | THRESH_OTSU);

  //displayImg("adaptiveThreshold", wip);

  return wip;
}


Mat LanesDetection::computeCombinedBinaryThresholding(Mat src){
  int height = src.size().height;
  int width = src.size().width;
  //compute binary image/
  Mat vanishingPointMap = src.clone();
  Mat lightnessMat, saturationMat;
  Mat grayMat = src.clone();

  for ( int i = 1; i < blurKernel ; i = i + 2 ){
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
  //displayImg("planes[2]", planes[2]);

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
  //cout << "lightness: " << lightnessAvg << endl;

  //change s_channel based on l_channel
  for(int i = 0; i < height; i++){
    for(int j = 0; j< width; j++){
      if(lightnessMat.at<uchar>(i,j) < lightnessAvg){
        vanishingPointMap.at<uchar>(i,j) = 0;
      }else{
        vanishingPointMap.at<uchar>(i,j);
      }
    }
  }
  //displayImg("vanishingPointMapThres", vanishingPointMap);

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
  //adaptiveThreshold(vanishingPointMap,vanishingPointMap,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,105,0);
  threshold(vanishingPointMap,vanishingPointMap,0,255,THRESH_BINARY | THRESH_OTSU);

  inRange(abs_grad_x, 50, 255 , abs_grad_x); //Scalar(255, 255, 255)
  adaptiveThreshold(abs_grad_x,abs_grad_x,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,33,0);
  //threshold(abs_grad_x,abs_grad_x,0,255,THRESH_BINARY | THRESH_OTSU);
  //threshold(abs_grad_x,abs_grad_x,THRESH_OTSU,255,THRESH_OTSU);

  //displayImg("abs_grad_x", abs_grad_x);
  //displayImg("vanishingPointMapThres", vanishingPointMap);

  bitwise_or(abs_grad_x, vanishingPointMap, combined_binary);

  //displayImg("combined_binary", combined_binary);

  //end compute binary image/
  return combined_binary;
}

Point getAbsolutePosition(Point relativePosition, vector<Point> roi, int matWidth){
  Point bottomLeft = roi[0];
  Point topRight = roi[2];

  if(topRight.x > 0 && bottomLeft.x < matWidth){
    //keep the ROI inside the matrix
    if(bottomLeft.x < 0){
      bottomLeft.x = 0;
    }
    if(topRight.x > matWidth){
      topRight.x = matWidth;
    }
  }
  return Point(bottomLeft.x + relativePosition.x, topRight.y + relativePosition.y);
}

int LanesDetection::detectLanes(Mat src){

  //Profile
  if(profile){
    cout << "******** New Frame **********" << endl;
  }
  timeval start, end, tot_start, tot_end;
  long startMillis, endMillis, tot_startMillis, tot_endMillis;

  if(profile){
    gettimeofday(&tot_start, NULL);
    tot_startMillis = (tot_start.tv_sec * 1000) + (tot_start.tv_usec / 1000);
  }

  //cout << "* frame *" << endl;
  int turn = 0;
  //* Capture frame *
  Mat wip;
  int width = src.size().width;
  int height = src.size().height;
  const int rect_width = width/rectWidthRatio;
  const int rect_offset = height/rectOffsetRatio;
  const int rect_height = (height - rect_offset)/nRect;
  const int straight_tolerance = width/straightToleranceRatio;
  const int max_rmse = height/maxRmseRatio; //height perchè la parabola orizzontale è calcolata da x a y
  const int mask_offset = height/maskOffsetRatio;



  //*** Camera calibration ***
  if(profile){
    gettimeofday(&start, NULL);
    startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
  }
  src = calibrateCamera(src);
  if(profile){
    gettimeofday(&end, NULL);
    endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
    cout << "Camera calibration: " << endMillis - startMillis << endl;
  }

  wip = src.clone();




  //*** Vanishing Point ***
  if(profile){
    gettimeofday(&start, NULL);
    startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
  }
  vector<Point2f> perspTransfOutPoints;
  if(counter >= vanishingPointWindowOffset && counter < vanishingPointWindow+vanishingPointWindowOffset ){//counter==0){
    perspTransfInPoints = findPerspectiveInPoints(src, vanishingPointAvg);
  }
  if(profile){
    gettimeofday(&end, NULL);
    endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
    cout << "Vanishing point: " << endMillis - startMillis << endl;
  }



  // fixed vanishing point
  //perspTransfInPoints = findPerspectiveInPoints(src, vanishingPointAvg);
  vector<Point2f> test_Points;
  //test_Points = findPerspectiveInPoints(src, vanishingPointAvg);
  if(perspTransfInPoints.size()>0){ //If vanishing point has been found

    //*** Perspective Transform ***
    perspTransfOutPoints.push_back(Point2f( (width/2)-(width/3), (height/2)+(height/2) ));  // perspTransfOutPoints.push_back(Point2f( 0,height));
    perspTransfOutPoints.push_back(Point2f( (width/2)-(width/3), (height/2)-(height/5) ));  // perspTransfOutPoints.push_back(Point2f( 0, 0));
    perspTransfOutPoints.push_back(Point2f( (width/2)+(width/3), (height/2)-(height/5) ));  // perspTransfOutPoints.push_back(Point2f( width, 0));
    perspTransfOutPoints.push_back(Point2f( (width/2)+(width/3), (height/2)+(height/2) ));  // perspTransfOutPoints.push_back(Point2f( width, height));

    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    wip = perspectiveTransform(wip, perspTransfInPoints, perspTransfOutPoints);
    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Perspective transform: " << endMillis - startMillis << endl;
    }

    //**** Curve Mask *****

    Mat leftMat = Mat::zeros(height, width, CV_8U);
    Mat rightMat = Mat::zeros(height, width, CV_8U);
    if(someLeft && someRight){
        Mat leftMask = Mat::zeros(height, width, CV_8U);
        polylines( leftMask, lastOkFittedLeft, 0, 255, mask_offset, 0);
        wip.copyTo(leftMat,leftMask);
        Mat rightMask = Mat::zeros(height, width, CV_8U);
        polylines( rightMask, lastOkFittedRight, 0, 255, mask_offset, 0);
        wip.copyTo(rightMat,rightMask);

    }

    //*** Binary thresholding ***
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    if(someLeft && someRight){
      leftMat = computeBinaryThresholding(leftMat);
      rightMat = computeBinaryThresholding(rightMat);
    }
    displayImg("left", leftMat);
    displayImg("right", rightMat);
    wip = computeBinaryThresholding(wip);


    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Binary thresholding: " << endMillis - startMillis << endl;
    }

    //***** Find curve points ******
    Mat rect_persp;
    Mat rectangles = wip.clone();
    cvtColor( rectangles, rectangles, CV_GRAY2BGR );
    vector<Point> leftRectCenters; //filled by function findCurvePoints
    vector<Point> rightRectCenters;
    vector<Point> leftBarycenters;
    vector<Point> rightBarycenters;
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    if(someLeft && someRight){
      findCurvePoints(someLeft, leftRectCenters, leftBarycenters, 0, leftMat, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkLeftRectCenters, lastOkBetaLeft, mask_offset, lastOkFittedLeft);
      findCurvePoints(someRight, rightRectCenters, rightBarycenters, 1, rightMat, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkRightRectCenters, lastOkBetaRight, mask_offset, lastOkFittedRight);

    }else{
      findCurvePoints(someLeft, leftRectCenters, leftBarycenters, 0, wip, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkLeftRectCenters, lastOkBetaLeft, mask_offset, lastOkFittedLeft);
      findCurvePoints(someRight, rightRectCenters, rightBarycenters, 1, wip, width, height, rect_offset, rect_height, rect_width, rectangles, lastOkRightRectCenters, lastOkBetaRight, mask_offset, lastOkFittedRight);
    }

    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Curve points computation: " << endMillis - startMillis << endl;
    }


    //**** Fit curves *****
    vector<Point> fittedRight;
    vector<Point> fittedLeft;
    vector<float> leftBeta;
    vector<float> rightBeta;

    if(interpolationType == 0){
      //* Least squares nth-nd order polynomial fitting    x = beta_2*y^2 + beta_1*y + beta_0 *
      leftBeta = polyFit(leftBarycenters,wip, order);
      if(profile){
        gettimeofday(&start, NULL);
        startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
      }
      if(leftBeta.size() > 0){
        fittedLeft = computePoly(leftBeta, height);
        lastOkBetaLeft = leftBeta;
      }
      rightBeta = polyFit(rightBarycenters,wip, order);
      if(rightBeta.size() > 0){
        fittedRight = computePoly(rightBeta, height);
        lastOkBetaRight = rightBeta;
      }
      if(profile){
        gettimeofday(&end, NULL);
        endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
        cout << "Curve fitting: " << endMillis - startMillis << endl;
      }
    }

    else if(interpolationType == 1){
      //**** B-spline *******
      if(leftBarycenters.size() > 3){
        tinyspline::BSpline leftSpline(
          3, // ... of degree 3...
          2, // ... in 2D...
          leftBarycenters.size(), // ... consisting of 7 control points...
          TS_CLAMPED // ... using a clamped knot vector.
        );

        // Setup control points.
        std::vector<tinyspline::real> leftCtrlp = leftSpline.ctrlp();
        for(int i=0;i<leftBarycenters.size();i++){
          leftCtrlp[i*2] = leftBarycenters[i].x;
          leftCtrlp[i*2+1] = leftBarycenters[i].y;

        }
        leftSpline.setCtrlp(leftCtrlp);



        // Evaluate `spline` at u = 0.4 using 'evaluate'.
        for(int i=0;i<height;i++){
          // Stores our evaluation results.
          float eval = (float) i/(float) height;
          std::vector<tinyspline::real> result = leftSpline.evaluate(eval).result();
          fittedLeft.push_back(Point(result[0],result[1]));
        }
      }
      else{
        leftBeta = polyFit(leftBarycenters,wip,1);
        if(leftBeta.size() > 0){
          fittedLeft = computePoly(leftBeta, height);
        }
      }


      if(rightBarycenters.size()>3){
        tinyspline::BSpline rightSpline(
          3, // ... of degree 3...
          2, // ... in 2D...
          rightBarycenters.size(), // ... consisting of 7 control points...
          TS_CLAMPED // ... using a clamped knot vector.
        );

        // Setup control points.
        std::vector<tinyspline::real> rightCtrlp = rightSpline.ctrlp();
        for(int i=0;i<rightBarycenters.size();i++){
          rightCtrlp[i*2] = rightBarycenters[i].x;
          rightCtrlp[i*2+1] = rightBarycenters[i].y;
        }
        rightSpline.setCtrlp(rightCtrlp);



        // Evaluate `spline` at u = 0.4 using 'evaluate'.
        for(int i=0;i<height;i++){
          std::vector<tinyspline::real> result = rightSpline.evaluate((float) i/(float) height).result();
          fittedRight.push_back(Point(result[0], result[1]));
        }
      }
      else{
        rightBeta = polyFit(rightBarycenters,wip, order);
        if(rightBeta.size() > 0){
          fittedRight = computePoly(rightBeta, height);
        }
      }


    }




        //**** Classify Curves ****
    /*
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    classifyCurve(someLeft, leftBadSeries, leftOkSeries, leftBarycenters);
    classifyCurve(someRight, rightBadSeries, rightOkSeries, rightBarycenters);
    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Curve classification: " << endMillis - startMillis << endl;
    }*/
    if(fittedLeft.size()>0){
      someLeft = true;
    }else{
      someLeft = false;
    }
    if(fittedRight.size()>0){
      someRight = true;
    }else{
      someRight = false;
    }

    //****Update curves *****
    if(someLeft){
      lastOkFittedLeft = fittedLeft;
    }
    if(someRight){
      lastOkFittedRight = fittedRight;
    }

    //*** Draw curves ****
    if(display){
      polylines( rectangles, lastOkFittedRight, 0, lastOkFittedColor, 8, 0);
      polylines( rectangles, lastOkFittedLeft, 0, lastOkFittedColor, 8, 0);
      polylines( rectangles, fittedLeft, 0, curFittedColor, 8, 0);
      polylines( rectangles, fittedRight, 0, curFittedColor, 8, 0);
    }





    //**** Find average curve *****
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    vector<float> avgBeta = vector<float>();
    vector<Point> avgCurve;
    if(leftBeta.size() > 0 && rightBeta.size() > 0){//someRight && someLeft){
      for(int i=0; i<leftBeta.size(); i++){
        avgBeta.push_back((leftBeta[i]+rightBeta[i])/2);//avgBeta.push_back((lastOkBetaLeft[i]+lastOkBetaRight[i])/2);
      }
      avgCurve = computePoly(avgBeta, height);
      lastOkAvgCurve = avgCurve;
    }
    if(display){
      polylines( rectangles, avgCurve, 0, avgCurveAvg, 8, 0);
    }

    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Middle curve computation: " << endMillis - startMillis << endl;
    }

    //**** Find direction ****
    float dir = 0;
    float u = 0;
    float p = 0.9;
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    for(int i=0; i<avgCurve.size(); i++){
      //dir+=avgCurve[i].x;
      u = p*u + (1-p);
      dir+=u*avgCurve[i].x;
    }
    dir/=avgCurve.size();
    if(display){
      circle( rectangles, Point(dir,height), 5, Scalar( 0, 255, 0 ),  3, 3 );
      circle( rectangles, Point(width/2,height), 5, Scalar( 0, 100, 255 ),  3, 3 );
    }

    turn = computeDirection(dir, width/2);
    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Direction computation: " << endMillis - startMillis << endl;
    }

    //***** Display Images ******
    if(display){
      displayImg("Rectangles",rectangles);
    }

    //displayImg("Wip",wip);
    //displayImg("Src",src);

    //*** Inverse perspective transform ***
    if(profile){
      gettimeofday(&start, NULL);
      startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
    }
    rect_persp = rectangles.clone();
    perspectiveTransform(rect_persp,perspTransfOutPoints,perspTransfInPoints);
    Mat out;
    addWeighted( src, 1, rect_persp, 1, 0.0, out);

    if(display){
      displayImg("Output", out);
    }


    if(profile){
      gettimeofday(&end, NULL);
      endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
      cout << "Inverse perspective: " << endMillis - startMillis << endl;
    }
  }
  if(display){
    displayImg("Input",src);
  }



  counter++;

  if(profile){
    gettimeofday(&tot_end, NULL);
    tot_endMillis  = (tot_end.tv_sec * 1000) + (tot_end.tv_usec / 1000);
    cout << "Tot: " << tot_endMillis - tot_startMillis << endl;
  }

  return turn;

}
