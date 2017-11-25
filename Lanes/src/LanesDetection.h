#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;
/*
 * Lanes detection class
 */
 class LanesDetection{
 private:
   int cannyLowThreshold;
   int cannyHighThresholdRatio;
   int cannyKernel;
   int blurKernel;
   int maskOffsetRatio;
   int rectWidthRatio;
   int rectOffsetRatio;
   int nRect;
   int rectThicknessRatio;
   int totMinWeight;
   int maxDirChanges;
   int straightToleranceRatio;
   int maxRmseRatio;
   int maxBadCurves;
   int minGoodCurves;
   int minBarycenters;
   int nextBaryMaxDistance;
   int rmseTolerance;
   int minSimilarCurves;
   int adjRmseThreshold;
   int nLongLines;
   float maxSlope;
   float minSlope;
   int windowWidth;
   int windowHeight;
   int horizonOffsetRatio;
   int straightRange;
   int vanishingPointWindow;
   int vanishingPointWindowOffset;
   int order;
   int nBarycentersWindow;
   //colors
   Scalar rectColor;
   Scalar lastOkFittedColor;
   Scalar avgCurveAvg;
   Scalar curFittedColor;
   Scalar whiteFilteringThreshold;
   //Dynamic attributes
   bool someLeft;
   bool someRight;
   int leftBadSeries;
   int rightBadSeries;
   int rightOkSeries;
   int leftOkSeries;
   int rightSimilarSeries;
   int leftSimilarSeries;
   Point vanishingPointAvg;
   vector<Point> lastOkFittedRight;
   vector<Point> lastOkFittedLeft;
   vector<Point> lastOkRightRectCenters;
   vector<Point> lastOkLeftRectCenters;
   vector<Point> lastFittedRight;
   vector<Point> lastFittedLeft;
   vector<Point2f> perspTransfInPoints;
   vector<float> lastOkBetaLeft;
   vector<float> lastOkBetaRight;
   int counter;
   //camera calibration
   double fxRatio;
   double cxRatio;
   double fyRatio;
   double cyRatio;
   double dist1;
   double dist2;
   double dist5;



 public:
   // getter and setter
   int getCannyLowThreshold();
   int getCannyHighThresholdRatio();
   int getCannyKernel();
   int getBlurKernel();
   int getMaskOffsetRatio();
   int getRectWidthRatio();
   int getRectOffsetRatio();
   int getNRect();
   int getRectThicknessRatio();
   int getTotMinWeight();
   int getMaxDirChanges();
   int getStraightToleranceRatio();
   int getMaxRmseRatio();
   int getMaxBadCurves();
   int getMinGoodCurves();
   int getMinBarycenters();
   int getNextBaryMaxDistance();
   int getRmseTolerance();
   int getMinSimilarCurves();
   int getAdjRmseThreshold();
   int getNLongLines();
   float getMaxSlope();
   float getMinSlope();
   int getWindowWidth();
   int getWindowHeight();
   int getHorizonOffsetRatio();
   int getStraightRange();
   int getVanishingPointWindow();
   int getVanishingPointWindowOffset();
   int getOrder();
   int getNBarycentersWindow();
   Scalar getRectColor();
   Scalar getLastOkFittedColor();
   Scalar getAvgCurveAvg();
   Scalar getCurFittedColor();
   Scalar getWhiteFilteringThreshold();
   double getFxRatio();
   double getCxRatio();
   double getFyRatio();
   double getCyRatio();
   double getDist1();
   double getDist2();
   double getDist5();

   void setNRect(int nRect);
   void setRectThicknessRatio(int rectThicknessRatio);
   void setTotMinWeight(int totMinWeight);
   void setMaxDirChanges(int maxDirChanges);
   void setStraightToleranceRatio(int straightToleranceRatio);
   void setMaxRmseRatio(int maxRmseRatio);
   void setMaxBadCurves(int setMaxBadCurves);
   void setMinGoodCurves(int minGoodCurves);
   void setMinBarycenters(int minBarycenters);
   void setNextBaryMaxDistance(int nextBaryMaxDistance);
   void setRmseTolerance(int rmseTolerance);
   void setMinSimilarCurves(int minSimilarCurves);
   void setAdjRmseThreshold(int adjRmseThreshold);
   void setNLongLines(int nLongLines);
   void setMaxSlope(float maxSlope);
   void setMinSlope(float minSlope);
   void setWindowWidth(int windowWidth);
   void setWindowHeight(int windowHeight);
   void setHorizonOffsetRatio(int horizonOffsetRatio);
   void setStraightRange(int straightRange);
   void setVanishingPointWindow(int vanishingPointWindow);
   void setVanishingPointWindowOffset(int vanishingPointWindowOffset);
   void setOrder(int order);
   void setNBarycentersWindow(int nBarycentersWindow);
   void setRectColor(Scalar rectColor);
   void setLastOkFittedColor(Scalar lastOkFittedColor);
   void setAvgCurveAvg(Scalar avgCurveAvg);
   void setCurFittedColor(Scalar curFittedColor);
   void setWhiteFilteringThreshold(Scalar whiteFilteringThreshold);
   void setCannyLowThreshold(int cannyLowThreshold);
   void setCannyHighThresholdRatio(int cannyHighThresholdRatio);
   void setCannyKernel(int cannyKernel);
   void setBlurKernel(int blurKernel);
   void setMaskOffsetRatio(int maskOffsetRatio);
   void setRectWidthRatio(int rectWidthRatio);
   void setRectOffsetRatio(int rectOffsetRatio);
   void setFxRatio(double fxRatio);
   void setCxRatio(double cxRatio);
   void setFyRatio(double fyRatio);
   void setCyRatio(double cyRatio);
   void setDist1(double dist1);
   void setDist2(double dist2);
   void setDist5(double dist5);

   //Constructor
   LanesDetection();
   //functions
   vector<Point> computeRect(Point center, int rect_width,int rect_height);
   void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles);
   void displayImg(const char* window_name,Mat mat);
   Mat perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints);
   float movingAverage(float avg, float new_sample);
   Mat calibrateCamera(Mat in);
   vector<Point> laneConnectedComponent(Mat mat);
   int distPointToLine(Point P1, Point P2, Point point);
   int distPointToPoint(Point P1, Point P2);
   Point computeBarycenter(vector<Point> points, Mat mat, vector<Point> lastOkRectCenters, vector<Point> barycenters);
   vector<float> polyFit(vector<Point> points,Mat mat, int fitOrder);
   int findHistAcc(Mat mat, int pos, int rect_offset);
   Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset);
   float computeRmse(vector<Point> curve1, vector<Point> curve2);
   int dirChanges(vector<Point> points, int tolerance);
   void classifyCurve(bool &some_curve, int &curve_bad_series, int &curve_ok_series, vector<Point> barycenters);
   Point nextRectCenter(int y, vector<Point> points, Mat mat, int fitOrder);
   int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, vector<Point> & barycenters, int pos, Mat wip, int width,
   int height, int rect_offset, int rect_height, int rect_width, Mat rectangles, vector<Point> &lastOkRectCenters, vector<float> &beta, int offset); //pos: 0=left, 1=right
   vector<Point2f> findPerspectiveInPoints(Mat src, Point &vanishing_point_avg);
   vector<Point> computePoly(vector<float> beta, int n_points);
   int computeDirection(float actualPos, float desiredPos);
   Mat computeCombinedBinaryThresholding(Mat src);
   Mat computeBinaryThresholding(Mat src);
   int detectLanes(Mat src);
 };
