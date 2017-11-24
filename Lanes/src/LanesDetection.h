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

   //Constructor
   LanesDetection();
   //functions
   vector<Point> computeRect(Point center, int rect_width,int rect_height);
   void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles);
   void displayImg(const char* window_name,Mat mat);
   Mat perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints);
   float movingAverage(float avg, float new_sample);
   Point laneConnectedComponent(Mat mat);
   Point computeBarycenter(vector<Point> points, Mat mat);
   vector<float> polyFit(vector<Point> points,Mat mat, int fitOrder);
   int findHistAcc(Mat mat, int pos);
   Mat curve_mask(vector<Point> curve1, vector<Point> curve2, Mat mat, int offset);
   float computeRmse(vector<Point> curve1, vector<Point> curve2);
   int dirChanges(vector<Point> points, int tolerance);
   bool classifyCurve(vector<Point> &fittedCurve, bool &some_curve, int &curve_similar_series, int &curve_bad_series,
   int &curve_ok_series, vector<Point> &lastFittedCurve, vector<Point> &lastOkFittedCurve, vector<Point> &lastOkCurveRectCenters, vector<Point> &curveRectCenters, vector<float> beta, vector<float> &lastOkBeta);
   Point nextRectCenter(int y, vector<Point> points, Mat mat, int fitOrder);
   int findCurvePoints(bool &some_curve, vector<Point> &rectCenters, vector<Point> & barycenters, int pos, Mat wip, int width,
   int height, int rect_offset, int rect_height, int rect_width, Mat rectangles, vector<Point> &lastOkRectCenters, vector<float> &beta, int offset); //pos: 0=left, 1=right
   vector<Point2f> findPerspectiveInPoints(Mat src, Point &vanishing_point_avg);
   vector<Point> computePoly(vector<float> beta, int n_points);
   int computeDirection(float actualPos, float desiredPos);
   Mat computeCombinedBinaryThresholding(Mat src);
   Mat computeBinaryThresholding(Mat src);
   int detectLanes(Mat src, vector<Point> &lastOkFittedRight, vector<Point> &lastOkFittedLeft, vector<Point> &lastOkRightRectCenters,
                   vector<Point> &lastOkLeftRectCenters, vector<Point> &lastFittedRight, vector<Point> &lastFittedLeft,
                   vector<Point2f> &perspTransfInPoints, vector<float> &lastOkBetaLeft, vector<float> &lastOkBetaRight,
                   bool &some_left, bool &some_right, int &left_bad_series, int &right_bad_series, int &right_ok_series,
                   int &left_ok_series, int &right_similar_series, int &left_similar_series, int &counter, Point &vanishing_point_avg);
 };
