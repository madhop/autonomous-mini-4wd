#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

//* Prototypes *
vector<Point> computeRect(Point center, int rect_width,int rect_height);
void drawRect(vector<Point> rect_points, Scalar rect_color, int thickness, Mat rectangles);
void displayImg(const char* window_name,Mat mat);
Mat perspectiveTransform(Mat mat, vector<Point2f> perspTransfInPoints, vector<Point2f> perspTransfOutPoints);
float movingAverage(float avg, float new_sample);
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
vector<Point2f> findPerspectiveInPoints(Mat src);
vector<Point> computePoly(vector<float> beta, int n_points);
int computeDirection(float actualPos, float desiredPos);
Mat computeCombinedBinaryThresholding(Mat src);
Mat computeBinaryThresholding(Mat src);
int detectLanes(Mat src, vector<Point> &lastOkFittedRight, vector<Point> &lastOkFittedLeft, vector<Point> &lastOkRightRectCenters,
                vector<Point> &lastOkLeftRectCenters, vector<Point> &lastFittedRight, vector<Point> &lastFittedLeft,
                vector<Point2f> &perspTransfInPoints, vector<float> &lastOkBetaLeft, vector<float> &lastOkBetaRight,
                bool &some_left, bool &some_right, int &left_bad_series, int &right_bad_series, int &right_ok_series,
                int &left_ok_series, int &right_similar_series, int &left_similar_series, int &counter, Point &vanishing_point_avg);
