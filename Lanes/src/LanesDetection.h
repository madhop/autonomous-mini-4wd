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
   int canny_low_threshold;
   int canny_high_threshold_ratio;
   int canny_kernel;
   int blur_kernel;
   int mask_offset_ratio;
   int rect_width_ratio;
   int rect_offset_ratio;
   int n_rect;
   int rect_thickness_ratio;
   int tot_min_weight;
   int max_dir_changes;
   int straight_tolerance_ratio;
   int max_rmse_ratio;
   int max_bad_curves;
   int min_good_curves;
   int min_barycenters; //in realtà andrebbe messo come ratio e diviso per n_rect
   int next_bary_max_distance; //anche qui va messo ratio
   int rmse_tolerance;
   int min_similar_curves;
   int adj_rmse_threshold;
   int n_long_lines; //number of lines for vanishing point
   int max_slope;
   float min_slope;
   int window_width;
   int window_height;
   int horizon_offset_ratio;
   int straight_range;//cambiare con ratio
   int vanishing_point_window;
   int vanishing_point_window_offset;
   int order;
   int n_barycenters_window;
   //colors
   Scalar rect_color;
   Scalar last_ok_fitted_color;
   Scalar avg_curve_avg;
   Scalar cur_fitted_color;
   Scalar white_filtering_threshold;

 public:
   // getter and setter
   int getCannyLowThreshold();
   void setCannyLowThreshold(int cannyLowThreshold);
   int getCannyHighThresholdRatio();
   void setCannyHighThresholdRatio(int cannyHighThresholdRatio);
   int getCannyKernel();
   void setCannyKernel(int cannyKernel);
   int blurKernel();
   void blurKernel(int blurKernel);
   int getMaskOffsetRatio();
   void setMaskOffsetRatio(int maskOffsetRatio);
   int getRectWidthRatio();
   void setRectWidthRatio(int rectWidthRatio);
   int getRectOffsetRatio();
   void setRectOffsetRatio(int rectOffsetRatio);

   //Constructor
   LanesDetection();
   LanesDetection(int canny_low_thr, int canny_high_threshold_rat, int canny_k, int blur_k, int mor, int rwr,
     int ror, int num_rect, int rtr, int tmw, int mdc, int str, int mrr,
     int mbc, int mgc, int min_bar, int nbmd, int rmse_tol, int min_similar_c, int adj_rmse_th,
     int n_long_li, int max_sl, float min_sl, int window_w, int window_h, int horizon_offset_rat, int straight_ran, int vanishing_point_wind,
     int vanishing_point_window_offs, int ord, int n_bar_window);
   //functions
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
