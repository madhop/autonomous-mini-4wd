#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include "lanes_detection.h"
#include <unistd.h>
#include <sys/time.h>
#include "LanesDetection.h"

using namespace std;
using namespace cv;

//* Global variables *

#define canny_low_threshold 50
#define canny_high_threshold_ratio 3
#define canny_kernel 3
#define blur_kernel 5
#define mask_offset_ratio 3
#define rect_width_ratio 5
#define rect_offset_ratio 20
#define n_rect 20
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
#define straight_range 3 //cambiare con ratio
#define vanishing_point_window 10
#define vanishing_point_window_offset 1
#define order 2
#define n_barycenters_window 3

/** @function main */
int main( int argc, char** argv ){
  //* Load video *
  VideoCapture cap(argv[1]);//"http://192.168.1.6:8080/?action=stream");//argv[1]); // open the default camera
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
Point vanishing_point_avg = Point(0,0);

int counter = 0;

LanesDetection lanesDetection (canny_low_threshold, canny_high_threshold_ratio, canny_kernel, blur_kernel, mask_offset_ratio, rect_width_ratio,
  rect_offset_ratio, n_rect, rect_thickness_ratio, tot_min_weight, max_dir_changes, straight_tolerance_ratio, max_rmse_ratio, max_bad_curves, min_good_curves,
  min_barycenters, next_bary_max_distance, rmse_tolerance, min_similar_curves, adj_rmse_threshold, n_long_lines, max_slope, min_slope, window_width, window_height,
  horizon_offset_ratio, straight_range, vanishing_point_window, vanishing_point_window_offset, order, n_barycenters_window);

  //LanesDetection *lanesDetection = new LanesDetection();

for(;;){
  Mat src;
  /*
  timeval start;
  gettimeofday(&start, NULL);
  long startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);

  timeval end;
  gettimeofday(&end, NULL);
  long endMillis = (end.tv_sec * 1000) + (end.tv_usec / 1000);
  cout << "elapsed time: " << endMillis - startMillis << endl;*/

  cap >> src;
  lanesDetection.displayImg("src", src);
  cout << "QUI ENTRA!!" << endl;

  int turn = lanesDetection.detectLanes(src,lastOkFittedRight, lastOkFittedLeft, lastOkRightRectCenters, lastOkLeftRectCenters,
                        lastFittedRight, lastFittedLeft, perspTransfInPoints, lastOkBetaLeft, lastOkBetaRight,
                        some_left, some_right, left_bad_series, right_bad_series, right_ok_series,
                        left_ok_series, right_similar_series, left_similar_series, counter, vanishing_point_avg);

  cout << "turn: " << turn << endl;


  //* Write to video *
  //outputVideo << src;
  lanesDetection.displayImg("src", src);

  //* Kill frame *
  waitKey(0);


}
return 0;
}
