#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include "LanesDetection.h"

using namespace std;
using namespace cv;



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




LanesDetection lanesDetection = LanesDetection();
timeval tot_start, tot_end;
long tot_startMillis, tot_endMillis;
gettimeofday(&tot_start, NULL);
tot_startMillis = (tot_start.tv_sec * 1000) + (tot_start.tv_usec / 1000);
for(;;){
  Mat src;
  cap >> src;
  if(src.empty()){
    break;
  }

  /*
  timeval start, end;
  long startMillis, endMillis;
  gettimeofday(&start, NULL);
  startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
  */
  //vector<vector<Point>> lanes = lanesDetection.detectLanes(src);
  vector<vector<Point>> lanes = lanesDetection.detectLanesImage(src);
  cout << "How many lanes? " << lanes.size() << endl;
  /*
  cout << "turn: " << turn << endl;
  gettimeofday(&end, NULL);
  endMillis = (end.tv_sec * 1000) + (end.tv_usec / 1000);
  cout << "Frame elapsed time: " << endMillis - startMillis << endl;
  */

  //* Write to video *
  //outputVideo << src;

  //* Kill frame *
  waitKey(0);


}
gettimeofday(&tot_end, NULL);
tot_endMillis = (tot_end.tv_sec * 1000) + (tot_end.tv_usec / 1000);
cout << "Tot elapsed time: " << tot_endMillis - tot_startMillis << endl;

return 0;
}
