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
  
  int turn = lanesDetection.detectLanes(src);
  cout << "turn: " << turn << endl;

  //* Write to video *
  //outputVideo << src;


  //* Kill frame *
  waitKey(0);


}
return 0;
}
