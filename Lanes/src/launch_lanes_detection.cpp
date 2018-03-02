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

  VideoWriter lanesVideo;
  lanesVideo.open("lanes.avi", VideoWriter::fourcc('P','I','M','1'), 20, Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true); //cap.get(CV_CAP_PROP_FPS)
  VideoWriter rectanglesVideo;
  rectanglesVideo.open("rectangles.avi", VideoWriter::fourcc('P','I','M','1'), 20, Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
  VideoWriter rectanglesBirdVideo;
  rectanglesBirdVideo.open("rectanglesBird.avi", VideoWriter::fourcc('P','I','M','1'), 20, Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);



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
    /*lanesDetection.setRectWidthRatio(23);
    lanesDetection.setMaskOffsetRatio(9);
    lanesDetection.setPerspAnchorOffsetRatio(1);
    lanesDetection.setOrder(2);*/
    vector<vector<Point>> lanes = lanesDetection.detectLanesImage(src);
    //vector<vector<Point3f>> lanes = lanesDetection.detectLanesWorld(src);
    if (lanes.size() == 0) {
      cout << "No lanes detected " << endl;
    }

    /*
    cout << "turn: " << turn << endl;
    gettimeofday(&end, NULL);
    endMillis = (end.tv_sec * 1000) + (end.tv_usec / 1000);
    cout << "Frame elapsed time: " << endMillis - startMillis << endl;
    */
    Mat lanesMat = lanesDetection.getLanesMat();
    Mat rectanglesPerspMat = lanesDetection.getRectanglesPerspMat();
    Mat rectanglesBirdMat = lanesDetection.getRectanglesBirdMat();

    //* Write to video *
    if (lanesMat.size().width > 0 && lanesMat.size().height>0) {
      lanesVideo << lanesMat;
    }
    if (rectanglesPerspMat.size().width > 0 && rectanglesPerspMat.size().height>0) {
      rectanglesVideo << rectanglesPerspMat;
    }
    if (rectanglesBirdMat.size().width > 0 && rectanglesBirdMat.size().height>0) {
      rectanglesBirdVideo << rectanglesBirdMat;
    }

    //* Kill frame *
    //waitKey(0);
    waitKey(100);
  }
  gettimeofday(&tot_end, NULL);
  tot_endMillis = (tot_end.tv_sec * 1000) + (tot_end.tv_usec / 1000);
  cout << "Tot elapsed time: " << tot_endMillis - tot_startMillis << endl;

  return 0;
}
