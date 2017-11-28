#include <stdlib.h>
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;
/*
 * Camera Params
 */
class Camera_Params{
 private:
   int cameraType;
   double fxRatio;
   double cxRatio;
   double fyRatio;
   double cyRatio;
   double dist1;
   double dist2;
   double dist5;
 public:
   Camera_Params(int cameraType);
   Camera_Params();
   int getCameraType();
   double getFxRatio();
   double getCxRatio();
   double getFyRatio();
   double getCyRatio();
   double getDist1();
   double getDist2();
   double getDist5();

   void setCameraType();
   void setFxRatio();
   void setCxRatio();
   void setFyRatio();
   void setCyRatio();
   void setDist1();
   void setDist2();
   void setDist5();

 };
