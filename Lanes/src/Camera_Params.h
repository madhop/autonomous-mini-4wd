#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#pragma once

using namespace std;

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
   Camera_Params();
   Camera_Params(int cameraType);
   int getCameraType();
   double getFxRatio();
   double getCxRatio();
   double getFyRatio();
   double getCyRatio();
   double getDist1();
   double getDist2();
   double getDist5();

   void setCameraType(int cameraType);
   void setFxRatio(double fxRatio);
   void setCxRatio(double cxRatio);
   void setFyRatio(double fyRatio);
   void setCyRatio(double cyRatio);
   void setDist1(double dist1);
   void setDist2(double dist2);
   void setDist5(double dist5);

 };
