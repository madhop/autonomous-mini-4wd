#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Camera_Params.h"


using namespace std;
using namespace cv;


Camera_Params::Camera_Params(int cameraType){

  double fx_ratio;
  double cx_ratio;
  double fy_ratio;
  double cy_ratio;
  double dist_1;
  double dist_2;
  double dist_5;

  if(cameraType == 0){ //GoPro Hero 4
   fx_ratio =  0.4544404367948488;
   cx_ratio = 0.5008333333333333;
   fy_ratio = 807.894109857508963;
   cy_ratio = 0.5014814814814815;
   dist_1 = -2.6760855717017523e-01;
   dist_2 = 8.9295931009928706e-02;
   dist_5 = -1.4378376364459476e-02;

 }
 this->cameraType = cameraType;
 this->fxRatio = fx_ratio;
 this->cyRatio = cy_ratio;
 this->fyRatio = fy_ratio;
 this->cyRatio = cy_ratio;
 this->dist1 = dist_1;
 this->dist2 = dist_2;
 this->dist5 = dist_5;
}

double Camera_Params::getFxRatio(){
 return fxRatio;
}
double Camera_Params::getCxRatio(){
 return cxRatio;
}
double Camera_Params::getFyRatio(){
 return fyRatio;
}
double Camera_Params::getCyRatio(){
 return cyRatio;
}
double Camera_Params::getDist1(){
 return dist1;
}
double Camera_Params::getDist2(){
 return dist2;
}
double Camera_Params::getDist5(){
 return dist5;
}
int Camera_Params::getCameraType(){
  return cameraType;
}

void Camera_Params::setFxRatio(double fxRatio){
 this->fxRatio = fxRatio;
}
void Camera_Params::setCxRatio(double cxRatio){
 this->cyRatio = cxRatio;
}
void Camera_Params::setFyRatio(double fyRatio){
 this->fyRatio = fyRatio;
}
void Camera_Params::setCyRatio(double cyRatio){
 this->cyRatio = cyRatio;
}
void Camera_Params::setDist1(double dist1){
 this->dist1 = dist1;
}
void Camera_Params::setDist2(double dist2){
 this->dist2 = dist2;
}
void Camera_Params::setDist5(double dist5){
 this->dist5 = dist5;
}
void Camera_Params::setCameraType(int cameraType){
  this->cameraType = cameraType;
}
