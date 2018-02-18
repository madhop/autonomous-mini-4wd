if(profile){
  gettimeofday(&end, NULL);
  endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
  cout << "Middle curve computation: " << endMillis - startMillis << endl;
}

//**** Find direction ****
float dir = 0;
float u = 0;
float p = 0.9;
if(profile){
  gettimeofday(&start, NULL);
  startMillis = (start.tv_sec * 1000) + (start.tv_usec / 1000);
}
for(int i=0; i<avgCurve.size(); i++){
  //dir+=avgCurve[i].x;
  u = p*u + (1-p);
  dir+=u*avgCurve[i].x;
}
dir/=avgCurve.size();
if(display){
  circle( rectangles, Point(dir,height), 5, Scalar( 0, 255, 0 ),  3, 3 );
  circle( rectangles, Point(width/2,height), 5, Scalar( 0, 100, 255 ),  3, 3 );
}

turn = computeDirection(dir, width/2);
if(profile){
  gettimeofday(&end, NULL);
  endMillis  = (end.tv_sec * 1000) + (end.tv_usec / 1000);
  cout << "Direction computation: " << endMillis - startMillis << endl;
}
