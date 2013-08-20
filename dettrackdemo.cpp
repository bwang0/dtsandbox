#include "objdettrack.h"
#include <vector>

std::string fface_cas_fn = "./cvdata/haarcascade_frontalface_alt.xml";
std::string pface_cas_fn = "./cvdata/haarcascade_profileface.xml";

std::string window_name = "Capture - Face detection";
cv::RNG rng(12345);

int main(){
  //initialize capturing
  cv::VideoCapture cap;
  int camNum = 0; //webcam
  cap.open(camNum);
  if( !cap.isOpened() ){
    std::cout << "***Could not initialize capturing...***\n";
    std::cout << "Current parameter's value: \n";
    return -1;
  }
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);

  //load pretrained cascade classifiers in xml format
  cv::CascadeClassifier fface_cas;
  cv::CascadeClassifier pface_cas;
  if( !fface_cas.load( fface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !pface_cas.load( pface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };

  //place all cascade classifiers together
  std::vector<cv::CascadeClassifier> allCas;
  allCas.push_back(fface_cas);
  allCas.push_back(pface_cas);

  //create display window
  cv::namedWindow("haar cascade and corner optical flow tracking", 0);

  cv::Mat currframe;
  BJ::ObjDetTrack faceDT(allCas, 0.7, BJ::VIDEO_COLOR_RGB);

  for(;;)
  {
    cap>>currframe;
    if(currframe.empty()){
      break;
    }

    std::cout<<": "<<currframe.cols<<" "<<currframe.rows<<std::endl;
    std::vector<cv::Rect> dtwindows = faceDT.detTrackSwitch(currframe);

    for(uint32_t i=0; i<dtwindows.size(); i++){
      rectangle(currframe, dtwindows[i], cv::Scalar(255,255,255));
    }

    char c = (char) cv::waitKey(10);
    switch(c){
    case 'd':
      printf("======current shrink ratio : %f\n", faceDT.getShrinkRatio());
      faceDT.setShrinkRatio(faceDT.getShrinkRatio()-0.05);
      break;
    case 'u':
      printf("======current shrink ratio : %f\n", faceDT.getShrinkRatio());
      faceDT.setShrinkRatio(faceDT.getShrinkRatio()+0.05);
      break;
    default:
      break;
    }

    //display
    cv::imshow("haar cascade and corner optical flow tracking", currframe);

  }

}
