/******************************************************************************
 * object detection + tracking class (2nd version)
 *
 * Copyright (C) Bluejeans Network, All Rights Reserved
 *
 * Author: Brian Wang
 * Date:   07/05/2013
 *****************************************************************************/

#include "objdettrack2.h"

#include <stdlib.h>
#include <unistd.h>
#include <set>
#include <cmath>


namespace BJ {



/**
 * Do pre-processing (down sample, convert to gray scale, and equalize)
 * upon frames before detection.
 *
 * \param preframe expects either YUV, RGB, or Y (the intensity channel of YUV)
 * \param postframe gray scale, color histogram equalized
 * \param performResize whether to perform the resize operations
 */
bool
ObjDetTrack::detectPreProcess(const cv::Mat& preframe, cv::Mat& postframe, bool performResize)
{
  cv::Mat dsframe; //down sampled frame
  if(performResize)
  {
    //lower the resolution so to speed up detection
    //but only shrink to a minimum of 640 (C_TARGET_DETECTION_SHRINK_RESOLUTION)
    if(preframe.cols < C_TARGET_DETECTION_SHRINK_RESOLUTION)
    {
      shrinkratio = 1;
    }
    else
    {
      shrinkratio = C_TARGET_DETECTION_SHRINK_RESOLUTION / (float)preframe.cols;
    }
  
    //apparently if src and dst mat are the same, resize segfaults or error
    //out. In the future, test if this is still the case.
    resize( preframe, dsframe, cv::Size(0,0), shrinkratio, shrinkratio );
  } else{
    dsframe = preframe;
  }

  //convert to gray and equalize
  
  //if already gray, then just equalize
  if(dsframe.channels() == 3)
  {
    switch(inputFrameType)
    {
    case VIDEO_COLOR_YUV:
      BJTRACE("MH_CV",ERROR,"Error: Input frame is YUV. Pass only the Y channel.");
      return false;
    case VIDEO_COLOR_RGB:
      cvtColor( dsframe, postframe, cv::COLOR_RGB2GRAY );
      equalizeHist( postframe, postframe );
      break;
    case VIDEO_COLOR_BGR:
      cvtColor( dsframe, postframe, cv::COLOR_BGR2GRAY );
      equalizeHist( postframe, postframe );  
      break;
    default:
      break;
    }
  } else
  {
    equalizeHist( dsframe, postframe );
  }
  
  reducePixelRepresentation(postframe, (256/16));

  return true;
}

/**
 * Detect objects using cascade detection of haar-wavelets (haar cascade).
 * A variant of Viola-Jones face detection framework.
 *
 * \param currframe Frame that one wants to detect objects.
 * 
 * \param detectAfterRot Whether to rotate currframe by small angles
 * during detection phase. Currently detection is done with haar cascade
 * which is not rotation invariant (can not detect an object if it is 
 * presented rotated). Haar cascade could potentially detect object
 * rotated by a very small angle (+/-15 degrees). Do detection on rotated
 * frames and the straight frame in order to detect the object in
 * much more angular variations.
 */
std::vector<cv::Rect>
ObjDetTrack::haarCascadeDetect(const cv::Mat& currframe, bool detectAfterRot)
{
  cv::Mat frame_gray;
  bool presuccess = detectPreProcess(currframe, frame_gray);
  if(!presuccess){
    return std::vector<cv::Rect>();
  }

  //put all detection results here
  std::vector<cv::Rect> detResult;
  
  //simple straight (no rotation) cascade face/object detection
  std::vector<cv::Rect> straightDetResult = runAllCascadeOnFrame(frame_gray, 2, cv::Size(20*shrinkratio,20*shrinkratio));
  //std::cout<<"detected =="<<(int)straightDetResult.size()<<"== straight"<<std::endl;
  detResult.insert(detResult.end(), straightDetResult.begin(), straightDetResult.end());

  //implements detection after small angle rotations here. Could be really slow
  //Maybe only do this if no straight detResults
  if(detectAfterRot){// && detResult.size() == 0){
    std::vector<double> rotAngles;
    rotAngles.push_back(-30);
    rotAngles.push_back(30);

    for(uint32_t ang_ind=0; ang_ind<rotAngles.size(); ang_ind++){
      cv::Mat frameAfterRot;
      cv::Mat revRotM = rotateFrame(frame_gray, frameAfterRot, rotAngles[ang_ind]);
      std::vector<cv::Rect> rotDetResult = runAllCascadeOnFrame(frameAfterRot, 2, cv::Size(20*shrinkratio,20*shrinkratio));

      std::ostringstream strs;
      strs << rotAngles[ang_ind];
      std::string anglestr = strs.str();
      //std::cout<<"detected =="<<rotDetResult.size()<<"== sideways angle "<<rotAngles[ang_ind]<<std::endl;

      std::vector<cv::Rect> revRotDetResult = revRotOnRects(rotDetResult, revRotM, frame_gray.size());

      detResult.insert(detResult.end(), revRotDetResult.begin(), revRotDetResult.end());
    }
  }

  //eliminate duplicates/overlapping detection
  //they are defined as detection that are overlapping >50% area with other detection rectangles
  removeOverlapWindows(detResult);
  //std::cout<<"=post overlap elimination: detected "<<detResult.size()<<" faces/objects"<<std::endl;

  //reverse downsampling of image, by resizing detection rectangles/windows
  for(uint32_t i=0; i<detResult.size(); i++){
    resizeRect(detResult[i], 1/shrinkratio, 1/shrinkratio);
  }

  return detResult;
}

/**
 * Sparse optical flow tracking (Lucas-Kanade method)
 */
std::vector<cv::Rect> 
ObjDetTrack::opticalFlowTracking(const cv::Mat& currframe)
{
  cv::Mat graycurrframe;
  cvtColor(currframe, graycurrframe, cv::COLOR_RGB2GRAY);
  
  assert(objTrackWindow.size() > 0);

  //if new objects detected or old object lost or refresh requested, then find features again
  if(objCornerFeat.size() != objTrackWindow.size()){
    objCornerFeat.clear();

    for(uint32_t i=0; i<objTrackWindow.size(); i++){
      std::vector<cv::Point2f> cornerFeat;

      cv::Mat objImage(graycurrframe, objTrackWindow[i]);
      
      cv::Mat maskOval = cv::Mat::zeros(objImage.size(), CV_8UC1);
      cv::RotatedRect myrotrect = cv::RotatedRect(cv::Point2f(maskOval.cols/2, maskOval.rows/2),
         cv::Size2f(maskOval.cols*0.7, maskOval.rows), 0);
      ellipse(maskOval, myrotrect, cv::Scalar(255), -1, 8); //draw a filled ellipse

      /* maxCorners=50, qualityLevel=0.01, minDistance=5 */
      double minDistance = std::max((double)std::sqrt(objTrackWindow[i].area()/100) , 5.0);
      goodFeaturesToTrack(objImage, cornerFeat, 50, 0.01, minDistance, maskOval);
      
      //coordinates of objCornerFeat is absolute to the whole image,
      //while coordinates of cornerFeat is relative to the objTrackWindow
      cv::Point2f originOffset(objTrackWindow[i].x, objTrackWindow[i].y);
      for(uint32_t j=0; j<cornerFeat.size(); j++) 
        cornerFeat[j] += originOffset;
      objCornerFeat.push_back(cornerFeat);
      
      //std::cout<<"=|num corner feat:"<<cornerFeat.size()<<"|="<<std::endl;
    }
    
    return objTrackWindow;
  }

  //loop over corner features for every DT window
  for(uint32_t i=0; i<objTrackWindow.size(); i++){
    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(previousFrame, graycurrframe, objCornerFeat[i], nextPts, status, err);
        
    //calculate stddev and mean
    float numValidPts = 0.0;
    cv::Point2f meanPt(0.0,0.0);
    for(uint32_t j=0; j<nextPts.size(); j++){
      if(status[j] == 1){
        meanPt += nextPts[j];
        numValidPts += 1.0;
      }
    }
    meanPt = meanPt * (1.0/numValidPts);
    
    cv::Point2f stddev(0.0,0.0);
    for(uint32_t j=0; j<nextPts.size(); j++){
      if(status[j] == 1){
        cv::Point2f tmp(nextPts[j] - meanPt);
        stddev += cv::Point2f(tmp.x*tmp.x , tmp.y*tmp.y);
      }
    }
    stddev = stddev * (1.0/numValidPts);
    stddev.x = std::sqrt(stddev.x);
    stddev.y = std::sqrt(stddev.y);
    
    //invalidates all points too far from mean
    cv::Point2f ab = stddev * 2.5;
    for(uint32_t j=0; j<nextPts.size(); j++){
      if(status[j] == 1){
        cv::Point2f tmp(nextPts[j] - meanPt);
        if((tmp.x/ab.x)*(tmp.x/ab.x)+(tmp.y/ab.y)*(tmp.y/ab.y)>1.0){
          status[j] = 0;
        }
      }
    }
    
    //remove all corner features that is not found from the tracking set
    //and all points too far from mean
    //remove/erase idiom, except customized std::remove for non-unary predicates
    //modified version of a suggested implementation of std::remove on cppreference.com
    std::vector<cv::Point2f>::iterator ow = objCornerFeat[i].begin();
    for(uint32_t j=0; j<objCornerFeat[i].size(); j++){
      if(status[j] == 1){ //condition such that jth item is not to be removed
        (*ow) = nextPts[j];
        ow++;
      }
    }
    objCornerFeat[i].erase(ow, objCornerFeat[i].end());
    
    //if number of valid corner features are below minimum amount (10), redo detection
    //std::cout<<"=|num corner feat left:"<<objCornerFeat[i].size()<<"|="<<std::endl;
    if(objCornerFeat[i].size() < C_NUM_MIN_CORNER_FEATURES){
      detOrTrackFlag = 0;
    }
    
    //update window is defined by points farthest from the cluster center
    float sm_x = objCornerFeat[i][0].x;
    float lg_x = objCornerFeat[i][0].x;
    float sm_y = objCornerFeat[i][0].y;
    float lg_y = objCornerFeat[i][0].y;
    for(uint32_t j=0; j<objCornerFeat[i].size(); j++){
      if(objCornerFeat[i][j].x < sm_x)
        sm_x = objCornerFeat[i][j].x;
      else if(objCornerFeat[i][j].x > lg_x)
        lg_x = objCornerFeat[i][j].x;
      if(objCornerFeat[i][j].y < sm_y)
        sm_y = objCornerFeat[i][j].y;
      else if(objCornerFeat[i][j].y > lg_y)
        lg_y = objCornerFeat[i][j].y;
    }
    
    if(sm_x < 0) sm_x = 0;
    if(sm_y < 0) sm_y = 0;
    if(lg_x >= graycurrframe.cols) lg_x = graycurrframe.cols-1;
    if(lg_y >= graycurrframe.rows) lg_y = graycurrframe.rows-1;
    
    //objTrackWindow[i] = cv::Rect(sm_x,sm_y,lg_x-sm_x,lg_y-sm_y);
    //objTrackWindow[i] = cv::Rect(meanPt.x-stddev.x*2.0,meanPt.y-stddev.y*2.0,stddev.x*4.0,stddev.y*4.0);
    cv::Size2f refSize = cv::Size2f(startingWindow[i].width,startingWindow[i].height);
    objTrackWindow[i] = cv::Rect(meanPt.x-refSize.width/2,meanPt.y-refSize.height/2,refSize.width,refSize.height);

    /*
    std::cout<<"DT window update:"<<(int)sm_x<<","<<(int)sm_y<<","
      <<(int)(lg_x-sm_x)<<","<<(int)(lg_y-sm_y)<<std::endl;
    std::cout<<"Approx by mean+stddev:"
      <<(int)(meanPt.x-stddev.x*2.0)<<","
      <<(int)(meanPt.y-stddev.y*2.0)<<","
      <<(int)(stddev.x*4.0)<<","
      <<(int)(stddev.y*4.0)<<std::endl;
    */
  } //end loop over corner features for every DT window

  return objTrackWindow;
}

} // namespace BJ