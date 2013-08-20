/******************************************************************************
 * object detection + tracking class
 *
 * Copyright (C) Bluejeans Network, All Rights Reserved
 *
 * Author: Brian Wang
 * Date:   07/05/2013
 *****************************************************************************/

#include "objdettrack.h"

#include <stdlib.h>
#include <unistd.h>
#include <set>
#include <cmath>

#include "objdtconstants.cpp"

#define DETTRACK_DEBUG 0

//extra include for testing
#include <chrono>
#define BJTRACE(TITLE,TYPE,MSG) std::cout<<TITLE<<" "<<TYPE<<" "<<MSG<<std::endl;
#define ERROR "error"


namespace BJ {

/**
 * The core functionality (detection+tracking) of ObjDetTrack class. 
 * Pass in sequential frames of a video and perform object detection 
 * and/or tracking on these frames.
 *
 * \param currframe Suppose to be a single frame of a sequential video
 * that one wants to detect/track objects.
 *
 * \return a vector of detection/tracking boxes bounding the objects
 * of interest.
 */
std::vector<cv::Rect> 
ObjDetTrack::detTrackSwitch(const cv::Mat& currframe)
{
  //check if the input frames (currframe) changed resolution from the
  //last time that detTrackSwitch is called. If so, drop past results
  //and re-detect and then track.
  if(currframe.size() != expectedFrameSize){
    detOrTrackFlag = 0;
    expectedFrameSize = currframe.size();
  }
  
  //check if we have previous frame. If we do not, then save.
  //optical flow tracking needs previous frame to work.
  if(previousFrame.empty()){
    cvtColor(currframe, previousFrame, cv::COLOR_RGB2GRAY);
    return std::vector<cv::Rect>();
  }
  
  //Choose to perform detection or tracking
  switch(detOrTrackFlag){
    
  case 0: //DETECTION PHASE

    if(numNoObjFrame > 0 && numNoObjFrame%C_MAX_NO_OBJ_THEN_SLEEP_INTERVAL == 0){
      std::chrono::milliseconds t15frames(1500);
      std::this_thread::sleep_for(t15frames);
    }
    
    objTrackWindow = haarCascadeDetect(currframe);
    
    //if detected any object of interest, then set to tracking phase.
    if(objTrackWindow.size() > 0){
      detOrTrackFlag = 1;
      phaseChangeCycle = cycle;

      objHueHist.clear(); //clear old color histogram
      objCornerFeat.clear(); //clear old corner features
      
      startingWindow = objTrackWindow; //make a backup copy of initial detection windows
      
      verifyFailCount.clear();
      verifyFailCount.insert(verifyFailCount.begin(), objTrackWindow.size(), 0);
      
      numNoObjFrame = 0;
    } else{
      numNoObjFrame++;
    }
    
    if(detOrTrackFlag == 0) break;
    //else continue to tracking phase without getting a new frame from video
    //source. That way with quick movement, you would not use outdated 
    //detection window to track on new object detection.
  
  case 1: //TRACKING PHASE

    //meanShiftTracking(currframe);
    opticalFlowTracking(currframe);
    
    //redo detection if too many cycles passed without a detection phase
    if((cycle - phaseChangeCycle)%C_REDO_DETECTION_MAX_INTERVAL == 0){
      std::cout<<"Max tracking interval reached! Redo detection"<<std::endl;
      detOrTrackFlag = 0;
      phaseChangeCycle = cycle;
    }

    /*
    //verify that object can be detected in or around tracking window
    if((cycle - phaseChangeCylce)%C_TRACKING_VERIFY_INTERVAL == 0){
      haarCascadeVerify(*currframe);
      for(uint32_t i=0; i<objTrackWindow.size(); i++){
        if(verifyFailCount[i] >= C_MAX_VERIFY_FAIL_BEFORE_REDETECT){
          detOrTrackFlag = 0;
          phaseChangeCycle = cycle;
          break;
        }
      }
    }
    */

    //redo detection if tracking window area changed too much or much wider than high
    //doesn't matter now since I fixed the window size to starting size always
    for(uint32_t i=0; i<objTrackWindow.size(); i++){
      if(objTrackWindow[i].area() > startingWindow[i].area() * 6.0
        || objTrackWindow[i].area() < startingWindow[i].area() / 6.0
        || objTrackWindow[i].width / objTrackWindow[i].height > 2){
        detOrTrackFlag = 0;
        phaseChangeCycle = cycle;
        break;
      }
    }
    
    //redo detection if large overlap between tracking window
    if((cycle - phaseChangeCycle)%C_TRACKING_CHECK_OVERLAP_INTERVAL == 0 && objTrackWindow.size() > 1){
      uint32_t num_tracked_objects = objTrackWindow.size();
      removeOverlapWindows(objTrackWindow, C_TRACKING_OVERLAP_THRESHOLD);
      if(objTrackWindow.size() < num_tracked_objects){
        detOrTrackFlag = 0;
        phaseChangeCycle = cycle;
        break;
      }
    }

    break;
  }

  cycle++;
  
  cvtColor(currframe, previousFrame, cv::COLOR_RGB2GRAY);

  return objTrackWindow;
}

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
  
/**
 * No longer the tracking algorithm of choice. Not used by DetTrackSwitch.
 * 
 * Do CAMShift tracking (which is essentially tracking on hue of the object
 * with adjustable window size)
 * Create color histogram (1 Dim.) using hue of HSV to describe the object
 * initially, and track the object using this color histogram
 *
 * \todo make camShiftTracking set flags that warn about signs of false detection
 * or tracking going astray.
 */
std::vector<cv::Rect> 
ObjDetTrack::camShiftTracking(const cv::Mat& currframe)
{
  assert(objTrackWindow.size() > 0);

  //assume currframe is in rgb
  //convert to hsv and extract hue
  cv::Mat hsv, hue;
  if(currframe.channels() == 3)
  {
    if(inputFrameType == VIDEO_COLOR_BGR)
    {
      cv::cvtColor(currframe, hsv, cv::COLOR_BGR2HSV);
    } else if(inputFrameType == VIDEO_COLOR_RGB)
    {
      cv::cvtColor(currframe, hsv, cv::COLOR_RGB2HSV);
    } else
    {
      BJTRACE("MH_CV",ERROR,"camShiftTracking can not handle any color other than RGB/BGR.");
      return std::vector<cv::Rect>();
    }
  } else
  {
    BJTRACE("MH_CV",ERROR,"camShiftTracking must be given colored frames");
    return std::vector<cv::Rect>();
  }
  
  int chs[] = {0, 0};
  hue.create(hsv.size(), hsv.depth());
  cv::mixChannels(&hsv, 1, &hue, 1, chs, 1);

  //create mask for pixels too black, white, or gray
  cv::Mat mask;
  int vmin = 10, vmax = 256, smin = 30;
  inRange(hsv, cv::Scalar(0, smin, MIN(vmin, vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

  const int hsize = 32;
  float hranges[] = {0, 180};
  const float* phranges = hranges;
  const int ch = 0;
  //if new objects detected or old object lost or refresh requested, then recreate color histograms
  if(objHueHist.size() != objTrackWindow.size()){
    objHueHist.clear();

    //create color histogram for a new object
    for(uint32_t i=0; i<objTrackWindow.size(); i++){
      cv::Mat roi(hue, objTrackWindow[i]);

      //create a mask that pass through only the oval/ellipse of the face detection window
      //the point was for the mask to only allow skin color into the histogram,
      //and block out background colors in the corners of the rectangle
      //but there is no point anymore, because I am tuning the color histogram purely for skin hues.
      //the code is still here in case I want to turn suppression off.
      // cv::Mat maskellipse = cv::Mat::zeros(mask.size(), CV_8UC1);
      // cv::Rect myrect = objTrackWindow[i];
      // cv::RotatedRect myrotrect = cv::RotatedRect(cv::Point2f(myrect.x+myrect.width/2, myrect.y+myrect.height/2),
      //   cv::Size2f(myrect.width, myrect.height), 0);
      // ellipse( maskellipse, myrotrect, cv::Scalar(255), -1, 8);
      // maskellipse &= mask;
      cv::Mat maskroi(mask, objTrackWindow[i]);

      objHueHist.push_back(cv::Mat());
      calcHist(&roi, 1, &ch, maskroi, objHueHist[i], 1, &hsize, &phranges);
      
#if DETTRACK_DEBUG
      //display color histogram before suppressing non-peak bins
      normalize(objHueHist[i], objHueHist[i], 0, 255, cv::NORM_MINMAX);
      displayColorHist("1", hsize, objHueHist[i]);
#endif

      //blue people suppression
      thereisnobluepeople(objHueHist[i]);
      int farthestBinFromPeak = 3;
      histPeakAccent(objHueHist[i], farthestBinFromPeak);
      normalize(objHueHist[i], objHueHist[i], 0, 255, cv::NORM_MINMAX);

#if DETTRACK_DEBUG
      //display color histogram after suppressing non-peak bins
      displayColorHist("2", hsize, objHueHist[i]);
#endif
    }
  }
 
  std::vector<cv::Rect> boundWindow;
  //backprojection and camshift
  for(uint32_t i=0; i<objTrackWindow.size(); i++){
    cv::Mat backproj;

    calcBackProject(&hue, 1, &ch, objHueHist[i], backproj, &phranges);
    backproj &= mask;

    //meanShift(backproj, objTrackWindow[i], 
    //  cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,10,1) );
    cv::RotatedRect trackBox = CamShift(backproj, objTrackWindow[i], 
      cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT, 10, 1) );
    
    boundWindow.push_back(trackBox.boundingRect());
  } 

  return boundWindow;
}

/**
 * Do mean shift tracking.
 * Create color histogram using hue of HSV to describe the object
 * initially, and track the object using this color histogram.
 */
std::vector<cv::Rect> 
ObjDetTrack::meanShiftTracking(const cv::Mat& currframe)
{
  assert(objTrackWindow.size() > 0);

  //assume currframe is in rgb
  //convert to hsv and extract hue
  cv::Mat hsv;
  if(currframe.channels() == 3)
  {
    if(inputFrameType == VIDEO_COLOR_BGR)
    {
      cv::cvtColor(currframe, hsv, cv::COLOR_BGR2HSV);
    } else if(inputFrameType == VIDEO_COLOR_RGB)
    {
      cv::cvtColor(currframe, hsv, cv::COLOR_RGB2HSV);
    } else
    {
      BJTRACE("MH_CV",ERROR,"meanShiftTracking can not handle any color other than RGB/BGR.");
      return std::vector<cv::Rect>();
    }
  } else
  {
    BJTRACE("MH_CV",ERROR,"Tracking must be given colored frames.");
    return std::vector<cv::Rect>();
  }
  
  int hsv2huesat[] = {0,0, 1,1};
  cv::Mat huesat;
  huesat.create(hsv.size(), CV_MAKETYPE(hsv.depth(),2));
  cv::mixChannels(&hsv, 1, &huesat, 1, hsv2huesat, 1);
  
  //create mask for pixels too black, white, or gray
  cv::Mat mask;
  int vmin = 10, vmax = 256, smin = 30;
  inRange(hsv, cv::Scalar(0, smin, MIN(vmin, vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

  //settings for calcHist
  const int hsize[] = {30, 32}; //quantize hue 30 lvls, sat 32 lvls
  float hranges[] = {0, 180};
  float sranges[] = {0, 256};
  const float* phranges[]= {hranges, sranges};
  const int ch[] = {0, 1};
  
  //if new objects detected or old object lost or refresh requested,
  //then recreate color histograms
  if(objHueHist.size() != objTrackWindow.size()){
    objHueHist.clear();
    
    //create color histogram for a new object
    for(uint32_t i=0; i<objTrackWindow.size(); i++){

      cv::Mat roi(huesat, objTrackWindow[i]);

      //create a mask that pass through only the oval/ellipse of the face detection window
      //the point was for the mask to only allow skin color into the histogram,
      //and block out background colors in the corners of the face bounding rectangle
      cv::Mat maskellipse = cv::Mat::zeros(mask.size(), CV_8UC1);
      cv::Rect myrect = objTrackWindow[i];
      cv::RotatedRect myrotrect = cv::RotatedRect(cv::Point2f(myrect.x+myrect.width/2, myrect.y+myrect.height/2),
         cv::Size2f(myrect.width*0.5, myrect.height*0.7), 0);
      ellipse( maskellipse, myrotrect, cv::Scalar(255), -1, 8);
      maskellipse &= mask;
      cv::Mat maskroi(mask, objTrackWindow[i]);

      objHueHist.push_back(cv::Mat());
      calcHist(&roi, 1, ch, maskroi, objHueHist[i], 2, hsize, phranges);

      //blue people suppression
      //thereisnobluepeople(objHueHist[i]);
      //int farthestBinFromPeak = 3;
      //histPeakAccent(objHueHist[i], farthestBinFromPeak);
      normalize(objHueHist[i], objHueHist[i], 0, 255, cv::NORM_MINMAX);
    }
  }

  //backprojection and meanshift
  for(uint32_t i=0; i<objTrackWindow.size(); i++){
    cv::Mat backproj;

    calcBackProject(&huesat, 1, ch, objHueHist[i], backproj, phranges);
    backproj &= mask;

    meanShift(backproj, objTrackWindow[i], 
      cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,10,1) );
  }

  return objTrackWindow;
}

/**
 * Verify that there is still an object in or around the tracking window
 */
void
ObjDetTrack::haarCascadeVerify(const cv::Mat& currframe)
{
  cv::Mat frame_gray;
  bool presuccess = detectPreProcess(currframe, frame_gray, false);
  if(!presuccess){
    return;
  }
  
  std::vector<cv::Rect> tmp;
  for(uint32_t i=0; i<objTrackWindow.size(); i++){
    cv::Rect expanded = objTrackWindow[i];
    //std::cout<<"pre expanded rect: "<<expanded.x<<" "<<expanded.y<<" "<<expanded.width<<" "<<expanded.height<<" "<<std::endl;
    //expand search window
    expanded -= cv::Point(0.1*expanded.width, 0.1*expanded.height);
    expanded += cv::Size(0.2*expanded.width,0.2*expanded.height);
    //std::cout<<"post expanded rect: "<<expanded.x<<" "<<expanded.y<<" "<<expanded.width<<" "<<expanded.height<<" "<<std::endl;
    
    if(expanded.x < 0){
      expanded.x = 0;
    } else if(expanded.x >= currframe.size().width){
      expanded.x = currframe.size().width-1;
    }
    if(expanded.y < 0){
      expanded.y = 0;
    } else if(expanded.y >= currframe.size().height){
      expanded.y = currframe.size().height-1;
    }
    if(expanded.x + expanded.width > currframe.size().width){
      expanded.width = currframe.size().width - expanded.x;
    }
    if(expanded.y + expanded.height > currframe.size().height){
      expanded.height = currframe.size().height - expanded.y;
    }
    
    cv::Mat window2Verify(frame_gray, expanded);
    tmp = runAllCascadeOnFrame(window2Verify, 1, cv::Size(0.9*objTrackWindow[i].width,0.9*objTrackWindow[i].height));
    if(tmp.size() == 0){
      verifyFailCount[i]++;
      std::cout<<"=failed verify("<<getpid()<<"): "<<verifyFailCount[i]<<"="<<std::endl;
    } else{
      //use the detection to adjust the tracking
      
      std::cout<<"=verify found("<<getpid()<<") "<<tmp.size()<<" faces"<<std::endl;
      int32_t avg_x = tmp[0].x;
      int32_t avg_y = tmp[0].y;
      uint32_t avg_w = tmp[0].width;
      uint32_t avg_h = tmp[0].height;
      for(uint32_t j=1; j<tmp.size(); j++){
        avg_x += tmp[i].x;
        avg_y += tmp[i].y;
        avg_w += tmp[i].width;
        avg_h += tmp[i].height;
      }
      avg_x /= tmp.size();
      avg_y /= tmp.size();
      avg_w /= tmp.size();
      avg_h /= tmp.size();
      objTrackWindow[i] = cv::Rect(avg_x,avg_y,avg_w,avg_h) + cv::Point(expanded.x,expanded.y);
    }
  }
}

/**
 * Helper function of a simple/fast way to smooth the noise out.
 * 
 * \param numLevels number of allowable levels of representation of colors. 
 */
void ObjDetTrack::reducePixelRepresentation(cv::Mat& frame, int numLevels)
{
  //not really quantization (as in picking most frequently occurring 
  //color and do NN to cluster every color to these)

  for(int i=0; i<frame.rows; i++){
    for(int j=0; j<frame.cols; j++){
      frame.at<uchar>(i,j) = floor(frame.at<uchar>(i,j)/numLevels) * numLevels;
    }
  }
}

/**
 * Remove detection windows that have too much overlap with each other.
 * Note that this reduces to the maximum independent set problem,
 * which can not be solved efficiently.
 * However, we assume small amount of detction/tracking windows < 10, 
 * so we could potentially brute force it.
 * But I found that to be too much work, soooo......
 * the below algorithm is a heuristic (only gives probably correct answers)
 * It will do ok, since I expect # of tracking window to be usually less than 5
 */
void ObjDetTrack::removeOverlapWindows(
  std::vector<cv::Rect>& trackingWindow,
	double overlapFracThres)
{
  std::set<uint32_t> rectToRemove_indices;
  std::set<uint32_t>::iterator it;
  for(uint32_t i=0; i<trackingWindow.size(); i++){
    //if i-th rect already marked as to be removed, continue to next rect
    it = rectToRemove_indices.find(i);
    if(it != rectToRemove_indices.end()) continue;
    
    for(uint32_t j=i+1; j<trackingWindow.size(); j++){
      //if j-th rect already marked as to be removed, continue to next rect
      it = rectToRemove_indices.find(j);
      if(it != rectToRemove_indices.end()) continue;
      
      double intersectArea = (trackingWindow[i] & trackingWindow[j]).area();
      double iratioOverlaped = intersectArea/trackingWindow[i].area();
      double jratioOverlaped = intersectArea/trackingWindow[j].area();
      double overlapFrac = std::max(iratioOverlaped, jratioOverlaped);
      if( overlapFrac > overlapFracThres ){
        //mark to be removed the rect with most area overlaped/covered by the other rect
        //either i or j rect overlaps over the threshold amount
        if(iratioOverlaped > jratioOverlaped){
          //i-th rect to be removed
          rectToRemove_indices.insert(i);
        } else{
          //j-th rect to be removed
          rectToRemove_indices.insert(j);
        }
      }     
    }
  }
  
  //carry out the removal
  it = rectToRemove_indices.begin();
  std::vector<cv::Rect> origTrackingWindow(trackingWindow);
  trackingWindow.clear();
  for(uint32_t i=0; i<origTrackingWindow.size(); i++ ){
    //i-th rect not one of the rect marked to be removed
    //or no more of the following rects are marked to be removed
    if( it == rectToRemove_indices.end() || *it != i ){
      //add to the final set of rect
      trackingWindow.push_back(origTrackingWindow[i]);
    } else{
      //assumes rectToRemove_indices is sorted,
      //with smallest index numbers first
      it++;
    }
  }
}

/**
 * Helper function to accent/exaggerate the peak in the color histogram.
 * Not used currently.
 *
 * \param hist assume it is 1 dim color histogram
 * \param farthestBinFromPeak any bin farther from the peak are set to 0
 */
void ObjDetTrack::histPeakAccent(cv::Mat& hist, int farthestBinFromPeak)
{
  float max = 0;
  int max_ind = 0;
  int hsize = hist.size().height;

  //find peak hue
  for(int i=0; i<hsize; i++){
    if(max < hist.at<float>(i)){
      max = hist.at<float>(i);
      max_ind = i;
    }
  }

  if(farthestBinFromPeak <= 0){
    farthestBinFromPeak = 1;
  }

  //hue range wraps around
  for(int i=0; i<hsize; i++){
    int dist2peak = std::min( std::min(abs(i-max_ind),max_ind+(hsize-i)) , (hsize-max_ind)+i );

    //exponential decay hue contribution by distance from peak hue
    if(dist2peak < farthestBinFromPeak){
      hist.at<float>(i) = hist.at<float>(i)*exp(-0.5*dist2peak);
    }
    //set hue contribution to 0 for hues too far from peak hue
    else{
      hist.at<float>(i) = 0;
    }
  }
}

/**
 * Helper function to suppress blue hue bins in the color histogram.
 * Not used currently.
 *
 * \param hist assume it is 1 dim color histogram
 */
void ObjDetTrack::thereisnobluepeople(cv::Mat& hist)
{
  int hsize = hist.size().height;

  //take advantage of the fact that no one's skin hue is blue
  //reduce hue contribution from blue range of the hue
  for(int i=(hsize*2/5); i<(int)(hsize*4/5); i++){
    hist.at<float>(i) = hist.at<float>(i)*0.3;
  }
}

/**
 * Scale cv::Rect(s) based on how much image they are stretched.
 */
void ObjDetTrack::resizeRect(cv::Rect& myrect, double widthScale, double heightScale)
{
  myrect.x = (int) (myrect.x * widthScale);
  myrect.y = (int) (myrect.y * heightScale);

  myrect.width = (int) (myrect.width * widthScale);
  myrect.height = (int) (myrect.height * heightScale);
}

/**
 * For all detectors loaded, run them on frame.
 * 
 * \param minNeighbors default are 2 detection windows each candidate rectangle
 * should have to retain it.
 * \param minSize default is cv::Size(). Size of the smallest detection window
 */
std::vector<cv::Rect> 
ObjDetTrack::runAllCascadeOnFrame(const cv::Mat& frame, int minNeighbors, cv::Size minSize)
{
  std::vector<cv::Rect> allCasResult;

  for(uint32_t i=0; i<allcas.size(); i++){
    std::vector<cv::Rect> casResult;
    allcas[i].detectMultiScale(frame, casResult, 1.1, minNeighbors, 0|cv::CASCADE_SCALE_IMAGE, minSize );
    allCasResult.insert(allCasResult.end(), casResult.begin(), casResult.end());
  }

  return allCasResult;
}

/**
 * Rotate an image and expand the canvas so that the corner don't go out
 * of the boundary. Black pixels for parts of the canvas that the image
 * do not rotate onto.
 * 
 * \param frameAfterRot the place to write the output frame.
 * \param rotDegree degree to rotate the image by. Positive is counter clockwise.
 *
 * \return the matrix to reverse the rotation.
 */
cv::Mat ObjDetTrack::rotateFrame(const cv::Mat& frame, cv::Mat& frameAfterRot, double rotDegree)
{
  cv::Mat rotM = cv::getRotationMatrix2D(1/2*cv::Point2f(frame.cols,frame.rows), rotDegree, 1); 
  std::vector<cv::Point2f> afterRotCorners;
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(0,0)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(frame.cols,0)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(0,frame.rows)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(frame.cols,frame.rows)));

  cv::Rect bRect = minAreaRect(afterRotCorners).boundingRect();
  rotM.at<double>(0,2) = -bRect.x;
  rotM.at<double>(1,2) = -bRect.y;
  warpAffine(frame, frameAfterRot, rotM, bRect.size());

  cv::Mat revRotM;
  invertAffineTransform(rotM, revRotM);

  return revRotM;
}

/**
 * Rotate cv::Rect(s) instead of an image. As the name suggests, use to
 * reverse rotate detection windows from rotated image back to the coordinates
 * of the original image.
 */
std::vector<cv::Rect> ObjDetTrack::revRotOnRects(
  std::vector<cv::Rect> rotDetResult, 
  cv::Mat revRotM, 
  cv::Size2f orig_size)
{
  std::vector<cv::Rect> casResultOrigCoord;

  //Probably use minAreaRect next time I refactor this
  //minAreaRect finds me the bounding RotatedRect which I can use to find 
  //the bounding Rect around the RotatedRect

  for(uint32_t j=0; j<rotDetResult.size(); j++){
    cv::Rect cr = rotDetResult[j];
    cv::Point2f crinit[] = 
      {transformPt(revRotM, cv::Point2f(cr.x,cr.y)),
       transformPt(revRotM, cv::Point2f(cr.x+cr.width,cr.y)),
       transformPt(revRotM, cv::Point2f(cr.x,cr.y+cr.height)),
       transformPt(revRotM, cv::Point2f(cr.x+cr.width,cr.y+cr.height))};

    int min_x = orig_size.width;
    int max_x = 0;
    int min_y = orig_size.height;
    int max_y = 0;
    for(int i=0; i<4; i++){
      min_x = std::min((int)crinit[i].x, min_x);
      max_x = std::max((int)crinit[i].x, max_x);
      min_y = std::min((int)crinit[i].y, min_y);
      max_y = std::max((int)crinit[i].y, max_y);
    }
    min_x = std::max(min_x, 0);
    max_x = std::min(max_x, (int)orig_size.width);
    min_y = std::max(min_y, 0);
    max_y = std::min(max_y, (int)orig_size.height);
    casResultOrigCoord.push_back( cv::Rect(cv::Point2f(min_x,min_y),cv::Size2f(max_x-min_x,max_y-min_y)) );
  }

  return casResultOrigCoord; 
}

/**
 * Affine transform a point
 * \param affM the affine transform matrix
 * \param pt point to be transformed
 * \return transformed point
 */
cv::Point2f ObjDetTrack::transformPt(cv::Mat affM, cv::Point2f pt)
{
  return cv::Point2f(
    affM.at<double>(0,0)*pt.x+affM.at<double>(0,1)*pt.y+affM.at<double>(0,2),
    affM.at<double>(1,0)*pt.x+affM.at<double>(1,1)*pt.y+affM.at<double>(1,2));
}

/**
 * Default constructor
 */
ObjDetTrack::ObjDetTrack()
{
  shrinkratio = 1;
  detOrTrackFlag = 0;
}

/**
 * Full constructor
 * 
 * \param inputFrameTypeInit one of the enums such as 
 * VIDEO_COLOR_YUV, VIDEO_COLOR_RGB, VIDEO_COLOR_BGR
 */
ObjDetTrack::ObjDetTrack(
  std::vector<cv::CascadeClassifier> newAllCas, 
  double newShrinkRatio, 
  int inputFrameTypeInit)
{
  allcas = newAllCas;
  shrinkratio = newShrinkRatio;
  inputFrameType = inputFrameTypeInit;
  detOrTrackFlag = 0;
  
  expectedFrameSize = cv::Size2i(0,0);
}

std::vector<cv::CascadeClassifier> ObjDetTrack::getAllCas()
{
  return allcas;
}

void ObjDetTrack::setAllCas(std::vector<cv::CascadeClassifier> newAllCas)
{
  allcas = newAllCas;
}

const std::vector<cv::Mat>& ObjDetTrack::getObjHueHist()
{
  return objHueHist;
}

void ObjDetTrack::setObjHueHist(std::vector<cv::Mat> newObjHueHist)
{
  objHueHist = newObjHueHist;
}

double ObjDetTrack::getShrinkRatio()
{
  return shrinkratio;
}

void ObjDetTrack::setShrinkRatio(double newShrinkRatio)
{
  if(newShrinkRatio>0)
  {
    shrinkratio = newShrinkRatio;
  }
}

uint32_t ObjDetTrack::getCycle(){
  return cycle;
}

void ObjDetTrack::setCycle(uint32_t newCycleNum)
{
  cycle = newCycleNum;
}

int ObjDetTrack::getDetOrTrackFlag()
{
  return detOrTrackFlag;
}

void ObjDetTrack::setDetOrTrackFlag(int newFlag)
{
  detOrTrackFlag = newFlag;
}

cv::Size2i ObjDetTrack::getExpectedFrameSize()
{
  return expectedFrameSize;
}

} // namespace BJ