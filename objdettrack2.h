/******************************************************************************
 * object detection + tracking class (2nd version)
 *
 * Copyright (C) Bluejeans Network, All Rights Reserved
 *
 * Author: Brian Wang
 * Date:   07/05/2013
 *****************************************************************************/

#ifndef OBJDETTRACK2_H
#define OBJDETTRACK2_H

#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <math.h> 
#include <string>
#include <thread>
#include <chrono>

namespace BJ {

/**
 * Indicates the color encoding cv::Mat frames
 */
enum{
    VIDEO_COLOR_YUV = 0,
    VIDEO_COLOR_RGB = 1, ///< Store Red byte first, then Green, then Blue in memory
    VIDEO_COLOR_BGR = 2  ///< Blue byte, then Green, then Red
};

struct objDesc{
    cv::Rect window;
    std::vector<cv::Point2f> cornerFeat;
    uint32_t id;
};

/**
 * Basic framework for object detection and tracking. Detection and tracking
 * algorithm can be 
 */
class ObjDetTrack{
//most important user facing function
public:
    std::vector<objDesc> detTrackSwitch(const cv::Mat& currframe);

    //constructors 
    ObjDetTrack();
    ObjDetTrack(std::vector<cv::CascadeClassifier> allcas,
                double shrinkratio, int inputFrameTypeInit);
//end most important user face function

public:
    void haarCascadeDetect(const cv::Mat& currframe, bool detectAfterRot=false);
    void opticalFlowTracking(const cv::Mat& currframe);
    void haarCascadeVerify(const cv::Mat& currframe);
    void removeOverlapWindows(std::vector<objDesc>& trackingWindow, double overlapFracThres=0.5);
    bool detectPreProcess(const cv::Mat& preframe, cv::Mat& postframe, bool performResize=true);

    //getters/setters
    std::vector<cv::CascadeClassifier> getAllCas();
    void setAllCas(std::vector<cv::CascadeClassifier> newAllCas);
    const std::vector<cv::Mat>& getObjHueHist();
    void setObjHueHist(std::vector<cv::Mat> newObjHueHist);
    double getShrinkRatio();
    void setShrinkRatio(double newShrinkRatio);
    uint32_t getCycle();
    void setCycle(uint32_t newCycleNum);
    int getDetOrTrackFlag();
    void setDetOrTrackFlag(int newFlag);
    cv::Size2i getExpectedFrameSize();

private:
    ///initialize in the beginning, could add new classifier afterwards.
    std::vector<cv::CascadeClassifier> allcas;

    ///updated each cycle.
    std::vector<cv::Rect> objTrackWindow;

    ///updated after each new successful detection phase.
    std::vector<cv::Rect> startingWindow;

    ///updated upon first tracking cycle after detection. Currently dim 1 hist.
    std::vector<cv::Mat> objHueHist;

    ///number of times a tracking window failed verification
    std::vector<uint32_t> verifyFailCount;

    ///number of frames with no objects detected and/or tracked
    uint32_t numNoObjFrame;

    ///a set of corner features for every single object
    std::vector<std::vector<cv::Point2f> > objCornerFeat;

    ///previous frame saved in gray scale
    cv::Mat previousFrame;

    ///incremented by 1 each time detTrackSwitch is called.
    uint32_t cycle;
    double shrinkratio;
    int detOrTrackFlag;

    ///the type (color format) of the video frames, such as VIDEO_COLOR_YUV, etc.
    int inputFrameType;

    uint32_t phaseChangeCycle;

    /**
    * frame resolution received in the last cycle. A new frame could
    * potentially be of different resolution, and detection/tracking
    * would reset.
    */
    cv::Size2i expectedFrameSize;

    //helpers for CAMshift tracking
    void histPeakAccent(cv::Mat& hist, int farthestBinFromPeak);
    void thereisnobluepeople(cv::Mat& hist);

    //helpers for haar wavelet cascade face/object detection
    std::vector<cv::Rect> runAllCascadeOnFrame(const cv::Mat& frame, int minNeighbors=2, cv::Size minSize=cv::Size());
    void reducePixelRepresentation(cv::Mat& frame, int numLevels);

    //extra functionality beyond opencv's rotation ability
    //keep entire image without cropping parts that don't fit in old frame
    cv::Mat rotateFrame(const cv::Mat& frame, cv::Mat& frameAfterRot, double rotDegree);
    std::vector<cv::Rect> revRotOnRects(std::vector<cv::Rect> rotDetResult, cv::Mat revRotM, cv::Size2f orig_size);

    //general helper/functionality that opencv should have
    cv::Point2f transformPt(cv::Mat affM, cv::Point2f pt);
    void resizeRect(cv::Rect& myrect, double widthScale, double heightScale);
};

} // namespace BJ

#endif