///After this many cycles and no object detection, sleep 
#define C_MAX_NO_OBJ_THEN_SLEEP_INTERVAL 50

///Aim to shrink frames to this resolution before detection, in order to
///prevent expensive detection being running on some high definition video
#define C_TARGET_DETECTION_SHRINK_RESOLUTION 720

///After this many tracking cycles, redo detection phase unconditionally
#define C_REDO_DETECTION_MAX_INTERVAL 90

///During tracking cycles, after this many cycles, check for overlap 
///between tracking windows
#define C_TRACKING_CHECK_OVERLAP_INTERVAL 10

///During tracking cycles, this is the threshold overlap percentage of
///the overlapped area out of tracking window area. Above this threshold,
///2 windows are considered overlapped
#define C_TRACKING_OVERLAP_THRESHOLD 0.3

///During tracking cycles, after this many cycles, do a verify
#define C_TRACKING_VERIFY_INTERVAL 60

///During tracking cycles, after this many verify fails, redo detection
#define C_MAX_VERIFY_FAIL_BEFORE_REDETECT 3

#define C_NUM_MIN_CORNER_FEATURES 10