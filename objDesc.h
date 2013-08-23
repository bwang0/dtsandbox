#include "opencv2/opencv.hpp"
#include <vector>

namespace BJ {

class objDesc{
    cv::Rect window;
    std::vector<cv::Point2f> cornerFeat;
    uint32_t id;
};

}
