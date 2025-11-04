#include "ImageOps.hpp"

#include <stdexcept>

namespace ImageOps {

cv::Mat toGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image.clone();
    }
    if (image.channels() != 3 || image.depth() != CV_8U) {
        throw std::runtime_error("仅支持 8 位三通道彩色图像转换为灰度");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    return gray;
}

cv::Mat scaleByPercentage(const cv::Mat& image, double scale, int interpolation) {
    if (image.empty()) {
        throw std::runtime_error("无法缩放空图像");
    }
    if (scale <= 0.0) {
        throw std::runtime_error("缩放比例必须大于 0");
    }
    if (scale == 1.0) {
        return image.clone();
    }

    cv::Mat result;
    cv::resize(image, result, cv::Size(), scale, scale, interpolation);
    return result;
}

} // namespace ImageOps
