#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ImageOps {

cv::Mat toGrayscale(const cv::Mat& image);

cv::Mat scaleByPercentage(const cv::Mat& image, double scale, int interpolation = cv::INTER_LINEAR);

} // namespace ImageOps
