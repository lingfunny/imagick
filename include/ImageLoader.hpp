#pragma once

#include <string>

#include <opencv2/core.hpp>

struct ImageData {
    std::string magic;
    int width = 0;
    int height = 0;
    int maxValue = 255;
    cv::Mat image;
};

class ImageLoader {
public:
    static ImageData load(const std::string& path);
    static void save(const std::string& path, const cv::Mat& image, int maxValue = 255, bool useBinaryColor = true);
};
