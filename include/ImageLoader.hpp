#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

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
    static void compress(const std::string& path, const cv::Mat& image, int maxValue = 255);
    static ImageData decompress(const std::string& path);
    static void saveTriples(const std::string& path, const cv::Mat& image, int maxValue = 255);
    struct PixelTriple {
        int row = 0;
        int col = 0;
        int channels = 0;
        std::array<std::uint8_t, 3> value{0, 0, 0};
    };
    static std::vector<PixelTriple> toTriples(const cv::Mat& image);    // convert matrix to pixel triples
};
