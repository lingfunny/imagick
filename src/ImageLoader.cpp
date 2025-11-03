#include "ImageLoader.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string readToken(std::istream& is) {
    std::string token;
    while (is >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string discard;
            std::getline(is, discard);
            continue;
        }
        return token;
    }
    throw std::runtime_error("意外到达文件末尾，PPM 数据不完整");
}

cv::Mat readAscii(const std::string& magic, std::istream& is, int width, int height, int maxValue) {
    const bool isColor = magic == "P3";
    const int channels = isColor ? 3 : 1;

    int type = (channels == 1) ? CV_8UC1 : CV_8UC3;

    cv::Mat image(height, width, type);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (channels == 1) {
                const int value = std::stoi(readToken(is));
                if (value < 0 || value > maxValue) {
                    throw std::runtime_error("检测到超出范围的像素值");
                }
                image.at<std::uint8_t>(row, col) = static_cast<std::uint8_t>(value);
            } else {
                auto& pixel = image.at<cv::Vec3b>(row, col);
                for (int ch = 0; ch < channels; ++ch) {
                    const int value = std::stoi(readToken(is));
                    if (value < 0 || value > maxValue) {
                        throw std::runtime_error("检测到超出范围的像素值");
                    }
                    pixel[ch] = static_cast<std::uint8_t>(value);
                }
            }
        }
    }

    return image;
}

cv::Mat readBinaryP6(std::istream& is, int width, int height, int maxValue) {
    if (maxValue > 255) {
        throw std::runtime_error("当前实现暂不支持大于 8 位的二进制 P6 图像");
    }

    const int channels = 3;
    const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channels;

    std::vector<std::uint8_t> buffer(total);
    is.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    if (is.gcount() != static_cast<std::streamsize>(buffer.size())) {
        throw std::runtime_error("P6 图像像素数据长度不匹配");
    }

    cv::Mat image(height, width, CV_8UC3);
    std::memcpy(image.data, buffer.data(), buffer.size());
    return image;
}

void writeHeader(std::ostream& os, const std::string& magic, int width, int height, int maxValue) {
    os << magic << '\n';
    os << width << ' ' << height << '\n';
    os << maxValue << '\n';
}

void writeAscii(const cv::Mat& image, std::ostream& os, int maxValue, bool isColor) {
    const int width = image.cols;
    const int height = image.rows;

    if (isColor) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                if (image.depth() == CV_8U) {
                    const auto pixel = image.at<cv::Vec3b>(row, col);
                    os << static_cast<int>(pixel[0]) << ' '
                       << static_cast<int>(pixel[1]) << ' '
                       << static_cast<int>(pixel[2]) << '\n';
                } else {
                    const auto pixel = image.at<cv::Vec<uint16_t, 3>>(row, col);
                    os << pixel[0] << ' ' << pixel[1] << ' ' << pixel[2] << '\n';
                }
            }
        }
    } else {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                if (image.depth() == CV_8U) {
                    os << static_cast<int>(image.at<std::uint8_t>(row, col)) << ' ';
                } else {
                    os << image.at<std::uint16_t>(row, col) << ' ';
                }
            }
            os << '\n';
        }
    }
}

void writeBinaryP6(const cv::Mat& image, std::ostream& os) {
    if (image.type() != CV_8UC3) {
        throw std::runtime_error("二进制 P6 输出仅支持 8 位 3 通道图像");
    }
    const std::size_t total = static_cast<std::size_t>(image.total()) * image.elemSize();
    os.write(reinterpret_cast<const char*>(image.data), static_cast<std::streamsize>(total));
}

} // namespace

ImageData ImageLoader::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("无法打开文件: " + path);
    }

    ImageData data;
    data.magic = readToken(ifs);
    if (data.magic != "P2" && data.magic != "P3" && data.magic != "P6") {
        throw std::runtime_error("仅支持 P2/P3/P6 格式，检测到: " + data.magic);
    }

    data.width = std::stoi(readToken(ifs));
    data.height = std::stoi(readToken(ifs));
    data.maxValue = std::stoi(readToken(ifs));

    if (data.width <= 0 || data.height <= 0) {
        throw std::runtime_error("图像尺寸非法");
    }
    if (data.maxValue <= 0) {
        throw std::runtime_error("最大像素值必须大于 0");
    }

    if (data.magic == "P2" || data.magic == "P3") {
        data.image = readAscii(data.magic, ifs, data.width, data.height, data.maxValue);
    } else {
        char whitespace = static_cast<char>(ifs.get());
        if (whitespace == '\r' && ifs.peek() == '\n') {
            ifs.get();
        }
        data.image = readBinaryP6(ifs, data.width, data.height, data.maxValue);
    }

    return data;
}

void ImageLoader::save(const std::string& path, const cv::Mat& image, int maxValue, bool useBinaryColor) {
    if (image.empty()) {
        throw std::runtime_error("尝试保存空图像");
    }

    if (image.depth() != CV_8U && image.depth() != CV_16U) {
        throw std::runtime_error("当前仅支持 8 位或 16 位图像保存");
    }

    const int channels = image.channels();
    const bool isColor = channels == 3;

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("无法写入文件: " + path);
    }

    if (isColor) {
        if (useBinaryColor) {
            if (image.depth() != CV_8U) {
                throw std::runtime_error("P6 仅支持 8 位图像");
            }
            writeHeader(ofs, "P6", image.cols, image.rows, maxValue);
            writeBinaryP6(image, ofs);
        } else {
            writeHeader(ofs, "P3", image.cols, image.rows, maxValue);
            writeAscii(image, ofs, maxValue, true);
        }
    } else {
        writeHeader(ofs, "P2", image.cols, image.rows, maxValue);
        writeAscii(image, ofs, maxValue, false);
    }
}
