#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageLoader.hpp"
#include "ImageOps.hpp"

namespace {

enum class OperationType {
    Compress,
    Decompress,
    Grayscale,
    ScalePercent,
    DumpTriples,
    Show
};

struct Operation {
    OperationType type;
    std::string parameter; // 留空表示该操作无需额外参数
};

struct CLIConfig {
    std::string inputPath;
    std::string outputPath;
    std::vector<Operation> operations;
};

void printUsage(std::ostream& os) {
    os << "用法: imagick [选项] <输入> <输出>\n"
       << "示例: imagick -g data/color-block.ppm out/gray.pgm\n"
       << "      imagick -r 50 data/lena-512-gray.ppm out/lena-256.pgm\n\n"
       << "  -h, --help                     显示本帮助并退出\n"
       << "  -g, --grayscale                将图像转换为灰度\n"
       << "  -r, --resize <percentage>      依据百分比对长宽等比例缩放\n"
       << "  -c, --compress                 按默认格式压缩图像\n"
       << "  -x, --extract                  从压缩数据解码图像\n"
       << "  -t, --triples                  导出非零像素三元组\n"
       << "  -s, --show                     在窗口中预览处理结果\n";
}

bool operationRequiresArgument(OperationType type) {
    switch (type) {
    case OperationType::ScalePercent:
        return true;
    case OperationType::Show:
    case OperationType::Compress:
    case OperationType::Decompress:
    case OperationType::Grayscale:
    case OperationType::DumpTriples:
        return false;
    }
    throw std::logic_error("未知的操作类型");
}

OperationType parseOperationToken(const std::string& token) {
    if (token == "-c" || token == "--compress") {
        return OperationType::Compress;
    }
    if (token == "-x" || token == "--extract") {
        return OperationType::Decompress;
    }
    if (token == "-g" || token == "--grayscale") {
        return OperationType::Grayscale;
    }
    if (token == "-r" || token == "--resize") {
        return OperationType::ScalePercent;
    }
    if (token == "-t" || token == "--triples") {
        return OperationType::DumpTriples;
    }
    if (token == "-s" || token == "--show") {
        return OperationType::Show;
    }
    throw std::runtime_error("未知的操作指令: " + token);
}

double parseScalePercentage(const std::string& token) {
    if (token.empty()) {
        throw std::runtime_error("-r 参数不能为空");
    }

    std::string numeric = token;
    if (numeric.back() == '%') {
        numeric.pop_back();
    }
    if (numeric.empty()) {
        throw std::runtime_error("-r 参数不能为空");
    }

    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(numeric, &parsed);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("无法解析缩放百分比: " + token);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("缩放百分比超出范围: " + token);
    }

    if (parsed != numeric.size()) {
        throw std::runtime_error("缩放百分比包含无法识别的字符: " + token);
    }
    if (value <= 0.0) {
        throw std::runtime_error("缩放百分比必须大于 0");
    }

    return value / 100.0;
}

CLIConfig parseArguments(int argc, char** argv) {
    if (argc <= 1) {
        printUsage(std::cout);
        std::exit(EXIT_SUCCESS);
    }

    CLIConfig config;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(std::cout);
            std::exit(EXIT_SUCCESS);
        }

        if (!arg.empty() && arg[0] == '-') {
            OperationType type = parseOperationToken(arg);
            std::string parameter;
            if (operationRequiresArgument(type)) {
                if (i + 1 >= argc) {
                    throw std::runtime_error(arg + " 需要参数");
                }
                parameter = argv[++i];
            }
            config.operations.push_back({type, parameter});
            continue;
        }

        positional.push_back(arg);
    }

    if (positional.empty()) {
        throw std::runtime_error("请指定输入文件路径");
    }
    if (positional.size() > 2) {
        throw std::runtime_error("请指定输入文件路径和输出文件路径");
    }

    config.inputPath = positional.front();
    config.outputPath = positional.back();

    return config;
}

void showImage(const cv::Mat& image, const std::string& windowTitle) {
    if (image.empty()) {
        throw std::runtime_error("无法展示空图像");
    }
    cv::Mat converted;
    const cv::Mat* toDisplay = &image;
    if (image.channels() == 3) {
        cv::cvtColor(image, converted, cv::COLOR_RGB2BGR);
        toDisplay = &converted;
    }
    cv::namedWindow(windowTitle, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowTitle, *toDisplay);
    cv::waitKey(0);
    cv::destroyWindow(windowTitle);
}

cv::Mat runOperations(const std::string& inputPath, const std::vector<Operation>& operations, int& maxValue, bool& preferBinaryColor) {
    ImageData data = ImageLoader::load(inputPath);
    maxValue = data.maxValue;
    preferBinaryColor = data.magic == "P6";

    cv::Mat current = data.image.clone();

    for (const Operation& op : operations) {
        switch (op.type) {
        case OperationType::Grayscale:
            current = ImageOps::toGrayscale(current);
            break;
        case OperationType::ScalePercent: {
            const double factor = parseScalePercentage(op.parameter);
            current = ImageOps::scaleByPercentage(current, factor);
            break;
        }
        case OperationType::Show:
            showImage(current, "result");
            break;
        case OperationType::Compress:
        case OperationType::Decompress:
        case OperationType::DumpTriples:
            throw std::logic_error("压缩和解压操作应在主函数中处理");
        }
    }

    preferBinaryColor = (current.depth() == CV_8U && current.channels() == 3);
    return current;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CLIConfig config = parseArguments(argc, argv);
        
        bool hasDecompress = false;
        bool hasTripleDump = false;
        bool hasShow = false;
        for (const auto& op : config.operations) {
            hasDecompress |= (op.type == OperationType::Decompress);
            hasTripleDump |= (op.type == OperationType::DumpTriples);
            hasShow |= (op.type == OperationType::Show);
        }
        
        if (hasDecompress) {
            for (std::size_t i = 0; i < config.operations.size(); ++i) {
                const auto type = config.operations[i].type;
                if (type == OperationType::Show) {
                    if (i + 1 != config.operations.size()) {
                        throw std::runtime_error("-s 必须位于操作序列末尾");
                    }
                    continue;
                }
                if (type != OperationType::Decompress) {
                    throw std::runtime_error("解压模式下仅支持 -x 以及可选的 -s");
                }
            }
            
            const ImageData data = ImageLoader::decompress(config.inputPath);
            const bool useBinaryColor = (data.image.depth() == CV_8U && data.image.channels() == 3);
            if (hasShow) {
                showImage(data.image, "result");
            }
            ImageLoader::save(config.outputPath, data.image, data.maxValue, useBinaryColor);
            std::cout << "解压完成，结果已保存到: " << config.outputPath << std::endl;
            return EXIT_SUCCESS;
        }

        if (hasTripleDump) {
            if (config.operations.size() != 1) {
                throw std::runtime_error("仅支持单独使用 -t");
            }

            const ImageData data = ImageLoader::load(config.inputPath);
            ImageLoader::saveTriples(config.outputPath, data.image, data.maxValue);
            std::cout << "三元组导出完成，已写入: " << config.outputPath << std::endl;
            return EXIT_SUCCESS;
        }
        
        std::vector<Operation> pipelineOps;
        pipelineOps.reserve(config.operations.size());
        bool hadCompress = false;
        for (std::size_t i = 0; i < config.operations.size(); ++i) {
            const auto& op = config.operations[i];
            if (op.type == OperationType::Compress) {
                if (hadCompress) {
                    throw std::runtime_error("-c 不能重复出现");
                }
                if (i + 1 != config.operations.size()) {
                    throw std::runtime_error("-c 必须位于操作序列末尾");
                }
                hadCompress = true;
            } else {
                pipelineOps.push_back(op);
            }
        }

        int maxValue = 255;
        bool preferBinaryColor = false;
        const cv::Mat result = runOperations(config.inputPath, pipelineOps, maxValue, preferBinaryColor);

        if (hadCompress) {
            ImageLoader::compress(config.outputPath, result, maxValue);
            std::cout << "压缩完成，已写入: " << config.outputPath << std::endl;
        } else {
            ImageLoader::save(config.outputPath, result, maxValue, preferBinaryColor);
            std::cout << "处理完成，已保存到: " << config.outputPath << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << std::endl;
        std::cerr << "使用 --help 查看命令说明。" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
