#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "ImageLoader.hpp"

namespace {

enum class OperationType {
    ToGray,
    Resize,
    Scale,
    Rotate,
    Normalize,
    Compress,
    Decompress,
    Save,
    Display
};

struct Operation {
    OperationType type;
    std::string parameter; // 留空表示该操作无需额外参数
};

struct CLIConfig {
    bool infoOnly = false;
    std::optional<bool> forceAscii; // true 表示 P2/P3，false 表示 P6
    std::optional<int> maxValueOverride;
    bool keepMetadata = false;
    std::string inputPath;
    std::string outputPath = "null:";
    std::vector<Operation> operations;
};

void printUsage(std::ostream& os) {
    os << "用法: imagick [全局选项] <输入> [操作 ...] <输出>\n"
       << "示例: imagick data/color-block.ppm -to-gray out.pgm\n\n"
       << "全局选项:\n"
       << "  --help             显示本帮助并退出\n"
       << "  --info             仅输出输入资源的元数据\n"
       << "  --ascii            强制以 P2/P3 ASCII 格式写出\n"
       << "  --binary           强制以 P6 二进制格式写出\n"
       << "  --max-value <n>    指定写出图像的最大像素值\n"
       << "  --keep-metadata    预留选项，后续用于元数据保留\n\n"
       << "操作指令 (按顺序执行):\n"
       << "  -to-gray                   彩色图像转灰度\n"
       << "  -resize WxH[:method]       重采样到指定尺寸\n"
       << "  -scale NxM                 按比例缩放\n"
       << "  -rotate deg                顺时针旋转角度\n"
       << "  -normalize                 灰度归一化\n"
       << "  -compress fmt              压缩为指定格式\n"
       << "  -decompress                从压缩格式解码\n"
       << "  -save path                 在管线中途保存快照\n"
       << "  -display                   调用窗口展示结果\n";
}

bool operationRequiresArgument(OperationType type) {
    switch (type) {
    case OperationType::Resize:
    case OperationType::Scale:
    case OperationType::Rotate:
    case OperationType::Compress:
    case OperationType::Save:
        return true;
    case OperationType::ToGray:
    case OperationType::Normalize:
    case OperationType::Decompress:
    case OperationType::Display:
        return false;
    }
    throw std::logic_error("未知的操作类型");
}

std::string_view operationLabel(OperationType type) {
    switch (type) {
    case OperationType::ToGray:
        return "to-gray";
    case OperationType::Resize:
        return "resize";
    case OperationType::Scale:
        return "scale";
    case OperationType::Rotate:
        return "rotate";
    case OperationType::Normalize:
        return "normalize";
    case OperationType::Compress:
        return "compress";
    case OperationType::Decompress:
        return "decompress";
    case OperationType::Save:
        return "save";
    case OperationType::Display:
        return "display";
    }
    throw std::logic_error("未知的操作类型");
}

OperationType parseOperationToken(const std::string& token) {
    if (token == "-to-gray" || token == "-grayscale") {
        return OperationType::ToGray;
    }
    if (token == "-resize") {
        return OperationType::Resize;
    }
    if (token == "-scale") {
        return OperationType::Scale;
    }
    if (token == "-rotate") {
        return OperationType::Rotate;
    }
    if (token == "-normalize") {
        return OperationType::Normalize;
    }
    if (token == "-compress") {
        return OperationType::Compress;
    }
    if (token == "-decompress") {
        return OperationType::Decompress;
    }
    if (token == "-save") {
        return OperationType::Save;
    }
    if (token == "-display") {
        return OperationType::Display;
    }
    throw std::runtime_error("未知的操作指令: " + token);
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

        if (arg == "--info") {
            config.infoOnly = true;
            continue;
        }
        if (arg == "--ascii") {
            if (config.forceAscii && !*config.forceAscii) {
                throw std::runtime_error("--ascii 与 --binary 冲突");
            }
            config.forceAscii = true;
            continue;
        }
        if (arg == "--binary") {
            if (config.forceAscii && *config.forceAscii) {
                throw std::runtime_error("--ascii 与 --binary 冲突");
            }
            config.forceAscii = false;
            continue;
        }
        if (arg == "--max-value") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--max-value 需要一个整数参数");
            }
            const std::string valueToken = argv[++i];
            try {
                const int value = std::stoi(valueToken);
                if (value <= 0) {
                    throw std::runtime_error("--max-value 必须大于 0");
                }
                config.maxValueOverride = value;
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("--max-value 参数不是有效整数: " + valueToken);
            } catch (const std::out_of_range&) {
                throw std::runtime_error("--max-value 参数超出整数范围: " + valueToken);
            }
            continue;
        }
        if (arg == "--keep-metadata") {
            config.keepMetadata = true;
            continue;
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
        throw std::runtime_error("缺少输入资源路径");
    }
    if (positional.size() > 2) {
        throw std::runtime_error("最多只能指定一个输入和一个输出资源");
    }

    config.inputPath = positional.front();
    if (positional.size() == 2) {
        config.outputPath = positional.back();
    }

    return config;
}

void reportPlan(const CLIConfig& config) {
    std::cout << "命令解析成功。" << std::endl;
    std::cout << "输入资源: " << config.inputPath << std::endl;
    std::cout << "输出资源: " << config.outputPath << std::endl;
    if (config.infoOnly) {
        std::cout << "已启用 --info" << std::endl;
    }
    if (config.forceAscii.has_value()) {
        std::cout << "写出模式: " << (*config.forceAscii ? "ASCII" : "Binary") << std::endl;
    }
    if (config.maxValueOverride.has_value()) {
        std::cout << "写出最大像素值: " << *config.maxValueOverride << std::endl;
    }
    if (config.keepMetadata) {
        std::cout << "保留元数据选项已记录 (功能待实现)" << std::endl;
    }
    if (config.operations.empty()) {
        std::cout << "操作序列为空，后续将扩展默认行为。" << std::endl;
    } else {
        std::cout << "操作序列:" << std::endl;
        for (std::size_t index = 0; index < config.operations.size(); ++index) {
            const auto& op = config.operations[index];
            std::cout << "  [" << index << "] " << operationLabel(op.type);
            if (!op.parameter.empty()) {
                std::cout << " " << op.parameter;
            }
            std::cout << std::endl;
        }
    }
    std::cout << "图像处理管线的执行逻辑将在后续步骤实现。" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CLIConfig config = parseArguments(argc, argv);
        reportPlan(config);
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << std::endl;
        std::cerr << "使用 --help 查看命令说明。" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
