#pragma once
#include <algorithm>
#include <cmath>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 255
#include "magic_enum.hpp"

namespace scripts
{
    typedef float Float;
    typedef size_t UInt;
    typedef unsigned char Byte;

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    static const auto nwl = std::string("\r\n");
#else // assuming Linux
    static const auto nwl = std::string("\n");
#endif

    enum class Scripts
    {
        densenet = 0,
        efficientnetv2 = 1,
        mobilenetv3 = 2,
        resnet = 3,
        shufflenetv2 = 4
    };

    enum class Datasets
    {
        cifar10 = 0,
        cifar100 = 1,
        fashionmnist = 2,
        mnist = 3,
        tinyimagenet = 4
    };

    enum class Fillers
    {
        Constant = 0,
        HeNormal = 1,
        HeUniform = 2,
        LeCunNormal = 3,
        LeCunUniform = 4,
        Normal = 5,
        TruncatedNormal = 6,
        Uniform = 7,
        XavierNormal = 8,
        XavierUniform = 9
    };

    enum class Activations
    {
        FRelu = 1,
        HardSwish = 10,
        Mish = 16,
        Relu = 19,
        Swish = 25,
        TanhExp = 27
    };

    static const auto StringToLower(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);
        return text;
    };

    constexpr auto ScaleVisible(const Fillers filler)
    {
        switch (filler)
        {
        case Fillers::Constant:
        case Fillers::Normal:
        case Fillers::TruncatedNormal:
        case Fillers::Uniform:
            return true;
        default:
            return false;
        }
    }

    struct EffNetRecord
    {
        UInt ExpandRatio;
        UInt Channels;
        UInt Iterations;
        UInt Stride;
        bool SE;
    };

    struct ScriptParameters
    {
        // Model defaullt parameters
        scripts::Scripts Script;
        scripts::Datasets Dataset;
        UInt C;
        UInt D = 1;
        UInt H;
        UInt W;
        UInt PadD = 0;
        UInt PadH = 0;
        UInt PadW = 0;
        bool MirrorPad = false;
        bool MeanStdNormalization = true;
        scripts::Fillers WeightsFiller = Fillers::HeNormal;
        Float WeightsScale = Float(0.05);
        Float WeightsLRM = Float(1);
        Float WeightsWDM = Float(1);
        bool HasBias = false;
        scripts::Fillers BiasesFiller = Fillers::Constant;
        Float BiasesScale = Float(0);
        Float BiasesLRM = Float(1);
        Float BiasesWDM = Float(1);
        bool BatchNormScaling = false;
        Float BatchNormMomentum = Float(0.995);
        Float BatchNormEps = Float(0.0001);
        Float Alpha = Float(0);
        Float Beta = Float(0);
        // Model common parameters
        UInt Groups;
        UInt Iterations;
        // Model specific parameters
        UInt Width;
        UInt GrowthRate;
        Float Dropout;
        Float Compression;
        bool Bottleneck;
        bool SqueezeExcitation;
        bool ChannelZeroPad;
        scripts::Activations Activation = Activations::Relu;
        std::vector<EffNetRecord> EffNet = { { 1, 24, 2, 1, false }, { 4, 48, 4, 2, false }, { 4, 64, 4, 2, false }, { 4, 128, 6, 2, true }, { 6, 160, 9, 1, true }, { 6, 256, 15, 2, true } };

        UInt Classes() const
        {
            switch (Dataset)
            {
            case Datasets::cifar10:
            case Datasets::fashionmnist:
            case Datasets::mnist:
                return 10;
            case Datasets::cifar100:
                return 100;
            case Datasets::tinyimagenet:
                return 200;
            default:
                return 0;
            }
        }

        bool RandomCrop() const
        {
            return PadH > 0 || PadW > 0;
        }

        UInt Depth() const
        {
            switch (Script)
            {
            case Scripts::densenet:
                return (Groups * Iterations * (Bottleneck ? 2u : 1u)) + ((Groups - 1) * 2);
            case Scripts::mobilenetv3:
                return (Groups * Iterations * 3) + ((Groups - 1) * 2);
            case Scripts::resnet:
                return (Groups * Iterations * (Bottleneck ? 3u : 2u)) + ((Groups - 1) * 2);
            case Scripts::shufflenetv2:
                return (Groups * (Iterations - 1) * 3) + (Groups * 5) + 1;
            default:
                return 0;
            }
        }

        bool WidthVisible() const { return Script == Scripts::mobilenetv3 || Script == Scripts::resnet || Script == Scripts::shufflenetv2; }
        bool GrowthRateVisible() const { return Script == Scripts::densenet; }
        bool DropoutVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet; }
        bool CompressionVisible() const { return Script == Scripts::densenet; }
        bool BottleneckVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet; }
        bool SqueezeExcitationVisible() const { return Script == Scripts::mobilenetv3 || Script == Scripts::shufflenetv2; }
        bool ChannelZeroPadVisible() const { return Script == Scripts::resnet; }
        bool EffNetVisible() const { return Script == Scripts::efficientnetv2; }
        
        auto GetName() const
        {
            auto common = std::string(magic_enum::enum_name<Scripts>(Script)) + std::string("-") + std::to_string(H) + std::string("x") + std::to_string(W) + std::string("-") + std::to_string(Groups) + std::string("-") + std::to_string(Iterations) + std::string("-");

            switch (Script)
            {
            case Scripts::densenet:
                return common + std::to_string(GrowthRate) + (Dropout > 0 ? std::string("-dropout") : std::string("")) + (Compression > 0 ? std::string("-compression") : std::string("")) + (Bottleneck ? std::string("-bottleneck") : std::string("")) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation)));
            case Scripts::efficientnetv2:
                return std::string(magic_enum::enum_name<Scripts>(Script)) + std::string("-") + std::to_string(H) + std::string("x") + std::to_string(W) + std::string("-");
            case Scripts::mobilenetv3:
                return common + std::to_string(Width) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation))) + (SqueezeExcitation ? std::string("-se") : std::string(""));
            case Scripts::resnet:
                return common + std::to_string(Width) + (Dropout > 0 ? std::string("-dropout") : std::string("")) + (Bottleneck ? std::string("-bottleneck") : std::string("")) + (ChannelZeroPad ? std::string("-channelzeropad") : std::string("")) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation)));
            case Scripts::shufflenetv2:
                return common + std::to_string(Width) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation))) + (SqueezeExcitation ? std::string("-se") : std::string(""));
            default:
                return common;
            }
        };
    };

    class ScriptsCatalog
    {
    public:
        static auto to_string(const bool variable)
        {
            return variable ? std::string("Yes") : std::string("No");
        }

        static auto to_string(const Datasets dataset)
        {
            return std::string(magic_enum::enum_name<Datasets>(dataset));
        }

        static auto to_string(const Fillers filler)
        {
            return std::string(magic_enum::enum_name<Fillers>(filler));
        }

        static UInt DIV8(UInt channels)
        {
            if (channels % 8ull == 0ull)
                return channels;

            return ((channels / 8ull) + 1ull) * 8ull;
        }

        static std::string In(std::string prefix, UInt id)
        {
            return prefix + std::to_string(id);
        }

        static std::string BatchNorm(UInt id, std::string inputs, std::string group = "", std::string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNorm" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string BatchNormActivation(UInt id, std::string inputs, std::string activation = "Relu", std::string group = "", std::string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNorm" + activation + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string BatchNormActivation(UInt id, std::string inputs, scripts::Activations activation = scripts::Activations::Relu, std::string group = "", std::string prefix = "B")
        {
            if (activation != scripts::Activations::FRelu)
            {
                return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNorm" + std::string(magic_enum::enum_name<scripts::Activations>(activation)) + nwl +
                    "Inputs=" + inputs + nwl + nwl;
            }
            else
            {
                return "[" + group + "B" + std::to_string(id) + "B1]" + nwl +
                    "Type=BatchNorm" + nwl +
                    "Inputs=" + inputs + nwl + nwl +

                    "[" + group + "DC" + std::to_string(id) + "DC]" + nwl +
                    "Type=DepthwiseConvolution" + nwl +
                    "Inputs=" + group + "B" + std::to_string(id) + "B1" + nwl +
                    "Kernel=3,3" + nwl +
                    "Pad=1,1" + nwl + nwl +

                    "[" + group + "B" + std::to_string(id) + "B2]" + nwl +
                    "Type=BatchNorm" + nwl +
                    "Inputs=" + group + "DC" + std::to_string(id) + "DC" + nwl + nwl +

                    "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=Max" + nwl +
                    "Inputs=" + group + "B" + std::to_string(id) + "B2," + group + "B" + std::to_string(id) + "B1" + nwl + nwl;
            }
        }

        static std::string BatchNormActivationDropout(UInt id, std::string inputs, scripts::Activations activation = scripts::Activations::Relu, Float dropout = 0.0f, std::string group = "", std::string prefix = "B")
        {
            if (activation != scripts::Activations::FRelu)
            {
                return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNorm" + std::string(magic_enum::enum_name<scripts::Activations>(activation)) + "Dropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    (dropout > 0.0f ? "Dropout=" + std::to_string(dropout) + nwl + nwl : nwl);
            }
            else
            {
                return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNormHardSwishDropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    (dropout > 0.0f ? "Dropout=" + std::to_string(dropout) + nwl + nwl : nwl);
            }
        }

        static std::string Convolution(UInt id, std::string inputs, UInt channels, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, std::string group = "", std::string prefix = "C", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Convolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + std::to_string(channels) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string DepthwiseConvolution(UInt id, std::string inputs, UInt multiplier = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, std::string group = "", std::string prefix = "DC", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=DepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                (multiplier > 1 ? "Mulltiplier=" + std::to_string(multiplier) + nwl : "") +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string PartialDepthwiseConvolution(UInt id, std::string inputs, UInt part = 1, UInt groups = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, std::string group = "", std::string prefix = "DC", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=PartialDepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Group=" + std::to_string(part) + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string DepthwiseMixedConvolution(UInt g, UInt id, std::string inputs, UInt strideX = 1, UInt strideY = 1, bool useChannelSplit = true, std::string group = "", std::string prefix = "DC")
        {
            switch (g)
            {
            case 0:
                return DepthwiseConvolution(id, inputs, 1, 3, 3, strideX, strideY, 1, 1, group, prefix);

            case 1:
                return useChannelSplit ? ChannelSplit(id, inputs, 2, 1, "Q1") + ChannelSplit(id, inputs, 2, 2, "Q2") +
                    DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, "B") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix) :
                    PartialDepthwiseConvolution(id, inputs, 1, 2, 3, 3, strideX, strideY, 1, 1, "A") + PartialDepthwiseConvolution(id, inputs, 2, 2, 5, 5, strideX, strideY, 2, 2, "B") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix);
                /*
                case 2:
                    return useChannelSplit ? ChannelSplit(id, inputs, 3, 1, "Q1") + ChannelSplit(id, inputs, 3, 2, "Q2") + ChannelSplit(id, inputs, 3, 3, "Q3") +
                        DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, "B") + DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix) :
                        PartialDepthwiseConvolution(id, inputs, 1, 3, 3, 3, strideX, strideY, 1, 1, "A") + PartialDepthwiseConvolution(id, inputs, 2, 3, 5, 5, strideX, strideY, 2, 2, "B") +
                        PartialDepthwiseConvolution(id, inputs, 3, 3, 7, 7, strideX, strideY, 3, 3, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix);
                */
            default:
                return useChannelSplit ? ChannelSplit(id, inputs, 4, 1, "Q1") + ChannelSplit(id, inputs, 4, 2, "Q2") + ChannelSplit(id, inputs, 4, 3, "Q3") + ChannelSplit(id, inputs, 4, 4, "Q4") +
                    DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, "B") +
                    DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, "C") + DepthwiseConvolution(id, In("Q4CS", id), 1, 9, 9, strideX, strideY, 4, 4, "D") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix) :
                    PartialDepthwiseConvolution(id, inputs, 1, 4, 3, 3, strideX, strideY, 1, 1, "A") + PartialDepthwiseConvolution(id, inputs, 2, 4, 5, 5, strideX, strideY, 2, 2, "B") +
                    PartialDepthwiseConvolution(id, inputs, 3, 4, 7, 7, strideX, strideY, 3, 3, "C") + PartialDepthwiseConvolution(id, inputs, 4, 4, 9, 9, strideX, strideY, 4, 4, "D") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix);
            }
        }

        static std::string ChannelSplit(UInt id, std::string inputs, UInt groups, UInt part, std::string group = "", std::string prefix = "CS")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelSplit" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Group=" + std::to_string(part) + nwl + nwl;
        }

        static std::string ChannelShuffle(UInt id, std::string inputs, UInt groups = 2, std::string group = "", std::string prefix = "CSH")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelShuffle" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl + nwl;
        }

        static std::string Concat(UInt id, std::string inputs, std::string group = "", std::string prefix = "CC")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Concat" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string AvgPooling(UInt id, std::string input, std::string kernel = "3,3", std::string stride = "2,2", std::string pad = "1,1", std::string group = "", std::string prefix = "P")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=AvgPooling" + nwl +
                "Inputs=" + input + nwl +
                "Kernel=" + kernel + nwl +
                "Stride=" + stride + nwl +
                "Pad=" + pad + nwl + nwl;
        }

        static std::string GlobalAvgPooling(std::string input, std::string group = "", std::string prefix = "GAP")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=GlobalAvgPooling" + nwl +
                "Inputs=" + input + nwl + nwl;
        }

        static std::string Add(UInt id, std::string inputs, std::string group = "", std::string prefix = "A")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Add" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string ChannelMultiply(std::string inputs, std::string group = "", std::string prefix = "CM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=ChannelMultiply" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Dropout(UInt id, std::string inputs, std::string group = "", std::string prefix = "D")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Dropout" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Activation(std::string inputs, std::string activation = "Relu", std::string group = "", std::string prefix = "ACT")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Activation" + nwl +
                "Inputs=" + inputs + nwl +
                "Activation=" + activation + nwl + nwl;
        }

        static std::string Cost(std::string inputs, Datasets dataset, UInt channels, std::string cost = "CategoricalCrossEntropy", Float eps = 0.0f, std::string group = "", std::string prefix = "Cost")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Cost" + nwl +
                "Inputs=" + inputs + nwl +
                "Cost=" + cost + nwl +
                "LabelIndex=" + ((dataset == Datasets::cifar100 && channels == 100) ? "1" : "0") + nwl +
                "Channels=" + std::to_string(channels) + nwl +
                "Eps=" + std::to_string(eps) + nwl + nwl;
        }

        static std::vector<std::string> MBConv(UInt id, std::string inputs, UInt inputChannels, UInt outputChannels, UInt stride = 1, UInt expandRatio = 4ull, bool se = false, scripts::Activations activation = scripts::Activations::HardSwish)
        {
            auto blocks = std::vector<std::string>();
            auto hiddenDim = DIV8(inputChannels * expandRatio);
            auto identity = stride == 1ull && inputChannels == outputChannels;

            if (se)
            {
                auto group = In("SE", id + 1);
                blocks.push_back(
                    Convolution(id, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(id, In("C", id), activation) +
                    DepthwiseConvolution(id + 1, In("B", id), 1, 3, 3, stride, stride, 1, 1) +
                    BatchNormActivation(id + 1, In("DC", id + 1), activation) +
                    GlobalAvgPooling(In("B", id + 1), group) +
                    Convolution(1, group + "GAP", DIV8(hiddenDim / expandRatio), 1, 1, 1, 1, 0, 0, group) +
                    BatchNormActivation(1, group + "C1", activation == scripts::Activations::FRelu ? scripts::Activations::HardSwish : activation, group) +
                    Convolution(2, group + "B1", hiddenDim, 1, 1, 1, 1, 0, 0, group) +
                    BatchNormActivation(2, group + "C2", "HardLogistic", group) +
                    ChannelMultiply(In("B", id + 1) + "," + group + "B2", group) +
                    Convolution(id + 2, group + "CM", DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(id + 2, In("C", id + 2)));
            }
            else
            {
                blocks.push_back(
                    Convolution(id, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(id, In("C", id), activation) +
                    DepthwiseConvolution(id + 1, In("B", id), 1, 3, 3, stride, stride, 1, 1) +
                    BatchNormActivation(id + 1, In("DC", id + 1), activation) +
                    Convolution(id + 2, In("B", id + 1), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(id + 2, In("C", id + 2)));
            }

            if (identity)
                blocks.push_back(Add(id + 2, In("B", id + 2) + "," + inputs));

            return blocks;
        }

        static std::string Generate(const ScriptParameters p)
        {
            const auto userLocale = std::setlocale(LC_ALL, "C");

            auto net =
                "[" + p.GetName() + "]" + nwl +
                "Dataset=" + to_string(p.Dataset) + nwl +
                "Dim=" + std::to_string(p.C) + "," + std::to_string(p.H) + "," + std::to_string(p.W) + nwl +
                ((p.PadH > 0 || p.PadW > 0) ? (!p.MirrorPad ? "ZeroPad=" + std::to_string(p.PadH) + "," + std::to_string(p.PadW) + nwl : "MirrorPad=" + std::to_string(p.PadH) + "," + std::to_string(p.PadW) + nwl) : "") +
                ((p.PadH > 0 || p.PadW > 0) ? "RandomCrop=Yes" + nwl : "") +
                "WeightsFiller=" + to_string(p.WeightsFiller) + (ScaleVisible(p.WeightsFiller) ? "(" + std::to_string(p.WeightsScale) + ")" : "") + nwl +
                (p.WeightsLRM != 1 ? "WeightsLRM=" + std::to_string(p.WeightsLRM) + nwl : "") +
                (p.WeightsWDM != 1 ? "WeightsWDM=" + std::to_string(p.WeightsWDM) + nwl : "") +
                (p.HasBias ? "BiasesFiller=" + to_string(p.BiasesFiller) + (ScaleVisible(p.BiasesFiller) ? "(" + std::to_string(p.BiasesScale) + ")" : "") + nwl +
                (p.BiasesLRM != 1 ? "BiasesLRM=" + std::to_string(p.BiasesLRM) + nwl : "") +
                (p.BiasesWDM != 1 ? "BiasesWDM=" + std::to_string(p.BiasesWDM) + nwl : "") : "Biases=No" + nwl) +
                (p.DropoutVisible() ? "Dropout=" + std::to_string(p.Dropout) + nwl : "") +
                "Scaling=" + to_string(p.BatchNormScaling) + nwl +
                "Momentum=" + std::to_string(p.BatchNormMomentum) + nwl +
                "Eps=" + std::to_string(p.BatchNormEps) + nwl + nwl;

            auto blocks = std::vector<std::string>();

            switch (p.Script)
            {
            case Scripts::densenet:
            {
                auto channels = DIV8(p.GrowthRate);

                net += Convolution(1, "Input", channels, 3, 3, 1, 1, 1, 1);

                if (p.Bottleneck)
                {
                    blocks.push_back(
                        BatchNormActivation(1, "C1", p.Activation) +
                        Convolution(2, "B1", DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                        BatchNormActivation(2, "C2", p.Activation) +
                        Convolution(3, "B2", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? Dropout(3, "C3") + Concat(1, "C1,D3") : Concat(1, "C1,C3")));
                }
                else
                {
                    blocks.push_back(
                        BatchNormActivation(1, "C1", p.Activation) +
                        Convolution(2, "B1", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? Dropout(2, "C2") + Concat(1, "C1,D2") : Concat(1, "C1,C2")));
                }

                auto CC = 1ull;
                auto C = p.Bottleneck ? 4ull : 3ull;

                channels += DIV8(p.GrowthRate);

                for (auto g = 1ull; g <= p.Groups; g++)
                {
                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("CC", CC), p.Activation) +
                                Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C + 1)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C + 1))));

                            C += 2;
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("CC", CC), p.Activation) +
                                Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C))));

                            C++;
                        }

                        CC++;
                        channels += DIV8(p.GrowthRate);
                    }

                    if (g < p.Groups)
                    {
                        channels = DIV8((UInt)std::floor(2.0 * channels * p.Compression));

                        if (p.Dropout > 0)
                            blocks.push_back(
                                Convolution(C, In("CC", CC), channels, 1, 1, 1, 1, 0, 0) +
                                Dropout(C, In("C", C)) +
                                AvgPooling(g, In("D", C), "2,2", "2,2", "0,0"));
                        else
                            blocks.push_back(
                                Convolution(C, In("CC", CC), channels, 1, 1, 1, 1, 0, 0) +
                                AvgPooling(g, In("C", C), "2,2", "2,2", "0,0"));
                        C++;
                        CC++;

                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("P", g), p.Activation) +
                                Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC, In("B", C) + "," + In("D", C + 1)) : Concat(CC, In("B", C) + "," + In("C", C + 1))));

                            C += 2;
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("P", g), p.Activation) +
                                Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC, In("B", C) + "," + In("D", C)) : Concat(CC, In("B", C) + "," + In("C", C))));

                            C++;
                        }

                        channels += DIV8(p.GrowthRate);
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    Convolution(C, In("CC", CC), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    Activation("GAP", "LogSoftmax") +
                    Cost("ACT", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::efficientnetv2:
            {
                auto inputChannels = DIV8(p.EffNet[0].Channels);
                auto C = 1ull;
                
                net +=
                    Convolution(C, "Input", inputChannels, 3, 3, 1, 1, 1, 1) +
                    BatchNormActivation(C, In("C", C), p.Activation);

                C++;
                auto inp = std::string("B1");
                for (auto rec : p.EffNet)
                {
                    auto outputChannels = DIV8(rec.Channels);
                    for (auto n = 0ull; n < rec.Iterations; n++)
                    {
                        auto stride = n == 0ull ? rec.Stride : 1ull;
                        auto identity = stride == 1ull && inputChannels == outputChannels;

                        auto subblocks = MBConv(C, inp, inputChannels, outputChannels, stride, rec.ExpandRatio, rec.SE, p.Activation);
                        for(auto blk : subblocks)
                            net += blk;

                        inputChannels = outputChannels;
                        C += 2;
                        inp = In((identity ? "A" : "B"), C);
                        C++;
                    }
                }

                net +=
                    Convolution(C, In("A", C - 2), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), p.Activation) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    Activation("GAP", "LogSoftmax") +
                    Cost("ACT", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.1f);
            }
            break;

            case Scripts::mobilenetv3:
            {
                auto se = p.SqueezeExcitation;
                auto channelsplit = true;
                auto W = p.Width * 16;

                net +=
                    Convolution(1, "Input", DIV8(W), 3, 3, 1, 1, 1, 1) +
                    BatchNormActivation(1, "C1", p.Activation);

                blocks.push_back(
                    Convolution(2, "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(2, "C2", p.Activation) +
                    DepthwiseMixedConvolution(0, 3, "B2", 1, 1, channelsplit) +
                    BatchNormActivation(3, "DC3", p.Activation) +
                    Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(4, "C4"));

                auto A = 1ull;
                auto C = 5ull;

                for (auto g = 1ull; g <= p.Groups; g++)
                {
                    auto mix = 0ull; // g - 1ull;

                    if (g > 1)
                    {
                        W *= 2;

                        auto group = In("SE", C + 1);
                        auto strSE =
                            se ? GlobalAvgPooling(In("B", C + 1), group) +
                            Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(1, group + "C1", p.Activation == Activations::FRelu ? Activations::HardSwish : p.Activation, group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(2, group + "C2", "HardLogistic", group) +
                            ChannelMultiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, In("A", A), DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C, In("C", C), p.Activation) +
                            DepthwiseMixedConvolution(mix, C + 1, In("B", C), 2, 2, channelsplit) +
                            BatchNormActivation(C + 1, In("DC", C + 1), p.Activation) +
                            strSE +
                            BatchNorm(C + 2, In("C", C + 2)));

                        C += 3;
                    }

                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        auto strOutputLayer = (i == 1 && g > 1) ? In("B", C - 1) : (i == 1 && g == 1) ? "B4" : In("A", A);

                        auto group = In("SE", C + 1);

                        auto strSE =
                            se ? GlobalAvgPooling(In("B", C + 1), group) +
                            Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(1, group + "C1", p.Activation == Activations::FRelu ? Activations::HardSwish : p.Activation, group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(2, group + "C2", "HardLogistic", group) +
                            ChannelMultiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, strOutputLayer, DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C, In("C", C), p.Activation) +
                            DepthwiseMixedConvolution(mix, C + 1, In("B", C), 1, 1, channelsplit) +
                            BatchNormActivation(C + 1, In("DC", C + 1), p.Activation) +
                            strSE +
                            BatchNorm(C + 2, In("C", C + 2)) +
                            Add(A + 1, In("B", C + 2) + "," + strOutputLayer));

                        A++;
                        C += 3;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    BatchNormActivation(C, In("A", A), p.Activation) +
                    Convolution(C, In("B", C), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    Activation("GAP", "LogSoftmax") +
                    Cost("ACT", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::resnet:
            {
                auto bn = p.Bottleneck ? 1ull : 0ull;
                const Float K = 2;
                auto W = p.Width * 16;
                auto A = 1ull;
                auto C = 5ull;

                net += Convolution(1, "Input", DIV8(W), 3, 3, 1, 1, 1, 1);

                if (p.Bottleneck)
                {
                    blocks.push_back(
                        BatchNormActivation(1, "C1", p.Activation) +
                        Convolution(2, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        BatchNormActivation(2, "C2", p.Activation) +
                        Convolution(3, "B2", DIV8((UInt)(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? BatchNormActivationDropout(3, "C3", p.Activation) : BatchNormActivation(3, "C3", p.Activation)) +
                        Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Convolution(5, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Add(1, "C4,C5"));

                    C = 6;
                }
                else
                {
                    blocks.push_back(
                        BatchNormActivation(1, "C1", p.Activation) +
                        Convolution(2, "B1", DIV8(W), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? BatchNormActivationDropout(2, "C2", p.Activation) : BatchNormActivation(2, "C2", p.Activation)) +
                        Convolution(3, "B2", DIV8(W), 3, 3, 1, 1, 1, 1) +
                        Convolution(4, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Add(1, "C3,C4"));
                }

                for (auto g = 0ull; g < p.Groups; g++)
                {
                    if (g > 0)
                    {
                        W *= 2;

                        auto strChannelZeroPad = p.ChannelZeroPad ?
                            AvgPooling(g, In("A", A)) +
                            "[CZP" + std::to_string(g) + "]" + nwl + "Type=ChannelZeroPad" + nwl + "Inputs=" + In("P", g) + nwl + "Channels=" + std::to_string(W) + nwl + nwl +
                            Add(A + 1, In("C", C + 1 + bn) + "," + In("CZP", g)) :
                            AvgPooling(g, In("B", C)) +
                            Convolution(C + 2 + bn, In("P", g), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            Add(A + 1, In("C", C + 1 + bn) + "," + In("C", C + 2 + bn));

                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("A", A), p.Activation) +
                                Convolution(C, In("B", C), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 2, 2, 1, 1) +
                                (p.Dropout > 0 ? BatchNormActivationDropout(C + 2, In("C", C + 1), p.Activation) : BatchNormActivation(C + 2, In("C", C + 1), p.Activation)) +
                                Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                strChannelZeroPad);
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("A", A), p.Activation) +
                                Convolution(C, In("B", C), DIV8(W), 3, 3, 2, 2, 1, 1) +
                                (p.Dropout > 0 ? BatchNormActivationDropout(C + 1, In("C", C), p.Activation) : BatchNormActivation(C + 1, In("C", C), p.Activation)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                strChannelZeroPad);
                        }

                        A++;
                        C += p.ChannelZeroPad ? 2 + bn : 3 + bn;
                    }

                    for (auto i = 1u; i < p.Iterations; i++)
                    {
                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("A", A), p.Activation) +
                                Convolution(C, In("B", C), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                BatchNormActivation(C + 1, In("C", C), p.Activation) +
                                Convolution(C + 1, In("B", C + 1), DIV8((UInt)(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? BatchNormActivationDropout(C + 2, In("C", C + 1), p.Activation) : BatchNormActivation(C + 2, In("C", C + 1), p.Activation)) +
                                Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                Add(A + 1, In("C", C + 2) + "," + In("A", A)));
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormActivation(C, In("A", A), p.Activation) +
                                Convolution(C, In("B", C), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? BatchNormActivationDropout(C + 1, In("C", C), p.Activation) : BatchNormActivation(C + 1, In("C", C), p.Activation)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                Add(A + 1, In("C", C + 1) + "," + In("A", A)));
                        }

                        A++;
                        C += 2 + bn;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    BatchNormActivation(C, In("A", A), p.Activation) +
                    Convolution(C, In("B", C), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    Activation("GAP", "LogSoftmax") +
                    Cost("ACT", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::shufflenetv2:
            {
                auto se = false;
                auto W = p.Width * 16;
                auto kernel = 3ull;
                auto pad = 1ull;

                net += Convolution(1, "Input", DIV8(W), kernel, kernel, 1, 1, pad, pad);

                blocks.push_back(
                    BatchNormActivation(1, "C1", p.Activation) +
                    Convolution(2, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(2, "C2", p.Activation) +
                    DepthwiseConvolution(3, "B2", 1, kernel, kernel, 1, 1, pad, pad) +
                    BatchNorm(3, "DC3") +
                    Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(4, "C4", p.Activation) +
                    Convolution(5, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    Concat(1, "C5,B4"));

                auto C = 6ull;
                auto A = 1ull;

                for (auto g = 1ull; g <= p.Groups; g++)
                {
                    if (g > 1)
                    {
                        se = p.SqueezeExcitation;
                        W *= 2;

                        blocks.push_back(
                            Convolution(C, In("CC", A), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C + 1, In("C", C), p.Activation) +
                            DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 2, 2, pad, pad) +
                            BatchNorm(C + 2, In("DC", C + 1)) +
                            Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C + 3, In("C", C + 2), p.Activation) +
                            DepthwiseConvolution(C + 3, In("CC", A), 1, kernel, kernel, 2, 2, pad, pad) +
                            BatchNorm(C + 4, In("DC", C + 3)) +
                            Convolution(C + 4, In("B", C + 4), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C + 5, In("C", C + 4), p.Activation) +
                            Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3)));

                        A++; C += 5;
                    }

                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        auto group = In("SE", C + 3);
                        auto strSE =
                            se ? GlobalAvgPooling(In("B", C + 3), group) +
                            Convolution(1, group + "GAP", DIV8(W / 4), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(1, group + "C1", p.Activation == Activations::FRelu ? Activations::HardSwish : p.Activation, group) +
                            Convolution(2, group + "B1", DIV8(W), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormActivation(2, group + "C2", "HardLogistic", group) +
                            ChannelMultiply(In("B", C + 3) + "," + group + "B2", group) +
                            Concat(A + 1, In("LCS", A) + "," + group + "CM") :
                            Concat(A + 1, In("LCS", A) + "," + In("B", C + 3));

                        blocks.push_back(
                            ChannelShuffle(A, In("CC", A), 2) +
                            ChannelSplit(A, In("CSH", A), 2, 1, "L") + ChannelSplit(A, In("CSH", A), 2, 2, "R") +
                            Convolution(C, In("RCS", A), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C + 1, In("C", C), p.Activation) +
                            DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                            BatchNorm(C + 2, In("DC", C + 1)) +
                            Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C + 3, In("C", C + 2), p.Activation) +
                            strSE);

                        A++; C += 3;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    Convolution(C, In("CC", A), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    Activation("GAP", "LogSoftmax") +
                    Cost("ACT", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            default:
            {
                net = std::string("Model not implemented");
                break;
            }
            }

            std::setlocale(LC_ALL, userLocale);

            return net;
        }
    private:
        // Disallow creating an instance of this object
        ScriptsCatalog() {}
    };
}