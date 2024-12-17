#pragma once
#include <algorithm>
#include <cmath>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 255
#include "magic_enum/magic_enum.hpp"

namespace scripts
{
    typedef float Float;
    typedef std::size_t UInt;
    typedef unsigned char Byte;

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    const auto nwl = std::string("\r\n");
#elif defined(__APPLE__)
    const auto nwl = std::string("\r");
#else // assuming Linux
    const auto nwl = std::string("\n");
#endif
   
    enum class Scripts
    {
        augshufflenet = 0,
        densenet = 1,
        efficientnetv2 = 2,
        mobilenetv3 = 3,
        resnet = 4,
        shufflenetv2 = 5
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

    enum class FillerModes
    {
        Avg = 0,
        In = 1,
        Out = 2
    };

    enum class Activations
    {
        FRelu = 1,
        HardSwish = 10,
        HardSigmoid =  11,
        Sigmoid = 12,
        Mish = 16,
        Relu = 19,
        Swish = 25,
        TanhExp = 27
    };

    static auto StringToLower(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);
        return text;
    };

    constexpr auto GainVisible(const Fillers filler)
    {
        switch (filler)
        {
        case Fillers::HeNormal:
        case Fillers::HeUniform:
        case Fillers::LeCunNormal:
        case Fillers::LeCunUniform:
        case Fillers::XavierNormal:
        case Fillers::XavierUniform:
            return true;
        default:
            return false;
        }
    }

    constexpr auto ModeVisible(const Fillers filler)
    {
        switch (filler)
        {
        case Fillers::HeNormal:
        case Fillers::HeUniform:
        case Fillers::LeCunNormal:
        case Fillers::LeCunUniform:
            return true;
        default:
            return false;
        }
    }

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

    struct EfficientNetRecord
    {
        UInt ExpandRatio;
        UInt Channels;
        UInt Iterations;
        UInt Stride;
        bool SE;

        std::string to_string()
        {
            return "(" + std::to_string(ExpandRatio) + "-" + std::to_string(Channels) + "-" + std::to_string(Iterations) + "-" + std::to_string(Stride) + (SE ? "-se" : "") + ")";
        }
    };

    struct ShuffleNetRecord
    {
        UInt Iterations;
        UInt Kernel;
        UInt Pad;
        UInt Shuffle;
        bool SE;

        std::string to_string()
        {
            return "(" + std::to_string(Iterations) + "-" + std::to_string(Kernel) + "-" + std::to_string(Pad) + "-" + std::to_string(Shuffle) + (SE ? "-se" : "") + ")";
        }
    };

    struct ScriptParameters
    {
        // Model default parameters
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
        scripts::FillerModes WeightsFillerMode = FillerModes::In;
        Float WeightsGain = Float(1);
        Float WeightsScale = Float(0.05);
        Float WeightsLRM = Float(1);
        Float WeightsWDM = Float(1);
        bool HasBias = false;
        scripts::Fillers BiasesFiller = Fillers::Constant;
        scripts::FillerModes BiasesFillerMode = FillerModes::In;
        Float BiasesGain = Float(1);
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
        Float DepthDrop = Float(0.2);
        bool FixedDepthDrop = false;
        UInt StrideHFirstConv = 1;
        UInt StrideWFirstConv = 1;
        scripts::Activations Activation = Activations::HardSwish;
        std::vector<EfficientNetRecord> EfficientNet = { { 1, 24, 2, 1, false }, { 4, 48, 4, 2, false }, { 4, 64, 4, 2, false }, { 4, 128, 6, 2, true }, { 6, 160, 9, 1, true }, { 6, 256, 15, 2, true } };
        std::vector<ShuffleNetRecord> ShuffleNet = { { 7, 3, 1, 2, false }, { 7, 3, 1, 2, true }, { 7, 3, 1, 2, true } };

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
            default:
                return 0;
            }
        }

        bool WidthVisible() const { return Script == Scripts::mobilenetv3 || Script == Scripts::resnet || Script == Scripts::shufflenetv2 || Script == Scripts::augshufflenet; ; }
        bool GrowthRateVisible() const { return Script == Scripts::densenet; }
        bool DropoutVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet || Script == Scripts::efficientnetv2; }
        bool DepthDropVisible() const { return Script == Scripts::densenet || Script == Scripts::mobilenetv3 || Script == Scripts::resnet || Script == Scripts::efficientnetv2; }
        bool CompressionVisible() const { return Script == Scripts::densenet; }
        bool BottleneckVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet; }
        bool SqueezeExcitationVisible() const { return Script == Scripts::mobilenetv3; }
        bool ChannelZeroPadVisible() const { return Script == Scripts::resnet; }
        bool EfficientNetVisible() const { return Script == Scripts::efficientnetv2; }
        bool ShuffleNetVisible() const { return Script == Scripts::shufflenetv2 || Script == Scripts::augshufflenet; }

        auto GetName() const
        {
            const auto common = std::string(magic_enum::enum_name<Scripts>(Script)) + std::string("-") + std::to_string(Groups) + std::string("-") + std::to_string(Iterations) + std::string("-");

            switch (Script)
            {
            case Scripts::densenet:
                return common + std::to_string(GrowthRate) + (Dropout > 0 ? std::string("-dropout") : std::string("")) + (DepthDrop > 0 ? (FixedDepthDrop ? std::string("-fixeddepthdrop") : std::string("-depthdrop")) : std::string("")) + (Compression > 0 ? std::string("-compression") : std::string("")) + (Bottleneck ? std::string("-bottleneck") : std::string("")) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation)));
            case Scripts::efficientnetv2:
            {
                auto name = std::string(magic_enum::enum_name<Scripts>(Script)) + (DepthDrop > 0 ? (FixedDepthDrop ? std::string("-fixeddepthdrop-") : std::string("-depthdrop-")) : std::string(""));
                for (auto rec : EfficientNet)
                    name += rec.to_string();
                return name;
            }
            case Scripts::mobilenetv3:
                return common + std::to_string(Width) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation))) + (SqueezeExcitation ? std::string("-se") : std::string("")) + (DepthDrop > 0 ? (FixedDepthDrop ? std::string("-fixeddepthdrop") : std::string("-depthdrop")) : std::string(""));
            case Scripts::resnet:
                return common + std::to_string(Width) + (Dropout > 0 ? std::string("-dropout") : std::string("")) + (DepthDrop > 0 ? (FixedDepthDrop ? std::string("-fixeddepthdrop") : std::string("-depthdrop")) : std::string("")) + (Bottleneck ? std::string("-bottleneck") : std::string("")) + (ChannelZeroPad ? std::string("-channelzeropad") : std::string("")) + std::string("-") + StringToLower(std::string(magic_enum::enum_name<Activations>(Activation)));
            case Scripts::augshufflenet:
            case Scripts::shufflenetv2:
            {
                auto name = std::string(magic_enum::enum_name<Scripts>(Script)) + std::string("-") + std::to_string(Width);
                for (auto rec : ShuffleNet)
                    name += rec.to_string();
                return name;
            }
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

        static auto to_string(const FillerModes fillerMode)
        {
            return std::string(magic_enum::enum_name<FillerModes>(fillerMode));
        }

        static auto to_string(const Activations activation)
        {
            return std::string(magic_enum::enum_name<Activations>(activation));
        }

        static UInt DIV8(UInt channels)
        {
            if (channels % 8ull == 0ull)
                return channels;

            return ((channels / 8ull) + 1ull) * 8ull;
        }

        static UInt DIV16(UInt channels)
        {
            if (channels % 16ull == 0ull)
                return channels;

            return ((channels / 16ull) + 1ull) * 16ull;
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
            if (activation == "Relu")
                return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNormRelu" + nwl +
                    "Inputs=" + inputs + nwl + nwl;
            else
                
                return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNormActivation" + nwl +
                    "Inputs=" + inputs + nwl + 
                    "Activation=" + activation + nwl + nwl;
                

                /*return "[" + group + "BN" + std::to_string(id) + "]" + nwl +
                    "Type=BatchNorm" + nwl +
                    "Inputs=" + inputs + nwl + nwl +
                    "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=Activation" + nwl +
                    "Inputs=" + group + "BN" + std::to_string(id) + nwl +
                    "Activation=" + activation + nwl + nwl;*/
        }

        static std::string BatchNormActivation(UInt id, std::string inputs, scripts::Activations activation = scripts::Activations::Relu, std::string group = "", std::string prefix = "B")
        {
            if (activation != scripts::Activations::FRelu)
            {
                if (activation == scripts::Activations::Relu)
                    return 
                        "[" + group + prefix + std::to_string(id) + "]" + nwl +
                        "Type=BatchNormRelu" + nwl +
                        "Inputs=" + inputs + nwl + nwl;
                else
                    
                    return 
                        "[" + group + prefix + std::to_string(id) + "]" + nwl +
                        "Type=BatchNormActivation" + nwl +
                        "Inputs=" + inputs + nwl + 
                        "Activation=" + std::string(magic_enum::enum_name<scripts::Activations>(activation)) + nwl + nwl;
                    

                    /*return "[" + group + "BN" + std::to_string(id) + "]" + nwl +
                        "Type=BatchNorm" + nwl +
                        "Inputs=" + inputs + nwl + nwl +
                        "[" + group + prefix + std::to_string(id) + "]" + nwl +
                        "Type=Activation" + nwl +
                        "Inputs=" + group + "BN" + std::to_string(id) + nwl +
                        "Activation=" + std::string(magic_enum::enum_name<scripts::Activations>(activation)) + nwl + nwl;*/
            }
            else
            {
                return 
                    "[" + group + "B" + std::to_string(id) + "B1]" + nwl +
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
                return
                    "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNormActivationDropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    "Activation=" + std::string(magic_enum::enum_name<scripts::Activations>(activation)) + nwl +
                    (dropout > 0.0f ? "Dropout=" + std::to_string(dropout) + nwl + nwl : nwl);
            }
            else
            {
                return 
                    "[" + group + prefix + std::to_string(id) + "]" + nwl +
                    "Type=BatchNormActivationDropout" + nwl +
                    "Inputs=" + inputs + nwl +
                    "Activation=HardSwish" + nwl +
                    (dropout > 0.0f ? "Dropout=" + std::to_string(dropout) + nwl + nwl : nwl);
            }
        }

        static std::string Convolution(UInt id, std::string inputs, UInt channels, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, std::string group = "", std::string prefix = "C", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Convolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + std::to_string(channels) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string DepthwiseConvolution(UInt id, std::string inputs, UInt multiplier = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, std::string group = "", std::string prefix = "DC", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=DepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                (multiplier > 1 ? "Mulltiplier=" + std::to_string(multiplier) + nwl : "") +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string PartialDepthwiseConvolution(UInt id, std::string inputs, UInt part = 1, UInt groups = 1, UInt kernelX = 3, UInt kernelY = 3, UInt strideX = 1, UInt strideY = 1, UInt padX = 1, UInt padY = 1, bool biases = false, std::string group = "", std::string prefix = "DC", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=PartialDepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Group=" + std::to_string(part) + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : "") +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl : "") +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
        }

        static std::string DepthwiseMixedConvolution(UInt g, UInt id, std::string inputs, UInt strideX = 1, UInt strideY = 1, bool biases = false, bool useChannelSplit = true, std::string group = "", std::string prefix = "DC")
        {
            switch (g)
            {
            case 0:
                return DepthwiseConvolution(id, inputs, 1, 3, 3, strideX, strideY, 1, 1, biases, group, prefix);

            case 1:
                return useChannelSplit ? ChannelSplit(id, inputs, 2, 1, "Q1") + ChannelSplit(id, inputs, 2, 2, "Q2") +
                    DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix) :
                    PartialDepthwiseConvolution(id, inputs, 1, 2, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 2, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id), group, prefix);
                /*
                case 2:
                    return useChannelSplit ? ChannelSplit(id, inputs, 3, 1, "Q1") + ChannelSplit(id, inputs, 3, 2, "Q2") + ChannelSplit(id, inputs, 3, 3, "Q3") +
                        DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") + DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, biases, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix) :
                        PartialDepthwiseConvolution(id, inputs, 1, 3, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 3, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                        PartialDepthwiseConvolution(id, inputs, 3, 3, 7, 7, strideX, strideY, 3, 3, biases, "C") +
                        Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix);
                */
            default:
                return useChannelSplit ? ChannelSplit(id, inputs, 4, 1, "Q1") + ChannelSplit(id, inputs, 4, 2, "Q2") + ChannelSplit(id, inputs, 4, 3, "Q3") + ChannelSplit(id, inputs, 4, 4, "Q4") +
                    DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, biases, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                    DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, biases, "C") + DepthwiseConvolution(id, In("Q4CS", id), 1, 9, 9, strideX, strideY, 4, 4, biases, "D") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix) :
                    PartialDepthwiseConvolution(id, inputs, 1, 4, 3, 3, strideX, strideY, 1, 1, biases, "A") + PartialDepthwiseConvolution(id, inputs, 2, 4, 5, 5, strideX, strideY, 2, 2, biases, "B") +
                    PartialDepthwiseConvolution(id, inputs, 3, 4, 7, 7, strideX, strideY, 3, 3, biases, "C") + PartialDepthwiseConvolution(id, inputs, 4, 4, 9, 9, strideX, strideY, 4, 4, biases, "D") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id) + "," + In("DDC", id), group, prefix);
            }
        }

        static std::string ChannelSplitRatioLeft(UInt id, std::string inputs, Float ratio = 0.375f, std::string group = "", std::string prefix = "CSRL")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelSplitRatioLeft" + nwl +
                "Inputs=" + inputs + nwl +
                "Ratio=" + std::to_string(ratio) + nwl + nwl;
        }

        static std::string ChannelSplitRatioRight(UInt id, std::string inputs, Float ratio = 0.375f, std::string group = "", std::string prefix = "CSRR")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelSplitRatioRight" + nwl +
                "Inputs=" + inputs + nwl +
                "Ratio=" + std::to_string(ratio) + nwl + nwl;
        }

        static std::string ChannelSplit(UInt id, std::string inputs, UInt groups, UInt part, std::string group = "", std::string prefix = "CS")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelSplit" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Group=" + std::to_string(part) + nwl + nwl;
        }

        static std::string Shuffle(UInt id, std::string inputs, UInt groups = 2, std::string group = "", std::string prefix = "SH")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Shuffle" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl + nwl;
        }

        static std::string Concat(UInt id, std::string inputs, std::string group = "", std::string prefix = "CC")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Concat" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Dense(UInt id, std::string inputs, UInt channels, bool biases = false, std::string group = "", std::string prefix = "DS", std::string weightsFiller = "")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Dense" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + std::to_string(channels) + nwl +
                (biases ? "Biases=Yes" + nwl : "") +
                (weightsFiller != "" ? "WeightsFiller=" + weightsFiller + nwl + nwl : nwl);
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
        
        static std::string GlobalMaxPooling(std::string input, std::string group = "", std::string prefix = "GMP")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=GlobalMaxPooling" + nwl +
                "Inputs=" + input + nwl + nwl;
        }

        static std::string Resampling(UInt id, std::string inputs, std::string group = "", std::string prefix = "R")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Resampling" + nwl +
                "Inputs=" + inputs + nwl +
                "Factor=0.5,0.5" + nwl +
                "Algorithm=Linear" + nwl + nwl;
        }

        static std::string ReductionAvg(UInt id, std::string inputs, std::string group = "", std::string prefix = "RAVG")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Reduction" + nwl +
                "Inputs=" + inputs + nwl +
                "Operation=Avg" + nwl + nwl;
        }

        static std::string ReductionMax(UInt id, std::string inputs, std::string group = "", std::string prefix = "RMAX")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Reduction" + nwl +
                "Inputs=" + inputs + nwl +
                "Operation=Max" + nwl + nwl;
        }

        static std::string Add(UInt id, std::string inputs, std::string group = "", std::string prefix = "A")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Add" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Multiply(std::string inputs, std::string group = "", std::string prefix = "CM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Multiply" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Dropout(UInt id, std::string inputs, std::string group = "", std::string prefix = "D")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Dropout" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Softmax(UInt id, std::string inputs, std::string group = "", std::string prefix = "SM")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Softmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Softmax(std::string inputs, std::string group = "", std::string prefix = "SM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=Softmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string LogSoftmax(UInt id, std::string inputs, std::string group = "", std::string prefix = "LSM")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=LogSoftmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string LogSoftmax(std::string inputs, std::string group = "", std::string prefix = "LSM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=LogSoftmax" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static std::string Activation(UInt id, std::string inputs, std::string activation = "Relu", std::string group = "", std::string prefix = "ACT")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
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

        static std::vector<std::string> FusedMBConv(UInt A, UInt C, std::string inputs, UInt inputChannels, UInt outputChannels, UInt stride = 1, UInt expandRatio = 4, bool se = false, scripts::Activations activation = scripts::Activations::HardSwish)
        {
            auto blocks = std::vector<std::string>();
            auto hiddenDim = DIV8(inputChannels * expandRatio);
            auto identity = stride == 1ull && inputChannels == outputChannels;

            if (se)
            {
                auto group = In("SE", C);
                blocks.push_back(
                    Convolution(C, inputs, hiddenDim, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C, In("C", C), activation) : BatchNormActivation(C, In("C", C), activation)) +

                    GlobalAvgPooling(In("B", C), group) +
                    Convolution(1, group + std::string("GAP"), DIV8(hiddenDim / expandRatio), 1, 1, 1, 1, 0, 0, true, group) +
                    BatchNormActivation(1, group + std::string("C1"), to_string(activation == scripts::Activations::FRelu ? scripts::Activations::HardSwish : activation), group) +
                    Convolution(2, group + std::string("B1"), hiddenDim, 1, 1, 1, 1, 0, 0, true, group) +
                    BatchNormActivation(2, group + std::string("C2"), to_string(scripts::Activations::HardSigmoid), group) +
                    Multiply(In("B", C) + std::string(",") + group + std::string("B2"), group) +

                    Convolution(C + 1, group + std::string("CM"), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C + 1)));
            }
            else
            {
                blocks.push_back(
                    Convolution(C, inputs, hiddenDim, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C, In("C", C), activation) : BatchNormActivation(C, In("C", C), activation)) +
                    Convolution(C + 1, In("B", C + 1), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C + 1)));
            }
                
            if (identity)
            {
                blocks.push_back(
                    Add(A, In("B", C + 1) + "," + inputs));
            }

            return blocks;
        }

        static std::vector<std::string> MBConv(UInt A, UInt C, std::string inputs, UInt inputChannels, UInt outputChannels, UInt stride = 1, UInt expandRatio = 4, bool se = false, scripts::Activations activation = scripts::Activations::HardSwish)
        {
            auto blocks = std::vector<std::string>();
            auto hiddenDim = DIV8(inputChannels * expandRatio);
            auto identity = stride == 1ull && inputChannels == outputChannels;

            if (se)
            {
                auto group = In("SE", C + 1);
                blocks.push_back(
                    Convolution(C, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C), 1, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C + 1, In("DC", C + 1), activation) : BatchNormActivation(C + 1, In("DC", C + 1), activation)) +


                    GlobalAvgPooling(In("B", C + 1), group) +
                    Convolution(1, group + std::string("GAP"), DIV8(hiddenDim / expandRatio), 1, 1, 1, 1, 0, 0, true, group) +
                    BatchNormActivation(1, group + std::string("C1"), to_string(activation == scripts::Activations::FRelu ? scripts::Activations::HardSwish : activation), group) +
                    Convolution(2, group + std::string("B1"), hiddenDim, 1, 1, 1, 1, 0, 0, true, group) +
                    BatchNormActivation(2, group + std::string("C2"), to_string(scripts::Activations::HardSigmoid), group) +
                    Multiply(In("B", C + 1) + std::string(",") + group + std::string("B2"), group) +

                    Convolution(C + 2, group + std::string("CM"), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 2, In("C", C + 2)));
            }
            else
            {
                blocks.push_back(
                    Convolution(C, inputs, hiddenDim, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C), 1, 3, 3, stride, stride, 1, 1) +
                    (expandRatio > 1 ? BatchNormActivationDropout(C + 1, In("DC", C + 1), activation) : BatchNormActivation(C + 1, In("DC", C + 1), activation)) +
                    Convolution(C + 2, In("B", C + 1), DIV8(outputChannels), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 2, In("C", C + 2)));
            }

            if (identity)
            {
                blocks.push_back(
                    Add(A, In("D", C + 2) + "," + inputs));
            }

            return blocks;
        }

        static std::string InvertedResidual(UInt A, UInt C, UInt channels, UInt kernel = 3, UInt pad = 1, bool subsample = false, UInt shuffle = 2, bool se = false, scripts::Activations activation = scripts::Activations::HardSwish)
        {
            if (subsample)
            {
                return
                    Convolution(C, In("CC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 1, In("DC", C + 1)) +
                    BatchNorm(C + 2, In("R", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    DepthwiseConvolution(C + 3, In("CC", A), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 3, In("DC", C + 3)) +
                    BatchNorm(C + 4, In("R", C + 3)) +
                    Convolution(C + 4, In("B", C + 4), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 5, In("C", C + 4), activation) +
                    Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3));
            }
            else
            {
                auto groupCH = In("CHATT", C + 3); // Channel Attention
                auto groupSP = In("SPATT", C + 3); // Spatial Attention
                auto strSE = se ?
                    GlobalAvgPooling(In("B", C + 3), groupCH) +
                    Convolution(1, groupCH + "GAP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(1, groupCH + In("C", 1), activation, groupCH) +
                    GlobalMaxPooling(In("B", C + 3), groupCH) +
                    Convolution(2, groupCH + "GMP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(2, groupCH + In("C", 2), activation, groupCH) +
                    Add(1, In(groupCH + "B", 1) + "," + In(groupCH + "B", 2), groupCH) +
                    Convolution(3, groupCH + "A1", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(3, groupCH + In("C", 3), Activations::HardSigmoid, groupCH) +
                    Multiply(In("B", C + 3) + "," + In(groupCH + "B", 3), groupCH) +
                    ReductionAvg(1, groupCH + "CM", groupSP) +
                    ReductionMax(1, groupCH + "CM", groupSP) +
                    Concat(1, In(groupSP + "RAVG", 1) + "," + In(groupSP + "RMAX", 1), groupSP) +
                    Convolution(1, groupSP + In("CC", 1), 1, 7, 7, 1, 1, 3, 3, false, groupSP) +
                    BatchNormActivation(1, groupSP + In("C", 1), Activations::HardSigmoid, groupSP) +
                    Multiply(groupCH + "CM," + groupSP + In("B", 1), groupSP) +
                    Concat(A + 1, In("LCS", A) + "," + groupSP + "CM") :
                    Concat(A + 1, In("LCS", A) + "," + In("B", C + 3));

                return
                    Shuffle(A, In("CC", A), shuffle) +
                    ChannelSplit(A, In("SH", A), 2, 1, "L") + ChannelSplit(A, In("SH", A), 2, 2, "R") +
                    Convolution(C, In("RCS", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    BatchNorm(C + 2, In("DC", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    strSE;
            }
        }

        static std::string AugmentedInvertedResidual(UInt A, UInt C, UInt channels, UInt kernel = 3, UInt pad = 1, bool subsample = false, UInt shuffle = 2, bool se = false, scripts::Activations activation = scripts::Activations::HardSwish)
        {
            if (subsample)
            {
                return
                    Convolution(C, In("CC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 1, In("DC", C + 1)) +
                    BatchNorm(C + 2, In("R", C + 1)) +
                    Convolution(C + 2, In("B", C + 2), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    DepthwiseConvolution(C + 3, In("CC", A), 1, kernel, kernel, 1, 1, pad, pad) +
                    Resampling(C + 3, In("DC", C + 3)) +
                    BatchNorm(C + 4, In("R", C + 3)) +
                    Convolution(C + 4, In("B", C + 4), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 5, In("C", C + 4), activation) +
                    Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3));
            }
            else
            {
                auto groupCH = In("CHATT", C + 3); // Channel Attention
                auto groupSP = In("SPATT", C + 3); // Spatial Attention
                auto strSE = se ?
                    GlobalAvgPooling(In("B", C + 3), groupCH) +
                    Convolution(1, groupCH + "GAP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(1, groupCH + In("C", 1), activation, groupCH) +
                    GlobalMaxPooling(In("B", C + 3), groupCH) +
                    Convolution(2, groupCH + "GMP", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(2, groupCH + In("C", 2), activation, groupCH) +
                    Add(1, In(groupCH + "B", 1) + "," + In(groupCH + "B", 2), groupCH) +
                    Convolution(3, groupCH + "A1", DIV8(channels), 1, 1, 1, 1, 0, 0, false, groupCH) +
                    BatchNormActivation(3, groupCH + In("C", 3), Activations::HardSigmoid, groupCH) +
                    Multiply(In("B", C + 3) + "," + In(groupCH + "B", 3), groupCH) +
                    ReductionAvg(1, groupCH + "CM", groupSP) +
                    ReductionMax(1, groupCH + "CM", groupSP) +
                    Concat(1, In(groupSP + "RAVG", 1) + "," + In(groupSP + "RMAX", 1), groupSP) +
                    Convolution(1, groupSP + In("CC", 1), 1, 7, 7, 1, 1, 3, 3, false, groupSP) +
                    BatchNormActivation(1, groupSP + In("C", 1), Activations::HardSigmoid, groupSP) +
                    Multiply(groupCH + "CM," + groupSP + In("B", 1), groupSP) +
                    Concat(A + 1, In("LCC", A) + "," + groupSP + "CM") :
                    Concat(A + 1, In("LCC", A) + "," + In("B", C + 3));

                return
                    Shuffle(A, In("CC", A), shuffle) +
                    ChannelSplitRatioLeft(A, In("SH", A), 0.375f) + ChannelSplitRatioRight(A, In("SH", A), 0.375f) +
                    Convolution(C, In("CSRR", A), DIV8((UInt)((2 * channels) * 0.375f)), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 1, In("C", C), activation) +
                    DepthwiseConvolution(C + 1, In("B", C + 1), 1, kernel, kernel, 1, 1, pad, pad) +
                    BatchNorm(C + 2, In("DC", C + 1)) +
                    ChannelSplit(A, In("B", C + 2), 2, 1, "L1") + ChannelSplit(A, In("B", C + 2), 2, 2, "R1") +
                    ChannelSplit(A, In("CSRL", A), 2, 1, "L2") + ChannelSplit(A, In("CSRL", A), 2, 2, "R2") +
                    Concat(A, In("L1CS", A) + "," + In("L2CS", A), "L") +
                    Concat(A, In("R1CS", A) + "," + In("R2CS", A), "R") +
                    Convolution(C + 2, In("RCC", A), channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(C + 3, In("C", C + 2), activation) +
                    strSE;
            }
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
                "WeightsFiller=" + to_string(p.WeightsFiller) + (ModeVisible(p.WeightsFiller) ? "(" + to_string(p.WeightsFillerMode) + "," + std::to_string(p.WeightsGain) + ")" : "") + (!ModeVisible(p.WeightsFiller) && GainVisible(p.WeightsFiller) ? "(" + to_string(p.WeightsGain) + ")" : "") + (ScaleVisible(p.WeightsFiller) ? "(" + to_string(p.WeightsScale) + ")" : "") + nwl +
                (p.WeightsLRM != 1 ? "WeightsLRM=" + std::to_string(p.WeightsLRM) + nwl : "") +
                (p.WeightsWDM != 1 ? "WeightsWDM=" + std::to_string(p.WeightsWDM) + nwl : "") +
                (p.HasBias ? "BiasesFiller=" + to_string(p.BiasesFiller) + (ModeVisible(p.BiasesFiller) ? "(" + to_string(p.BiasesFillerMode) + "," + std::to_string(p.BiasesGain) + ")" : "") + (!ModeVisible(p.BiasesFiller) && GainVisible(p.BiasesFiller) ? "(" + to_string(p.BiasesGain) + ")" : "") + (ScaleVisible(p.BiasesFiller) ? "(" + std::to_string(p.BiasesScale) + "," + std::to_string(p.BiasesGain) + ")" : "") + nwl +
                (p.BiasesLRM != 1 ? "BiasesLRM=" + std::to_string(p.BiasesLRM) + nwl : "") +
                (p.BiasesWDM != 1 ? "BiasesWDM=" + std::to_string(p.BiasesWDM) + nwl : "") : "Biases=No" + nwl) +
                (p.DropoutVisible() ? "Dropout=" + std::to_string(p.Dropout) + nwl : "") +
                (p.DepthDropVisible() ? "DepthDrop=" + std::to_string(p.DepthDrop) + nwl : "") +
                (p.DepthDropVisible() ? "FixedDepthDrop=" + to_string(p.FixedDepthDrop) + nwl : "") +
                "Scaling=" + to_string(p.BatchNormScaling) + nwl +
                "Momentum=" + std::to_string(p.BatchNormMomentum) + nwl +
                "Eps=" + std::to_string(p.BatchNormEps) + nwl + nwl;

            auto blocks = std::vector<std::string>();

            switch (p.Script)
            {
            case Scripts::augshufflenet:
            {
                auto channels = DIV8(p.Width * 16);

                net +=
                    Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                    BatchNormActivation(1, "C1", p.Activation) +
                    Convolution(2, "B1", channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(2, "C2", p.Activation) +
                    DepthwiseConvolution(3, "B2", 1, 3, 3, 1, 1, 1, 1) +
                    BatchNorm(3, "DC3") +
                    Convolution(4, "B3", channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(4, "C4", p.Activation) +
                    Convolution(5, "B1", channels, 1, 1, 1, 1, 0, 0) +
                    Concat(1, "C5,B4");

                auto C = 6ull;
                auto A = 1ull;
                auto subsample = false;
                for (const auto& rec : p.ShuffleNet)
                {
                    if (subsample)
                    {
                        channels *= 2;
                        net += AugmentedInvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, true, rec.Shuffle, rec.SE, p.Activation);
                        C += 5;
                    }
                    for (auto n = 0ull; n < rec.Iterations; n++)
                    {
                        net += AugmentedInvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, false, rec.Shuffle, rec.SE, p.Activation);
                        C += 3;
                    }
                    subsample = true;
                }

                net +=
                    Convolution(C, In("CC", A), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::densenet:
            {
                auto channels = DIV8(p.GrowthRate);

                net += Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1);

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
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);

            }
            break;

            case Scripts::efficientnetv2:
            {
                const auto width = Float(1);
                auto inputChannels = DIV8(UInt(width * (Float)p.EfficientNet[0].Channels));
                auto A = 1ull;
                auto C = 1ull;
                
                net +=
                    Convolution(C, "Input", inputChannels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                    BatchNormActivation(C, In("C", C), p.Activation);

                auto stage = 0ull;
                auto input = In("B", C++);
                for (const auto& rec : p.EfficientNet)
                {
                    auto beginStage = stage < 3ul;
                    auto outputChannels = DIV8(UInt(width * (Float)rec.Channels));
                    for (auto n = 0ull; n < rec.Iterations; n++)
                    {
                        auto stride = n == 0ull ? rec.Stride : 1ull;
                        auto identity = stride == 1ull && inputChannels == outputChannels;

                        auto subblocks = beginStage ? FusedMBConv(A, C, input, inputChannels, outputChannels, stride, rec.ExpandRatio, rec.SE, p.Activation) :
                                                           MBConv(A, C, input, inputChannels, outputChannels, stride, rec.ExpandRatio, rec.SE, p.Activation);
                        for(auto blk : subblocks)
                            net += blk;

                        inputChannels = outputChannels;
                        C += beginStage ? 1ull : 2ull;

                        if (identity)
                        {
                            input = In("A", A++);
                            C++;
                        }
                        else
                            input = In("B", C++);
                    }
                    stage++;
                }

                net +=
                    Convolution(C, In("A", A - 1), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivationDropout(C, In("C", C), p.Activation) +
                    GlobalAvgPooling(In("B", C)) +
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::mobilenetv3:
            {
                auto se = p.SqueezeExcitation;
                auto channelsplit = true;
                auto W = p.Width * 16;

                net +=
                    Convolution(1, "Input", DIV8(W), 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                    BatchNormActivation(1, "C1", p.Activation);

                blocks.push_back(
                    Convolution(2, "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(2, "C2", p.Activation) +
                    DepthwiseMixedConvolution(0, 3, "B2", 1, 1, p.HasBias, channelsplit) +
                    BatchNormActivation(3, "DC3", p.Activation) +
                    Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(4, "C4"));

                auto A = 1ull;
                auto C = 5ull;

                for (auto g = 1ull; g <= p.Groups; g++)
                {
                    auto mix = g - 1ull;

                    if (g > 1)
                    {
                        W *= 2;

                        auto group = In("SE", C + 1);
                        auto strSE =
                            se ? GlobalAvgPooling(In("B", C + 1), group) +
                            Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, false, group) +
                            BatchNormActivation(1, group + "C1", to_string(p.Activation == Activations::FRelu ? Activations::HardSwish : p.Activation), group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, false, group) +
                            BatchNormActivation(2, group + "C2", "HardSigmoid", group) +
                            Multiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, In("A", A), DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C, In("C", C), p.Activation) +
                            DepthwiseMixedConvolution(mix, C + 1, In("B", C), 2, 2, p.HasBias, channelsplit) +
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
                            Convolution(1, group + "GAP", DIV8((6 * W) / 4), 1, 1, 1, 1, 0, 0, true, group) +
                            BatchNormActivation(1, group + "C1", to_string(p.Activation == Activations::FRelu ? Activations::HardSwish : p.Activation), group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, true, group) +
                            BatchNormActivation(2, group + "C2", "HardSigmoid", group) +
                            Multiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, strOutputLayer, DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormActivation(C, In("C", C), p.Activation) +
                            DepthwiseMixedConvolution(mix, C + 1, In("B", C), 1, 1, p.HasBias, channelsplit) +
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
                    Convolution(C, In("CC", C), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::resnet:
            {
                auto bn = p.Bottleneck ? 1ull : 0ull;
                const Float K = 2;
                auto W = p.Width * 16;
                auto A = 1ull;
                auto C = 5ull;

                net += Convolution(1, "Input", DIV8(W), 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1);

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
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);
            }
            break;

            case Scripts::shufflenetv2:
            {
                auto channels = DIV8(p.Width * 16);

                net +=
                    Convolution(1, "Input", channels, 3, 3, p.StrideHFirstConv, p.StrideWFirstConv, 1, 1) +
                    BatchNormActivation(1, "C1", p.Activation) +
                    Convolution(2, "B1", channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(2, "C2", p.Activation) +
                    DepthwiseConvolution(3, "B2", 1, 3, 3, 1, 1, 1, 1) +
                    BatchNorm(3, "DC3") +
                    Convolution(4, "B3", channels, 1, 1, 1, 1, 0, 0) +
                    BatchNormActivation(4, "C4", p.Activation) +
                    Convolution(5, "B1", channels, 1, 1, 1, 1, 0, 0) +
                    Concat(1, "C5,B4");

                auto C = 6ull;
                auto A = 1ull;
                auto subsample = false;
                for(const auto& rec : p.ShuffleNet)
                {
                    if (subsample)
                    {
                        channels *= 2;
                        net += InvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, true, rec.Shuffle, rec.SE, p.Activation);
                        C += 5;
                    }
                    for (auto n = 0ull; n < rec.Iterations; n++)
                    {
                        net += InvertedResidual(A++, C, channels, rec.Kernel, rec.Pad, false, rec.Shuffle, rec.SE, p.Activation);
                        C += 3;
                    }
                    subsample = true;
                }

                net +=
                    Convolution(C, In("CC", A), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    LogSoftmax("GAP") +
                    Cost("LSM", p.Dataset, p.Classes(), "CategoricalCrossEntropy", 0.125f);

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