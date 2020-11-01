#pragma once
#include "Model.h"

using namespace std;

namespace dnn
{
    enum class Scripts
    {
        densenet = 0,
        mobilenetv3 = 1,
        resnet = 2,
        shufflenetv2 = 3
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

    struct ScriptParameters
    {
        // Model default parameters
        Scripts Script;
        Datasets Dataset;
        size_t C;
        size_t D = 1;
        size_t H;
        size_t W;
        size_t PadD = 0;
        size_t PadH = 0;
        size_t PadW = 0;
        bool MirrorPad = false;
        bool MeanStdNormalization = true;
        Fillers WeightsFiller = Fillers::HeNormal;
        Float WeightsScale = Float(0.05);
        Float WeightsLRM = Float(1);
        Float WeightsWDM = Float(1);
        bool HasBias = false;
        Fillers BiasesFiller = Fillers::Constant;
        Float BiasesScale = Float(0);
        Float BiasesLRM = Float(1);
        Float BiasesWDM = Float(1);
        bool BatchNormScaling = false;
        Float BatchNormMomentum = Float(0.995);
        Float BatchNormEps = Float(0.0001);
        Float Alpha = Float(0);
        Float Beta = Float(0);

        size_t Classes() const
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

        bool RandomCrop() const { return PadH > 0 || PadW > 0; }

        // Model common parameters
        size_t Groups;
        size_t Iterations;
       
        size_t Depth() const {
            switch (Script)
            {
            case Scripts::densenet:
                return (Groups * Iterations * (Bottleneck ? 2 : 1)) + ((Groups - 1) * 2);
            case Scripts::mobilenetv3:
                return (Groups * Iterations * 3) + ((Groups - 1) * 2);
            case Scripts::resnet:
                return (Groups * Iterations * (Bottleneck ? 3 : 2)) + ((Groups - 1) * 2);
            case Scripts::shufflenetv2:
                return (Groups * (Iterations - 1) * 3) + (Groups * 5) + 1;
            default:
                return 0;
            }
        }

        // Model specific parameters
        size_t Width;
        size_t GrowthRate;
        Float Dropout;
        Float Compression;
        bool Bottleneck;
        bool SqueezeExcitation;
        bool ChannelZeroPad;
        
        bool WidthVisible() const { return Script == Scripts::mobilenetv3 || Script == Scripts::resnet || Script == Scripts::shufflenetv2; }
        bool GrowthRateVisible() const { return Script == Scripts::densenet; }
        bool DropoutVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet; }
        bool CompressionVisible() const { return Script == Scripts::densenet; }
        bool BottleneckVisible() const { return Script == Scripts::densenet || Script == Scripts::resnet; }
        bool SqueezeExcitationVisible() const { return Script == Scripts::mobilenetv3; }
        bool ChannelZeroPadVisible() const { return Script == Scripts::resnet; }

        auto GetName() const
        {
            auto common = std::string(magic_enum::enum_name<Scripts>(Script)) + "-" + std::to_string(H) + "x" + std::to_string(W) + "-" + std::to_string(Groups) + "-" + std::to_string(Iterations) + "-";
            
            switch (Script)
            {
            case Scripts::densenet:
                return common + std::to_string(GrowthRate) + (Dropout > 0 ? "-dropout" : "") + (Compression > 0 ? "-compression" : "") + (Bottleneck ? "-bottleneck" : "");
            case Scripts::mobilenetv3:
                return common + std::to_string(Width) + (SqueezeExcitation ? "-se" : "");
            case Scripts::resnet:
                return common + std::to_string(Width) + (Dropout > 0 ? "-dropout" : "") + (Bottleneck ? "-bottleneck" : "") + (ChannelZeroPad ? "-channelzeropad" : "");
            case Scripts::shufflenetv2:
                return common + std::to_string(Width);
            default:
                return common;
            }
        };
    };

    class ScriptsCatalog
    {
    public:
        static size_t DIV8(size_t channels)
        {
            if (channels % 8 == 0)
                return channels;

            return ((channels / 8) + 1) * 8;
        }

        static auto In(string prefix, size_t id)
        {
            return prefix + std::to_string(id);
        }

        static auto BatchNorm(size_t id, string inputs, string group = "", string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNorm" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto BatchNormRelu(size_t id, string inputs, string group = "", string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNormRelu" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto BatchNormReluDropout(size_t id, string inputs, string group = "", string prefix = "B")
        {
            auto description = "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNormReluDropout" + nwl +
                "Inputs=" + inputs + nwl + nwl;

            return description;
        }

        static auto BatchNormHardLogistic(size_t id, string inputs, string group = "", string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNormHardLogistic" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto BatchNormHardSwish(size_t id, string inputs, string group = "", string prefix = "B")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=BatchNormHardSwish" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto Convolution(size_t id, string inputs, size_t channels, size_t kernelX = 3, size_t kernelY = 3, size_t strideX = 1, size_t strideY = 1, size_t padX = 1, size_t padY = 1, string group = "", string prefix = "C")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Convolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Channels=" + std::to_string(channels) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : string("")) +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl + nwl : nwl);
        }

        static auto DepthwiseConvolution(size_t id, string inputs, size_t multiplier = 1, size_t kernelX = 3, size_t kernelY = 3, size_t strideX = 1, size_t strideY = 1, size_t padX = 1, size_t padY = 1, string group = "", string prefix = "DC")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=DepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                (multiplier > 1 ? "Multiplier=" + std::to_string(multiplier) + nwl : string("")) +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : string("")) +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl + nwl : nwl);
        }

        static auto PartialDepthwiseConvolution(size_t id, string inputs, size_t part = 1, size_t groups = 1, size_t kernelX = 3, size_t kernelY = 3, size_t strideX = 1, size_t strideY = 1, size_t padX = 1, size_t padY = 1, string group = "", string prefix = "DC")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=PartialDepthwiseConvolution" + nwl +
                "Inputs=" + inputs + nwl +
                "Group=" + std::to_string(part) + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Kernel=" + std::to_string(kernelX) + "," + std::to_string(kernelY) + nwl +
                (strideX != 1 || strideY != 1 ? "Stride=" + std::to_string(strideX) + "," + std::to_string(strideY) + nwl : string("")) +
                (padX != 0 || padY != 0 ? "Pad=" + std::to_string(padX) + "," + std::to_string(padY) + nwl + nwl : nwl);
        }

        static auto ChannelSplit(size_t id, string inputs, size_t groups, size_t part, string group = "", string prefix = "CS")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelSplit" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl +
                "Group=" + std::to_string(part) + nwl + nwl;
        }

        static auto Concat(size_t id, string inputs, string group = "", string prefix = "CC")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Concat" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto DepthwiseMixedConvolution(size_t g, size_t id, string inputs, size_t strideX = 1, size_t strideY = 1, bool useChannelSplit = true, string group = "", string prefix = "DC")
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

            case 2:
                return useChannelSplit ? ChannelSplit(id, inputs, 3, 1, "Q1") + ChannelSplit(id, inputs, 3, 2, "Q2") + ChannelSplit(id, inputs, 3, 3, "Q3") +
                    DepthwiseConvolution(id, In("Q1CS", id), 1, 3, 3, strideX, strideY, 1, 1, "A") + DepthwiseConvolution(id, In("Q2CS", id), 1, 5, 5, strideX, strideY, 2, 2, "B") + DepthwiseConvolution(id, In("Q3CS", id), 1, 7, 7, strideX, strideY, 3, 3, "C") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix) :
                    PartialDepthwiseConvolution(id, inputs, 1, 3, 3, 3, strideX, strideY, 1, 1, "A") + PartialDepthwiseConvolution(id, inputs, 2, 3, 5, 5, strideX, strideY, 2, 2, "B") +
                    PartialDepthwiseConvolution(id, inputs, 3, 3, 7, 7, strideX, strideY, 3, 3, "C") +
                    Concat(id, In("ADC", id) + "," + In("BDC", id) + "," + In("CDC", id), group, prefix);

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

        static auto ChannelShuffle(size_t id, string inputs, size_t groups = 2, string group = "", string prefix = "CSH")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=ChannelShuffle" + nwl +
                "Inputs=" + inputs + nwl +
                "Groups=" + std::to_string(groups) + nwl + nwl;
        }

        static auto GlobalAvgPooling(string input, string group = "", string prefix = "GAP")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=GlobalAvgPooling" + nwl +
                "Inputs=" + input + nwl + nwl;
        }

        static auto Dropout(size_t id, string inputs, string group = "", string prefix = "D")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Dropout" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto Add(size_t id, const string inputs, string group = "", string prefix = "A")
        {
            return "[" + group + prefix + std::to_string(id) + "]" + nwl +
                "Type=Add" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto ChannelMultiply(string inputs, string group = "", string prefix = "CM")
        {
            return "[" + group + prefix + "]" + nwl +
                "Type=ChannelMultiply" + nwl +
                "Inputs=" + inputs + nwl + nwl;
        }

        static auto to_string(const bool variable)
        {
            return variable ? string("Yes") : string("No");
        }

        static auto to_string(const Datasets dataset)
        {
            return string(magic_enum::enum_name<Datasets>(dataset));
        }

        static auto to_string(const Fillers filler)
        {
            return string(magic_enum::enum_name<Fillers>(filler));
        }

        static auto Generate(const ScriptParameters p)
        {
            const auto userLocale = std::setlocale(LC_ALL, "C");

            auto net =
                "[" + p.GetName() + "]" + nwl +
                "Dataset=" + to_string(p.Dataset) + nwl +
                "Dim=" + to_string(p.C) + "," + to_string(p.H) + "," + to_string(p.W) + nwl +
                ((p.PadH > 0 || p.PadW > 0) ? (!p.MirrorPad ? "ZeroPad=" + to_string(p.PadH) + "," + to_string(p.PadW) + nwl : "MirrorPad=" + to_string(p.PadH) + "," + to_string(p.PadW) + nwl) : "") +
                ((p.PadH > 0 || p.PadW > 0) ? "RandomCrop=Yes" + nwl : "") +
                "WeightsFiller=" + to_string(p.WeightsFiller) + (ScaleVisible(p.WeightsFiller) ? "(" + to_string(p.WeightsScale) + ")" : "") + nwl +
                (p.WeightsLRM != 1 ? "WeightsLRM=" + to_string(p.WeightsLRM) + nwl : "") +
                (p.WeightsWDM != 1 ? "WeightsWDM=" + to_string(p.WeightsWDM) + nwl : "") +
                (p.HasBias ? "BiasesFiller=" + to_string(p.BiasesFiller) + (ScaleVisible(p.BiasesFiller) ? "(" + to_string(p.BiasesScale) + ")" : "") + nwl +
                (p.BiasesLRM != 1 ? "BiasesLRM=" + to_string(p.BiasesLRM) + nwl : "") +
                (p.BiasesWDM != 1 ? "BiasesWDM=" + to_string(p.BiasesWDM) + nwl : "") : "Biases=No" + nwl) +
                (p.DropoutVisible() ? "Dropout=" + to_string(p.Dropout) + nwl : "") +
                "Scaling=" + to_string(p.BatchNormScaling) + nwl +
                "Momentum=" + to_string(p.BatchNormMomentum) + nwl +
                "Eps=" + to_string(p.BatchNormEps) + nwl + nwl;
            
            auto blocks = std::vector<string>();

            switch (p.Script)
            {
            case Scripts::densenet:
            {
                auto channels = DIV8(p.GrowthRate);
                
                net += Convolution(1, "Input", channels, 3, 3, 1, 1, 1, 1);

                if (p.Bottleneck)
                {
                    blocks.push_back(
                        BatchNormRelu(1, "C1") +
                        Convolution(2, "B1", DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                        BatchNormRelu(2, "C2") +
                        Convolution(3, "B2", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? Dropout(3, "C3") + Concat(1, "C1,D3") : Concat(1, "C1,C3")));
                }
                else
                {
                    blocks.push_back(
                        BatchNormRelu(1, "C1") +
                        Convolution(2, "B1", DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? Dropout(2, "C2") + Concat(1, "C1,D2") : Concat(1, "C1,C2")));
                }

                auto CC = 1ull;
                auto C = p.Bottleneck ? 4ull : 3ull;
                
                channels += p.GrowthRate;

                for (auto g = 1ull; g <= p.Groups; g++)  // 32*32 16*16 8*8 or 28*28 14*14 7*7
                {
                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("CC", CC)) +
                                Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                BatchNormRelu(C + 1, In("C", C)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C + 1)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C + 1))));

                            C += 2;
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("CC", CC)) +
                                Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC + 1, In("CC", CC) + "," + In("D", C)) : Concat(CC + 1, In("CC", CC) + "," + In("C", C))));

                            C++;
                        }

                        CC++;
                        channels += p.GrowthRate;
                    }

                    if (g < p.Groups)
                    {
                        channels = DIV8(static_cast<size_t>(std::floor(Float(2.0) * channels * p.Compression)));
                       
                        if (p.Dropout > 0)
                            blocks.push_back(
                                Convolution(C, In("CC", CC), channels, 1, 1, 1, 1, 0, 0) +
                                Dropout(C, In("C", C)) +
                                "[P" + to_string(g) + "]" + nwl + "Type=AvgPooling" + nwl + "Inputs=D" + to_string(C) + nwl + "Kernel=2,2" + nwl + "Stride=2,2" + nwl + nwl);
                        else
                            blocks.push_back(
                                Convolution(C, "CC" + to_string(CC), channels, 1, 1, 1, 1, 0, 0) +
                                "[P" + to_string(g) + "]" + nwl + "Type=AvgPooling" + nwl + "Inputs=C" + to_string(C) + nwl + "Kernel=2,2" + nwl + "Stride=2,2" + nwl + nwl);
                        C++;
                        CC++;

                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("P", g)) +
                                Convolution(C, In("B", C), DIV8(4 * p.GrowthRate), 1, 1, 1, 1, 0, 0) +
                                BatchNormRelu(C + 1, In("C", C)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C + 1, In("C", C + 1)) + Concat(CC, In("B", C) + "," + In("D", C + 1)) : Concat(CC, In("B", C) + "," + In("C", C + 1))));

                            C += 2;
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("P", g)) +
                                Convolution(C, In("B", C), DIV8(p.GrowthRate), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? Dropout(C, In("C", C)) + Concat(CC, In("B", C) + "," + In("D", C)) : Concat(CC, In("B", C) + "," + In("C", C))));

                            C++;
                        }

                        channels += p.GrowthRate;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    Convolution(C, In("CC", CC), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    "[ACT]" + nwl + "Type=Activation" + nwl + "Inputs=GAP" + nwl + "Activation=Softmax" + nwl + nwl +
                    "[Cost]" + nwl + "Type=Cost" + nwl + "Inputs=ACT" + nwl + "Cost=CategoricalCrossEntropy" + nwl + "Channels=" + to_string(p.Classes());
            }
            break;

            case Scripts::mobilenetv3:
            {
                const auto channelsplit = true;
                auto W = p.Width * 16;

                net += 
                    Convolution(1, "Input", DIV8(W), 3, 3, 1, 1, 1, 1) + 
                    BatchNormHardSwish(1, "C1");

                blocks.push_back(
                    Convolution(2, "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                    BatchNormHardSwish(2, "C2") +
                    DepthwiseMixedConvolution(3, 3, "B2", 1, 1, channelsplit) +
                    BatchNormHardSwish(3, "DC3") +
                    Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(4, "C4"));

                auto A = 1ull;
                auto C = 6ull;

                for (auto g = 1ull; g <= p.Groups; g++)  // 32*32 16*16 8*8 or 28*28 14*14 7*7
                {
                    if (g > 1)
                    {
                        W *= 2;

                        auto group = In("SE", C + 1);
                        auto strSE =
                            p.SqueezeExcitation ? GlobalAvgPooling(In("B", C + 1), group) +
                            Convolution(1, group + "GAP", DIV8(6 * W / 4), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormHardSwish(1, group + "C1", group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormHardLogistic(2, group + "C2", group) +
                            ChannelMultiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        //auto strDropout = p.Dropout > 0 ? Dropout(C, In("A", A)) +
                        //    Convolution(C, In("D", C), 6 * W, 1, 1, 1, 1, 0, 0) :
                        //    Convolution(C, In("A", A), 6 * W, 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, In("A", A), DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormHardSwish(C, In("C", C)) +
                            DepthwiseMixedConvolution(3, C + 1, In("B", C), 2, 2, channelsplit) +
                            BatchNormHardSwish(C + 1, In("DC", C + 1)) +
                            strSE +
                            BatchNorm(C + 2, In("C", C + 2)));

                        C += 3;
                    }

                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        auto strOutputLayer = (i == 1 && g > 1) ? In("B", C - 1) : (i == 1 && g == 1) ? "B4" : In("A", A);

                        auto group = In("SE", C + 1);

                        auto strSE =
                            p.SqueezeExcitation ? GlobalAvgPooling(In("B", C + 1), group) +
                            Convolution(1, group + "GAP", DIV8(6 * W / 4), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormHardSwish(1, group + "C1", group) +
                            Convolution(2, group + "B1", DIV8(6 * W), 1, 1, 1, 1, 0, 0, group) +
                            BatchNormHardLogistic(2, group + "C2", group) +
                            ChannelMultiply(In("B", C + 1) + "," + group + "B2", group) +
                            Convolution(C + 2, group + "CM", DIV8(W), 1, 1, 1, 1, 0, 0) :
                            Convolution(C + 2, In("B", C + 1), DIV8(W), 1, 1, 1, 1, 0, 0);

                        blocks.push_back(
                            Convolution(C, strOutputLayer, DIV8(6 * W), 1, 1, 1, 1, 0, 0) +
                            BatchNormHardSwish(C, In("C", C)) +
                            DepthwiseMixedConvolution(g, C + 1, In("B", C), 1, 1, channelsplit) +
                            BatchNormHardSwish(C + 1, In("DC", C + 1)) +
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
                    BatchNormHardSwish(C, In("A", A)) +
                    Convolution(C, In("B", C), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    "[ACT]" + nwl + "Type=Activation" + nwl + "Inputs=GAP" + nwl + "Activation=Softmax" + nwl + nwl +
                    "[Cost]" + nwl + "Type=Cost" + nwl + "Inputs=ACT" + nwl + "Cost=CategoricalCrossEntropy" + nwl + "Channels=" + to_string(p.Classes());
            }
            break;

            case Scripts::resnet:
            {
                const Float K = 2;
                const auto bn = p.Bottleneck ? 1ull : 0ull;
                auto W = p.Width * 16;
                auto A = 1ull;
                auto C = 5ull;

                net += Convolution(1, "Input", DIV8(W), 3, 3, 1, 1, 1, 1);

                if (p.Bottleneck)
                {
                    blocks.push_back(
                        BatchNormRelu(1, "C1") +
                        Convolution(2, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        BatchNormRelu(2, "C2") +
                        Convolution(3, "B2", DIV8(static_cast<size_t>(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? BatchNormReluDropout(3, "C3") : BatchNormRelu(3, "C3")) +
                        Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Convolution(5, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Add(1, "C4,C5"));

                    C = 6;
                }
                else
                {
                    blocks.push_back(
                        BatchNormRelu(1, "C1") +
                        Convolution(2, "B1", DIV8(W), 3, 3, 1, 1, 1, 1) +
                        (p.Dropout > 0 ? BatchNormReluDropout(2, "C2") : BatchNormRelu(2, "C2")) +
                        Convolution(3, "B2", DIV8(W), 3, 3, 1, 1, 1, 1) +
                        Convolution(4, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                        Add(1, "C3,C4"));
                }

                for (auto g = 0ull; g < p.Groups; g++)  // 32*32 16*16 8*8 or 28*28 14*14 7*7
                {
                    if (g > 0)
                    {
                        W *= 2;

                        auto strChannelZeroPad = p.ChannelZeroPad ?
                            ("[AVG" + to_string(g) + "]" + nwl + "Type=AvgPooling" + nwl + "Inputs=A" + to_string(A) + nwl + "Kernel=3,3" + nwl + "Stride=2,2" + nwl + "Pad=1,1" + nwl + nwl +
                            "[CZP" + to_string(g) + "]" + nwl + "Type=ChannelZeroPad" + nwl + "Inputs=AVG" + to_string(g) + nwl + "Channels=" + to_string(W) + nwl + nwl +
                            Add(A + 1, In("C", C + 1 + bn) + "," + In("CZP", g))) :
                            (Convolution(C + 2 + bn, In("B", C), DIV8(W), 1, 1, 2, 2, 0, 0) +
                            Add(A + 1, In("C", C + 1 + bn) + "," + In("C", C + 2 + bn)));

                        if (p.Bottleneck)
                        {

                            blocks.push_back(
                                BatchNormRelu(C, In("A", A)) +
                                Convolution(C, In("B", C), DIV8(W), 1, 1, 2, 2, 0, 0) +
                                BatchNormRelu(C + 1, In("C", C)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(static_cast<size_t>(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? BatchNormReluDropout(C + 2, In("C", C + 1)) : BatchNormRelu(C + 2, In("C", C + 1))) +
                                Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                strChannelZeroPad);
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("A", A)) +
                                Convolution(C, In("B", C), DIV8(W), 3, 3, 2, 2, 1, 1) +
                                (p.Dropout > 0 ? BatchNormReluDropout(C + 1, In("C", C)) : BatchNormRelu(C + 1, In("C", C))) +
                                Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                strChannelZeroPad);
                        }

                        A++;
                        if (p.ChannelZeroPad)
                            C += 2 + bn;
                        else
                            C += 3 + bn;
                    }

                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        if (p.Bottleneck)
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("A", A)) +
                                Convolution(C, In("B", C), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                BatchNormRelu(C + 1, In("C", C)) +
                                Convolution(C + 1, In("B", C + 1), DIV8(static_cast<size_t>(K * W / 4)), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? BatchNormReluDropout(C + 2, In("C", C + 1)) : BatchNormRelu(C + 2, In("C", C + 1))) +
                                Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                                Add(A + 1, In("C", C + 2) + "," + In("A", A)));
                        }
                        else
                        {
                            blocks.push_back(
                                BatchNormRelu(C, In("A", A)) +
                                Convolution(C, In("B", C), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                (p.Dropout > 0 ? BatchNormReluDropout(C + 1, In("C", C)) : BatchNormRelu(C + 1, In("C", C))) +
                                Convolution(C + 1, In("B", C + 1), DIV8(W), 3, 3, 1, 1, 1, 1) +
                                Add(A + 1, In("C", C + 1) + "," + In("A", A)));
                        }

                        A++; C += 2 + bn;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    BatchNormRelu(C, In("A", A)) +
                    Convolution(C, In("B", C), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    "[ACT]" + nwl + "Type=Activation" + nwl + "Inputs=GAP" + nwl + "Activation=Softmax" + nwl + nwl +
                    "[Cost]" + nwl + "Type=Cost" + nwl + "Inputs=ACT" + nwl + "Cost=CategoricalCrossEntropy" + nwl + "Channels=" + to_string(p.Classes());
            }
            break;

            case Scripts::shufflenetv2:
            {
                const auto channelsplit = true;
                auto W = p.Width * 16;

                net += Convolution(1, "Input", DIV8(W), 3, 3, 1, 1, 1, 1);

                blocks.push_back(
                    BatchNormRelu(1, "C1") +
                    Convolution(2, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNormRelu(2, "C2") +
                    DepthwiseConvolution(3, "B2") +
                    BatchNorm(3, "DC3") +
                    Convolution(4, "B3", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    BatchNormRelu(4, "C4") +
                    Convolution(5, "B1", DIV8(W), 1, 1, 1, 1, 0, 0) +
                    Concat(1, "C5,B4"));
                
                auto C = 6ull;
                auto A = 1ull;

                for (auto g = 1ull; g <= p.Groups; g++)  // 32*32 16*16 8*8 or 28*28 14*14 7*7
                {
                    if (g > 1)
                    {
                        W *= 2;
                        
                        blocks.push_back(
                            Convolution(C, In("CC", A), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormRelu(C + 1, In("C", C)) +
                            DepthwiseMixedConvolution(0, C + 1, In("B", C + 1), 2, 2, channelsplit) +
                            BatchNorm(C + 2, In("DC", C + 1)) +
                            Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormRelu(C + 3, In("C", C + 2)) +
                            DepthwiseMixedConvolution(0, C + 3, In("CC", A), 2, 2, channelsplit) +
                            BatchNorm(C + 4, In("DC", C + 3)) +
                            Convolution(C + 4, In("B", C + 4), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormRelu(C + 5, In("C", C + 4)) +
                            Concat(A + 1, In("B", C + 5) + "," + In("B", C + 3)));

                        A++; C += 5;
                    }

                    for (auto i = 1ull; i < p.Iterations; i++)
                    {
                        blocks.push_back(
                            ChannelShuffle(A, In("CC", A), 2) +
                            ChannelSplit(A, In("CSH", A), 2, 1, "L") + ChannelSplit(A, In("CSH", A), 2, 2, "R") +
                            Convolution(C, In("RCS", A), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormRelu(C + 1, In("C", C)) +
                            DepthwiseMixedConvolution(0, C + 1, In("B", C + 1), 1, 1, channelsplit) +
                            BatchNorm(C + 2, In("DC", C + 1)) +
                            Convolution(C + 2, In("B", C + 2), DIV8(W), 1, 1, 1, 1, 0, 0) +
                            BatchNormRelu(C + 3, In("C", C + 2)) +
                            Concat(A + 1, In("LCS", A) + "," + In("B", C + 3)));

                        A++; C += 3;
                    }
                }

                for (auto block : blocks)
                    net += block;

                net +=
                    Convolution(C, In("CC", A), p.Classes(), 1, 1, 1, 1, 0, 0) +
                    BatchNorm(C + 1, In("C", C)) +
                    GlobalAvgPooling(In("B", C + 1)) +
                    "[ACT]" + nwl + "Type=Activation" + nwl + "Inputs=GAP" + nwl + "Activation=Softmax" + nwl + nwl +
                    "[Cost]" + nwl + "Type=Cost" + nwl + "Inputs=ACT" + nwl + "Cost=CategoricalCrossEntropy" + nwl + "Channels=" + to_string(p.Classes());
            }
            break;

            default:
            {
                net = string("Model not implemented");
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