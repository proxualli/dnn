#pragma once
#include "Model.h"

namespace dnn
{
	const std::string NormalizeDefinition(const std::string& definition)
	{
		auto defNorm = std::string(definition);

		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), tab, "");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), " ", "");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl + nwl + nwl + nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl + nwl + nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl + nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), nwl + nwl, nwl);
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "[", nwl + "[");

		defNorm = Trim(defNorm);

		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "=Yes", "=Yes");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "=No", "=No");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "=True", "=True");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "=False", "=False");

		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Inputs=", "Inputs=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "WeightsScale=", "WeightsScale=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "WeightsLRM=", "WeightsLRM=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "WeightsWDM=", "WeightsWDM=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "BiasesScale=", "BiasesScale=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "BiasesLRM=", "BiasesLRM=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "BiasesWDM=", "BiasesWDM=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Biases=", "Biases=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Momentum=", "Momentum=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Scaling=", "Scaling=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Eps=", "Eps=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Dim=", "Dim=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "MeanStd=", "MeanStd=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "ZeroPad=", "ZeroPad=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "MirrorPad=", "MirrorPad=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "RandomCrop=", "RandomCrop=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Dropout=", "Dropout=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "DepthDrop=", "DepthDrop=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "FixedDepthDrop=", "FixedDepthDrop=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Channels=", "Channels=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Kernel=", "Kernel=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Stride=", "Stride=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Dilation=", "Dilation=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Pad=", "Pad=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Alpha=", "Alpha=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Beta=", "Beta=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Factor=", "Factor=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Groups=", "Groups=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Group=", "Group=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Multiplier=", "Multiplier=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "AcrossChannel=", "AcrossChannel=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "LocalSize=", "LocalSize=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "K=", "K=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "CostIndex=", "CostIndex=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "GroupIndex=", "GroupIndex=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "LabelIndex=", "LabelIndex=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "LabelTrue=", "LabelTrue=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "LabelFalse=", "LabelFalse=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Weight=", "Weight=");
		defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), "Ratio=", "Ratio=");

		auto types = magic_enum::enum_names<LayerTypes>();
		for (const auto& type : types)
		{
			auto text = "Type=" + std::string(type);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
		}

		auto activations = magic_enum::enum_names<Activations>();
		for (const auto& activation : activations)
		{
			auto text = "Activation=" + std::string(activation);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
		}

		auto costs = magic_enum::enum_names<Costs>();
		for (const auto& cost : costs)
		{
			auto text = "Cost=" + std::string(cost);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
		}

		auto fillers = magic_enum::enum_names<Fillers>();
		for (const auto& filler : fillers)
		{
			auto textFillerWeights = "WeightsFiller=" + std::string(filler);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerWeights, textFillerWeights);

			auto textFillerBiases = "BiasesFiller=" + std::string(filler);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerBiases, textFillerBiases);
		}

		auto fillerModes = magic_enum::enum_names<FillerModes>();
		for (const auto& fillerMode : fillerModes)
		{
			auto textFillerMode = "(" + std::string(fillerMode) + ")";
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerMode, textFillerMode);
		}

		auto datasets = magic_enum::enum_names<Datasets>();
		for (const auto& dataset : datasets)
		{
			auto text = "Dataset=" + std::string(dataset);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
		}

		auto algorithms = magic_enum::enum_names<Algorithms>();
		for (const auto& algorithm : algorithms)
		{
			auto text = "Algorithm=" + std::string(algorithm);
			defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
		}

		return defNorm;
	}

	Model* Parse(const std::string& definition, CheckMsg& msg, const bool onlyCheck = false, Dataprovider* dataprovider = nullptr)
	{
		auto loc = std::locale::global(std::locale::classic());
		auto model = static_cast<Model*>(nullptr);
		auto dataset = Datasets::cifar10;
		auto classes = UInt(10);
		auto c = UInt(0);
		auto d = UInt(1);
		auto h = UInt(0);
		auto w = UInt(0);
		auto padD = UInt(0);
		auto padH = UInt(0);
		auto padW = UInt(0);
		auto inputsStr = std::vector<std::string>();
		auto layerType = LayerTypes::Input;
		auto isNormalizationLayer = false;
		auto scaling = true;
		auto momentum = Float(0.995);
		auto eps = Float(1E-04);
		auto epsSpecified = false;
		auto useDefaultParams = true;
		auto weightsFiller = Fillers::HeNormal;
		auto weightsFillerMode = FillerModes::In;
		auto defaultWeightsGain = Float(1);
		auto weightsGain = Float(1);
		auto defaultWeightsScale = Float(0.05);
		auto weightsScale = Float(0.05);
		auto weightsLRM = Float(1);
		auto weightsWDM = Float(1);
		auto biasesFiller = Fillers::Constant;
		auto biasesFillerMode = FillerModes::In;
		auto defaultBiasesGain = Float(1);
		auto biasesGain = Float(1);
		auto defaultBiasesScale = Float(0);
		auto biasesScale = Float(0);
		auto biasesLRM = Float(1);
		auto biasesWDM = Float(1);
		auto biases = true;
		auto dropout = Float(0);
		auto alpha = Float(0);
		auto beta = Float(0);
		auto acrossChannels = false;
		auto localSize = UInt(5);
		auto k = Float(1);
		auto multiplier = UInt(1);
		auto group = UInt(1);
		auto groups = UInt(1);
		auto ratio = Float(0.375);
		auto factorH = Float(1);
		auto factorW = Float(1);
		auto algorithm = Algorithms::Linear;
		auto groupIndex = UInt(0);
		auto labelIndex = UInt(0);
		auto weight = Float(1);
		auto labelTrue = Float(0.9);
		auto labelFalse = Float(0.1);
		auto costFunction = Costs::CategoricalCrossEntropy;
		auto activationFunction = Activations::Linear;
		auto kernelH = UInt(1);
		auto kernelW = UInt(1);
		auto dilationH = UInt(1);
		auto dilationW = UInt(1);
		auto strideH = UInt(1);
		auto strideW = UInt(1);
		auto depthDrop = Float(0);
		auto fixedDepthDrop = false;
		auto reduceOp = ReduceOperations::Avg;
		auto reduceP = Float(0);
		auto reduceEps = Float(0);

		auto iss = std::istringstream(definition);
		auto strLine = std::string(""), modelName = std::string(""), layerName = std::string(""), params = std::string("");
		auto layerNames = std::vector<std::pair<std::string, UInt>>();
		auto line = 0ull, col = 0ull, modelMandatory = 0ull, layerMandatory = 0ull;
		auto isModel = true;
			
		while (SafeGetline(iss, strLine))
		{
			line++;
				
			if (strLine == std::string(""))
				continue;

			col = strLine.find_first_not_of("[],.=()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
			if (col != std::string::npos)
			{
				col++;
				msg =  CheckMsg(line, col, std::string("Line contains illegal characters.").c_str());
				goto FAIL;
			}
			col = strLine.length() + 1;

			if (strLine[0] == '[' && strLine[strLine.length() - 1] == ']') 
			{
				layerName = strLine.erase(strLine.length() - 1, 1).erase(0, 1);

				col = layerName.find_first_not_of("()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
				if (col != std::string::npos)
				{
					col++;
					msg = CheckMsg(line, col, std::string("Model or layer name contains illegal characters."));
					goto FAIL;
				}
				col = strLine.length() + 1;

				if (isModel)
				{
					if (modelName.empty())
					{
						modelName = layerName;
						model = new Model(definition, dataprovider);
							
						layerNames.push_back(std::make_pair(std::string("Input"), line));
					}
					else
					{
						if (modelMandatory != 129)
						{
							msg = CheckMsg(line, col, std::string("Model doesn't have the Dataset and Dim specifiers."));
							goto FAIL;
						}

						model->WeightsFiller = weightsFiller;
						model->WeightsFillerMode = weightsFillerMode;
						model->WeightsGain = weightsGain;
						model->WeightsScale = weightsScale;
						model->WeightsLRM = weightsLRM;
						model->WeightsWDM = weightsWDM;
						model->BiasesFiller = biasesFiller;
						model->BiasesFillerMode = biasesFillerMode;
						model->BiasesGain = biasesGain;
						model->BiasesScale = biasesScale;
						model->BiasesLRM = biasesLRM;
						model->BiasesWDM = biasesWDM;
						model->AlphaFiller = alpha;
						model->BetaFiller = beta;
						model->HasBias = biases;
						model->BatchNormMomentum = momentum;
						model->BatchNormScaling = scaling;
						model->BatchNormEps = eps;
						model->Dropout = dropout;
						model->DepthDrop = depthDrop;
						model->FixedDepthDrop = fixedDepthDrop;
						model->Ratio = ratio;

						model->Layers.push_back(std::make_unique<Input>(model->Device, model->Format, std::string("Input"), c, model->RandomCrop ? d : d + padD, model->RandomCrop ? h : h + padH, model->RandomCrop ? w : w + padW));

						isModel = false;

						auto exists = false;
						for (auto& layer : layerNames)
							if (layer.first == layerName)
								exists = true;

						if (exists)
						{
							msg = CheckMsg(line, col, std::string("Name already in use, must be unique."));
							goto FAIL;
						}

						layerNames.push_back(std::make_pair(layerName, line));
					}
				}
				else
				{
					if (layerMandatory != 129)
					{
						msg = CheckMsg(line, col, std::string("Layer doesn't have Type and Inputs specifiers."));
						goto FAIL;
					}

					auto exists = false;
					for (auto& layer : layerNames)
						if (layer.first == layerName)
							exists = true;

					if (exists)
					{
						msg = CheckMsg(line, col, std::string("Layer name already in use, must be unique."));
						goto FAIL;
					}

					layerNames.push_back(std::make_pair(layerName, line));

					layerMandatory = 0;

					if (layerType == LayerTypes::Activation)
					{
						switch(activationFunction)
						{
						case Activations::BoundedRelu:
						case Activations::Elu:
						case Activations::HardSigmoid:
						case Activations::HardSwish:
						case Activations::Linear:
						case Activations::Swish:
							break;

						case Activations::Abs:
						case Activations::ASinh:
						case Activations::Exp:
						case Activations::GeluErf:
						case Activations::GeluTanh:
						case Activations::Log:
						case Activations::LogSigmoid:
						case Activations::Mish:
						case Activations::Pow:
						case Activations::Round:
						case Activations::Selu:
						case Activations::Sigmoid:
						case Activations::SoftPlus:
						case Activations::SoftRelu:
						case Activations::SoftSign:
						case Activations::Sqrt:
						case Activations::Square:
						case Activations::Tanh:
						case Activations::TanhExp:
							if (alpha != 0 || beta != 0)
							{
								msg = CheckMsg(line - 1, col, std::string("This Activation cannot have an Alpha or Beta parameter."));
								goto FAIL;
							}
							break;

						case Activations::Clip:
						case Activations::ClipV2:
							if (alpha == 0 && beta == 0)
							{
								msg = CheckMsg(line - 1, col, std::string("Activation used without Alpha and Beta parameter."));
								goto FAIL;
							}
							break;
															
						case Activations::Relu:
							if (alpha < 0)
							{
								msg = CheckMsg(line - 1, col, std::string("This Activation Alpha parameter must be positive."));
								goto FAIL;
							}
							if (beta != 0)
							{
								msg = CheckMsg(line - 1, col, std::string("This Activation doesn't have a Beta parameter."));
								goto FAIL;
							}
							break;

						}
					}

					if (layerType == LayerTypes::Convolution || layerType == LayerTypes::DepthwiseConvolution || layerType == LayerTypes::ConvolutionTranspose || layerType == LayerTypes::PartialDepthwiseConvolution || layerType == LayerTypes::MaxPooling || layerType == LayerTypes::AvgPooling)
					{
						auto kerH = 1 + ((int)kernelH - 1) * (int)dilationH;
						auto kerW = 1 + ((int)kernelW - 1) * (int)dilationW;
						auto y = ((h - kerH + 1) + (2 * padH)) / (double)strideH;
						auto x = ((w - kerW + 1) + (2 * padW)) / (double)strideW;

						auto ok = true;
						if (x - std::floor(x) != 0.0)
							ok = false;
						if (y - std::floor(y) != 0.0)
							ok = false;
						if (x != y)
							ok = false;
						if (dilationH < 1 || dilationW < 1)
							ok = false;
						if ((dilationH > 1 || dilationW > 1) && (strideH != 1 || strideW != 1))
							ok = false;
						if (!ok)
						{
							msg = CheckMsg(line - 1, col, std::string("Kernel, Stride, Dilation or Pad invalid in layer ") + layerNames[model->Layers.size()].first);
							goto FAIL;
						}
					}

					if (layerType == LayerTypes::Resampling)
					{
						auto y = h * (double)factorW;
						auto x = w * (double)factorH;

						auto ok = true;
						if (x - std::floor(x) != 0.0)
							ok = false;
						if (y - std::floor(y) != 0.0)
							ok = false;
						if (x != y)
							ok = false;

						if (!ok)
						{
							msg = CheckMsg(line - 1, col, std::string("Factor invalid in Resampling layer ") + layerNames[model->Layers.size()].first);
							goto FAIL;
						}
					}

					if (layerType == LayerTypes::Cost)
					{
						if (c != classes)
						{
							msg = CheckMsg(line - 1, col, std::string("Cost layers hasn't the same number of channels as the dataset (") + std::to_string(classes) + std::string(")."));
							goto FAIL;
						}
					}
				}

				if (!isModel)
				{
					const auto& name = layerNames[model->Layers.size()].first;
					const auto inputs = model->GetLayerInputs(inputsStr);

					switch (layerType)
					{
						case LayerTypes::Add:
						case LayerTypes::Substract:
						case LayerTypes::Max:
						case LayerTypes::Min:
						case LayerTypes::Multiply:
						case LayerTypes::Divide:
						case LayerTypes::Average:
						{
							if (inputs.size() != 2)
							{
								msg = CheckMsg(line, col, std::string("Layer ") + name + std::string(" has no two inputs."));
								goto FAIL;
							}

							/*if (inputs[0]->C != inputs[1]->C)
							{
								msg = CheckMsg(line, col, std::string("Layer ") + name + std::string(" has uneven channels in the input ") + inputs[1]->Name + std::string(", must have ") + std::to_string(inputs[0]->C) + std::string(" channels."));
								goto FAIL;
							}*/
						}
						break;
							
						case LayerTypes::Concat:
						{
							if (inputs.size() < 2)
							{
								msg = CheckMsg(line, col, std::string("Layer ") + name + std::string(" has just one input."));
								goto FAIL;
							}
						}
						break;

						default:
						{
							if (inputs.size() > 1)
							{
								msg = CheckMsg(line, col, std::string("Layer ") + name + std::string(" must have only one input."));
								goto FAIL;
							}
						}
					}

					try
					{
						switch (layerType)
						{
						case LayerTypes::Input:
							break;
						case LayerTypes::Activation:
							model->Layers.push_back(std::make_unique<Activation>(model->Device, model->Format, name, activationFunction, inputs, alpha, beta));
							break;
						case LayerTypes::Add:
							model->Layers.push_back(std::make_unique<Add>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Average:
							model->Layers.push_back(std::make_unique<Average>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::AvgPooling:
							model->Layers.push_back(std::make_unique<AvgPooling>(model->Device, model->Format, name, inputs, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW));
							break;
						case LayerTypes::BatchNorm:
							model->Layers.push_back(std::make_unique<BatchNorm>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::BatchNormActivation:
							model->Layers.push_back(std::make_unique<BatchNormActivation>(model->Device, model->Format, name, activationFunction, inputs, scaling, alpha, beta, momentum, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::BatchNormActivationDropout:
							model->Layers.push_back(std::make_unique<BatchNormActivationDropout>(model->Device, model->Format, name, activationFunction, inputs, dropout, dropout != model->Dropout, scaling, alpha, beta, momentum, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::BatchNormRelu:
							model->Layers.push_back(std::make_unique<BatchNormRelu>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::ChannelSplit:
							model->Layers.push_back(std::make_unique<ChannelSplit>(model->Device, model->Format, name, inputs, group, groups));
							break;
						case LayerTypes::ChannelSplitRatioLeft:
							model->Layers.push_back(std::make_unique<ChannelSplitRatioLeft>(model->Device, model->Format, name, inputs, ratio));
							break;
						case LayerTypes::ChannelSplitRatioRight:
							model->Layers.push_back(std::make_unique<ChannelSplitRatioRight>(model->Device, model->Format, name, inputs, ratio));
							break;
						case LayerTypes::ChannelZeroPad:
							model->Layers.push_back(std::make_unique<ChannelZeroPad>(model->Device, model->Format, name, inputs, c));
							break;
						case LayerTypes::Concat:
							model->Layers.push_back(std::make_unique<Concat>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Convolution:
							model->Layers.push_back(std::make_unique<Convolution>(model->Device, model->Format, name, inputs, c, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW, groups, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::ConvolutionTranspose:
							model->Layers.push_back(std::make_unique<ConvolutionTranspose>(model->Device, model->Format, name, inputs, c, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::Cost:
							model->Layers.push_back(std::make_unique<Cost>(model->Device, model->Format, name, costFunction, groupIndex, labelIndex, c, inputs, labelTrue, labelFalse, weight, epsSpecified ? eps : Float(0)));
							model->CostLayers.push_back(dynamic_cast<Cost*>(model->Layers[model->Layers.size() - 1].get()));
							model->CostFunc = costFunction;
							break;
						case LayerTypes::Dense:
							model->Layers.push_back(std::make_unique<Dense>(model->Device, model->Format, name, c, inputs, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::DepthwiseConvolution:
							model->Layers.push_back(std::make_unique<DepthwiseConvolution>(model->Device, model->Format, name, inputs, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW, multiplier, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::Divide:
							model->Layers.push_back(std::make_unique<Divide>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Dropout:
							model->Layers.push_back(std::make_unique<Dropout>(model->Device, model->Format, name, inputs, dropout, dropout != model->Dropout));
							break;
						case LayerTypes::GlobalAvgPooling:
							model->Layers.push_back(std::make_unique<GlobalAvgPooling>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::GlobalMaxPooling:
							model->Layers.push_back(std::make_unique<GlobalMaxPooling>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::GroupNorm:
							model->Layers.push_back(std::make_unique<GroupNorm>(model->Device, model->Format, name, inputs, scaling, groups, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::LayerNorm:
							model->Layers.push_back(std::make_unique<LayerNorm>(model->Device, model->Format, name, inputs, scaling, eps, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::LocalResponseNorm:
							model->Layers.push_back(std::make_unique<LocalResponseNorm>(model->Device, model->Format, name, inputs, acrossChannels, localSize, alpha, beta, k));
							break;
						case LayerTypes::LogSoftmax:
							model->Layers.push_back(std::make_unique<LogSoftmax>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Max:
							model->Layers.push_back(std::make_unique<Max>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::MaxPooling:
							model->Layers.push_back(std::make_unique<MaxPooling>(model->Device, model->Format, name, inputs, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW));
							break;
						case LayerTypes::Min:
							model->Layers.push_back(std::make_unique<Min>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Multiply:
							model->Layers.push_back(std::make_unique<Multiply>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::PartialDepthwiseConvolution:
							model->Layers.push_back(std::make_unique<PartialDepthwiseConvolution>(model->Device, model->Format, name, inputs, group, groups, kernelH, kernelW, strideH, strideW, dilationH, dilationW, padH, padW, multiplier, biases));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::PRelu:
							model->Layers.push_back(std::make_unique<PRelu>(model->Device, model->Format, name, inputs, alpha));
							model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
							break;
						case LayerTypes::Reduction:
							model->Layers.push_back(std::make_unique<Reduction>(model->Device, model->Format, name, inputs, reduceOp, reduceP, reduceEps));
							break;
						case LayerTypes::Resampling:
							model->Layers.push_back(std::make_unique<Resampling>(model->Device, model->Format, name, inputs, algorithm, factorH, factorW));
							break;
						case LayerTypes::Shuffle:
							model->Layers.push_back(std::make_unique<Shuffle>(model->Device, model->Format, name, inputs, groups));
							break;
						case LayerTypes::Softmax:
							model->Layers.push_back(std::make_unique<Softmax>(model->Device, model->Format, name, inputs));
							break;
						case LayerTypes::Substract:
							model->Layers.push_back(std::make_unique<Substract>(model->Device, model->Format, name, inputs));
							break;
						}
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, std::string("Exception occured when creating layer ") + name + nwl + nwl + std::string(exception.what()));
						goto FAIL;
					}

					group = 1;
					groups = 1;
					c = 0;
					d = 1;
					h = 0;
					w = 0;
					kernelH = 1;
					kernelW = 1;
					strideH = 1;
					strideW = 1;
					dilationH = 1;
					dilationW = 1;
					padD = 0;
					padH = 0;
					padW = 0;
					factorH = 1;
					factorW = 1;
					weight = Float(1);
					groupIndex = 0;
					labelIndex = 0;
					activationFunction = Activations::Linear;
					multiplier = 1;
					useDefaultParams = true;
					dropout = model->Dropout;
					biases = model->HasBias;
					weightsFiller = model->WeightsFiller;
					weightsFillerMode = model->WeightsFillerMode;
					weightsGain = model->WeightsGain;
					weightsScale = model->WeightsScale;
					weightsLRM = model->WeightsLRM;
					weightsWDM = model->WeightsWDM;
					biasesFiller = model->BiasesFiller;
					biasesFillerMode = model->BiasesFillerMode;
					biasesGain = model->BiasesGain;
					biasesScale = model->BiasesScale;
					biasesLRM = model->BiasesLRM;
					biasesWDM = model->BiasesWDM;
					alpha = model->AlphaFiller;
					beta = model->BetaFiller;
					momentum = model->BatchNormMomentum;
					scaling = model->BatchNormScaling;
					eps = model->BatchNormEps;
					epsSpecified = false;
					acrossChannels = false;
					localSize = 5;
					k = Float(1);
					labelTrue = Float(0.9);
					labelFalse = Float(0.1);
					ratio = model->Ratio;
					
				}
			}
			else if (strLine.find("Dataset=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("Dataset cannot be specified in a layer."));
					goto FAIL;
				}
				if (modelMandatory > 0)
				{
					msg = CheckMsg(line, col, std::string("Dataset must be specified first and only once in a model."));
					goto FAIL;
				}

				params = strLine.erase(0, 8);

				auto ok = false;
				auto datasets = magic_enum::enum_names<Datasets>();
				for (auto& set : datasets)
					if (params == std::string(set))
						ok = true;
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Dataset is not recognized."));
					goto FAIL;
				}

				auto set = magic_enum::enum_cast<Datasets>(params);
				if (set.has_value())
				{
					dataset = set.value();
					switch (dataset)
					{
					case Datasets::cifar100:
						classes = 100;
						break;
					case Datasets::tinyimagenet:
						classes = 200;
						break;
					default:
						classes = 10;
					}

					model->Dataset = dataset;
				}
				else
				{
					msg = CheckMsg(line, col, std::string("Dataset is not recognized."));
					goto FAIL;
				}

				modelMandatory += 1;
			}
			else if (strLine.rfind("Dim=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("Dim cannot be specified in a layer."));
					goto FAIL;
				}
				if (modelMandatory == 0)
				{
					msg = CheckMsg(line, col, std::string("Dim must be specified second in a model."));
					goto FAIL;
				}
				if (modelMandatory > 1)
				{
					msg = CheckMsg(line, col, std::string("Dim must be specified only once in a model."));
					goto FAIL;
				}

				params = strLine.erase(0, 4);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Dim not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 3)
				{
					msg = CheckMsg(line, col, std::string("Dim must have three values."));
					goto FAIL;
				}

				c = values[0];
				d = 1;
				h = values[1];
				w = values[2];

				if (values[0] != 1 && values[0] != 3)
				{
					msg = CheckMsg(line, col, std::string("First Dim (Channels) value must be 1 or 3."));
					goto FAIL;
				}
				if (values[1] < 28 || values[1] > 4096)
				{
					msg = CheckMsg(line, col, std::string("Second Dim (Height) value must be in the range [28-4096]."));
					goto FAIL;
				}
				if (values[2] < 28 || values[2] > 4096)
				{
					msg = CheckMsg(line, col, std::string("Third Dim (Width) value must be in the range [28-4096]."));
					goto FAIL;
				}

				model->C = values[0];
				model->D = 1;
				model->H = values[1];
				model->W = values[2];

				modelMandatory += 128;
			}	
			else if (strLine.rfind("MeanStd=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("MeanStd cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 8);

				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("MeanStd value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				model->MeanStdNormalization = StringToBool(params);
			}
			else if (strLine.rfind("MirrorPad=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("MirrorPad cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("MirrorPad not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("MirrorPad must have two values."));
					goto FAIL;
				}

				padD = 0;
				padH = values[0];
				padW = values[1];

				model->PadD = 0;
				model->PadH = values[0];
				model->PadW = values[1];
				model->MirrorPad = true;
			}
			else if (strLine.rfind("ZeroPad=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("ZeroPad cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 8);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("ZeroPad not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("ZeroPad must have two values."));
					goto FAIL;
				}

				padD = 0;
				padH = values[0];
				padW = values[1];

				model->PadD = 0;
				model->PadH = values[0];
				model->PadW = values[1];
				model->MirrorPad = false;
			}
			else if (strLine.rfind("RandomCrop=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("RandomCrop cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
						
				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("RandomCrop value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				model->RandomCrop = StringToBool(params);
			}
			else if (strLine.rfind("FixedDepthDrop=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("FixedDepthDrop cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 15);

				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("FixedDepthDrop value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				model->FixedDepthDrop = StringToBool(params);
			}
			else if (strLine.rfind("DepthDrop=") == 0)
			{
				if (!isModel)
				{
					msg = CheckMsg(line, col, std::string("DepthDrop cannot be specified in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				if (params.find_first_not_of(".0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("DepthDrop contains illegal characters."));
					goto FAIL;
				}

				try
				{
					depthDrop = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("DepthDrop value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (depthDrop < 0 || depthDrop >= 1)
				{
					msg = CheckMsg(line, col, std::string("DepthDrop value must be in the range [0-1["));
					goto FAIL;
				}

				model->DepthDrop = depthDrop;
			}
			else if (strLine.rfind("Ratio=") == 0)
			{
				if (layerType != LayerTypes::Input && layerType != LayerTypes::ChannelSplitRatioLeft && layerType != LayerTypes::ChannelSplitRatioRight)
				{
					msg = CheckMsg(line, col, std::string("Ratio cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 6);

				if (params.find_first_not_of(".0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Ratio contains illegal characters."));
					goto FAIL;
				}

				try
				{
					ratio = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Ratio value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (ratio <= 0 || ratio >= 1)
				{
					msg = CheckMsg(line, col, std::string("Ratio value must be in the range ]0-1["));
					goto FAIL;
				}

				if (isModel)
					model->Ratio = ratio;
			}
			else if (strLine.rfind("Type=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Type cannot be specified in a model."));
					goto FAIL;
				}
				if (layerMandatory > 0)
				{
					msg = CheckMsg(line, col, std::string("Type must be specified first and only once in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 5);

				auto ok = false;
				auto types = magic_enum::enum_names<LayerTypes>();
				for (const auto& type : types)
					if (params == std::string(type))
						ok = true;
						
				if (params == "Input")
				{
					msg = CheckMsg(line, col, std::string("Type Input cannot be used."));
					goto FAIL;
				}
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Type is not recognized."));
					goto FAIL;
				}

				auto type = magic_enum::enum_cast<LayerTypes>(params);
				if (type.has_value())
					layerType = type.value();
				else
				{
					msg = CheckMsg(line, col, std::string("Type is not recognized."));
					goto FAIL;
				}

				switch (layerType)
				{
					case LayerTypes::BatchNorm:
					case LayerTypes::BatchNormActivation:
					case LayerTypes::BatchNormActivationDropout:
					case LayerTypes::BatchNormRelu:
					case LayerTypes::LayerNorm:
						isNormalizationLayer = true;
						break;
					default:
						isNormalizationLayer = false;
				}

				layerMandatory += 1;
			}
			else if (strLine.rfind("Inputs=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Inputs cannot be specified in a model."));
					goto FAIL;
				}
				if (layerMandatory == 0)
				{
					msg = CheckMsg(line, col, std::string("Inputs must be specified second in a layer."));
					goto FAIL;
				}
				if (layerMandatory > 1)
				{
					msg = CheckMsg(line, col, std::string("Inputs must be specified only once in a layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);
						
				inputsStr = std::vector<std::string>();
				auto list = std::istringstream(params);
				std::string item;
				while (std::getline(list, item, ','))
					inputsStr.push_back(item);

				for (const auto& input : inputsStr)
				{
					auto ok = false;
					for (const auto& name : layerNames)
						if (name.first.compare(input) == 0)
							ok = true;
					if (!ok)
					{
						msg = CheckMsg(line, col, std::string("Inputs ") + input + std::string(" doesn't exists."));
						goto FAIL;
					}

					if (input == layerNames.back().first)
					{
						msg = CheckMsg(line, col, std::string("Inputs ") + input + std::string(" is circular and isn't allowed."));
						goto FAIL;
					}
				}

				layerMandatory += 128;
			}
			else if (strLine.rfind("Momentum=") == 0)
			{
				if (layerType != LayerTypes::Input && !isNormalizationLayer)
				{
					msg = CheckMsg(line, col, std::string("Eps cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 9);

				if (params.find_first_not_of(".0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Momentum contains illegal characters."));
					goto FAIL;
				}

				try 
				{
					momentum = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Momentum value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (momentum <= 0 || momentum >= 1)
				{
					msg = CheckMsg(line, col, std::string("Momentum value must be in the range ]0-1["));
					goto FAIL;
				}

				if (isModel)
					model->BatchNormMomentum = momentum;
			}
			else if (strLine.rfind("Scaling=") == 0)
			{
				if (layerType != LayerTypes::Input && !isNormalizationLayer)
				{
					msg = CheckMsg(line, col, std::string("Eps cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 8);

				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("Scaling value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				scaling = StringToBool(params);

				if (isModel)
					model->BatchNormScaling = scaling;
			}
			else if (strLine.rfind("Eps=") == 0)
			{
				if (layerType != LayerTypes::Input && !isNormalizationLayer && layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("Eps cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 4);
					
				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Eps contains illegal characters."));
					goto FAIL;
				}

				try
				{
					eps = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Eps value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (eps <= Float(0) || eps > Float(1))
				{
					msg = CheckMsg(line, col, std::string("Eps value must be in the range ]0-1]"));
					goto FAIL;
				}

				if (isModel)
					model->BatchNormEps = eps;

				epsSpecified = true;
			}
			else if (strLine.rfind("WeightsFiller=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("WeightsFiller cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 14);

				std::string value;
				auto ok = false;
				auto fillers = magic_enum::enum_names<Fillers>();
				for (auto& filler : fillers)
					if (params.rfind(std::string(filler)) == 0 && magic_enum::enum_cast<Fillers>(filler).has_value())
					{
						weightsFiller = magic_enum::enum_cast<Fillers>(filler).value();
						value = params.erase(0, filler.size());
						ok = true;
						break;
					}
					
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("WeightsFiller not recognized."));
					goto FAIL;
				}

				if (value.size() > 0)
				{
					switch (weightsFiller)
					{
					case Fillers::Constant:
					case Fillers::Normal:
					case Fillers::TruncatedNormal:
					case Fillers::Uniform:
					{
						if (value.find_first_not_of("().-eE0123456789") != std::string::npos)
						{
							msg = CheckMsg(line, col, std::string("WeightsScale contains illegal characters."));
							goto FAIL;
						}

						if (value.size() > 2 && value[0] == '(' && value[value.size() - 1] == ')')
						{
							ok = false;
							try
							{
								weightsScale = std::stof(value.substr(1, value.size() - 2));
								ok = true;
							}
							catch (std::exception exception)
							{
								msg = CheckMsg(line, col, std::string("WeightsScale value not recognized.") + nwl + std::string(exception.what()));
								goto FAIL;
							}

							if (!ok)
							{
								msg = CheckMsg(line, col, std::string("WeightsScale value not recognized."));
								goto FAIL;
							}
						}
						else
						{
							msg = CheckMsg(line, col, std::string("WeightsScale value not recognized."));
							goto FAIL;
						}
					}
					break;

					case Fillers::HeNormal:
					case Fillers::HeUniform:
					case Fillers::LeCunNormal:
					case Fillers::LeCunUniform:
					{
						auto fillerModes = magic_enum::enum_names<FillerModes>();
						for (auto& fillerMode : fillerModes)
						{
							if (value.rfind(std::string(fillerMode)) == 0 && magic_enum::enum_cast<FillerModes>(fillerMode).has_value())
							{
								weightsFillerMode = magic_enum::enum_cast<FillerModes>(fillerMode).value();
								value = params.erase(0, fillerMode.size());
								if (value.size() > 0)
								{
									if (value.find_first_not_of("(),.-eE0123456789") != std::string::npos)
									{
										msg = CheckMsg(line, col, std::string("WeightsGain contains illegal characters."));
										goto FAIL;
									}

									if (value.size() > 2 && value[0] == ',' && value[value.size() - 1] == ')')
									{
										ok = false;
										try
										{
											weightsGain = std::stof(value.substr(1, value.size() - 2));
											ok = true;
										}
										catch (std::exception exception)
										{
											msg = CheckMsg(line, col, std::string("WeightsGain value not recognized.") + nwl + std::string(exception.what()));
											goto FAIL;
										}

										if (!ok)
										{
											msg = CheckMsg(line, col, std::string("WeightsGain value not recognized."));
											goto FAIL;
										}
									}
									else
									{
										msg = CheckMsg(line, col, std::string("WeightsGain value not recognized."));
										goto FAIL;
									}
								}
								else
									ok = true;
							}
						}
					}
					break;
						
					case Fillers::XavierNormal:
					case Fillers::XavierUniform:
					{
						if (value.find_first_not_of("().-eE0123456789") != std::string::npos)
						{
							msg = CheckMsg(line, col, std::string("WeightsGain contains illegal characters."));
							goto FAIL;
						}

						if (value.size() > 2 && value[0] == '(' && value[value.size() - 1] == ')')
						{
							ok = false;
							try
							{
								weightsGain = std::stof(value.substr(1, value.size() - 2));
								ok = true;
							}
							catch (std::exception exception)
							{
								msg = CheckMsg(line, col, std::string("WeightsGain value not recognized.") + nwl + std::string(exception.what()));
								goto FAIL;
							}

							if (!ok)
							{
								msg = CheckMsg(line, col, std::string("WeightsGain value not recognized."));
								goto FAIL;
							}
						}
						else
						{
							msg = CheckMsg(line, col, std::string("WeightsGain value not recognized."));
							goto FAIL;
						}
					}
					break;
					}
				}
								
				if (isModel)
				{
					model->WeightsFiller = weightsFiller;
					switch (weightsFiller)
					{
					case dnn::Fillers::Constant:
					case dnn::Fillers::Normal:
					case dnn::Fillers::TruncatedNormal:
					case dnn::Fillers::Uniform:
					{
						model->WeightsScale = value.size() > 2 ? weightsScale : defaultWeightsScale;
					}
					break;

					case dnn::Fillers::HeNormal:
					case dnn::Fillers::HeUniform:
					case dnn::Fillers::LeCunNormal:
					case dnn::Fillers::LeCunUniform:
					{
						model->WeightsFillerMode = weightsFillerMode;
						model->WeightsGain = value.size() > 2 ? weightsGain : defaultWeightsGain;
					}
					break;
						
					case dnn::Fillers::XavierNormal:
					case dnn::Fillers::XavierUniform:
					{
						model->WeightsGain = value.size() > 2 ? weightsGain : defaultWeightsGain;
					}
					break;
					}
				}
				else 
					useDefaultParams = false;
			}
			else if (strLine.rfind("WeightsLRM=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("WeightsLRM cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
					
				if (params.find_first_not_of("-.eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("WeightsLRM contains illegal characters."));
					goto FAIL;
				}

				try
				{
					weightsLRM = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("WeightsLRM value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (isModel)
					model->WeightsLRM = weightsLRM;

				useDefaultParams = false;
			}
			else if (strLine.rfind("WeightsWDM=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("WeightsWDM cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);

				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("WeightsWDM contains illegal characters."));
					goto FAIL;
				}

				try
				{
					weightsWDM = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("WeightsWDM value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (isModel)
					model->WeightsWDM = weightsWDM;

				useDefaultParams = false;
			}
			else if (strLine.rfind("BiasesFiller=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("BiasesFiller cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 13);

				std::string value;
				auto ok = false;
				auto fillers = magic_enum::enum_names<Fillers>();
				for (auto& filler : fillers)
					if (params.rfind(std::string(filler)) == 0 && magic_enum::enum_cast<Fillers>(filler).has_value())
					{
						biasesFiller = magic_enum::enum_cast<Fillers>(filler).value();
						value = params.erase(0, filler.size());
						ok = true;
						break;
					}
					
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("BiasesFiller not recognized."));
					goto FAIL;
				}

				if (value.size() > 0)
				{
					switch (biasesFiller)
					{
					case Fillers::Constant:
					case Fillers::Normal:
					case Fillers::TruncatedNormal:
					case Fillers::Uniform:
					{
						if (value.find_first_not_of(".()-eE0123456789") != std::string::npos)
						{
							msg = CheckMsg(line, col, std::string("BiasesScale contains illegal characters."));
							goto FAIL;
						}

						if (value.size() > 2 && value[0] == '(' && value[value.size() - 1] == ')')
						{
							ok = false;

							try
							{
								biasesScale = std::stof(value.substr(1, value.size() - 2));
								ok = true;
							}
							catch (std::exception exception)
							{
								msg = CheckMsg(line, col, std::string("BiasesScale value not recognized.") + nwl + std::string(exception.what()));
								goto FAIL;
							}

							if (!ok)
							{
								msg = CheckMsg(line, col, std::string("BiasesScale value not recognized."));
								goto FAIL;
							}
						}
						else
						{
							msg = CheckMsg(line, col, std::string("BiasesScale value not recognized."));
							goto FAIL;
						}
					}
					break;

					case Fillers::HeNormal:
					case Fillers::HeUniform:
					case Fillers::LeCunNormal:
					case Fillers::LeCunUniform:
					{
						auto fillerModes = magic_enum::enum_names<FillerModes>();
						for (auto& fillerMode : fillerModes)
						{
							if (value.rfind(std::string(fillerMode)) == 0 && magic_enum::enum_cast<FillerModes>(fillerMode).has_value())
							{
								biasesFillerMode = magic_enum::enum_cast<FillerModes>(fillerMode).value();
								value = params.erase(0, fillerMode.size());
								if (value.size() > 0)
								{
									if (value.find_first_not_of("(),.-eE0123456789") != std::string::npos)
									{
										msg = CheckMsg(line, col, std::string("BiasesGain contains illegal characters."));
										goto FAIL;
									}

									if (value.size() > 2 && value[0] == ',' && value[value.size() - 1] == ')')
									{
										ok = false;
										try
										{
											biasesGain = std::stof(value.substr(1, value.size() - 2));
											ok = true;
										}
										catch (std::exception exception)
										{
											msg = CheckMsg(line, col, std::string("BiasesGain value not recognized.") + nwl + std::string(exception.what()));
											goto FAIL;
										}

										if (!ok)
										{
											msg = CheckMsg(line, col, std::string("BiasesGain value not recognized."));
											goto FAIL;
										}
									}
									else
									{
										msg = CheckMsg(line, col, std::string("BiasesGain value not recognized."));
										goto FAIL;
									}
								}
								else
									ok = true;
							}
						}
					}
					break;
						
					case Fillers::XavierNormal:
					case Fillers::XavierUniform:
					{
						if (value.find_first_not_of(".()-eE0123456789") != std::string::npos)
						{
							msg = CheckMsg(line, col, std::string("BiasesGain contains illegal characters."));
							goto FAIL;
						}

						if (value.size() > 2 && value[0] == '(' && value[value.size() - 1] == ')')
						{
							ok = false;

							try
							{
								biasesGain = std::stof(value.substr(1, value.size() - 2));
								ok = true;
							}
							catch (std::exception exception)
							{
								msg = CheckMsg(line, col, std::string("BiasesGain value not recognized.") + nwl + std::string(exception.what()));
								goto FAIL;
							}

							if (!ok)
							{
								msg = CheckMsg(line, col, std::string("BiasesGain value not recognized."));
								goto FAIL;
							}
						}
						else
						{
							msg = CheckMsg(line, col, std::string("BiasesGain value not recognized."));
							goto FAIL;
						}
					}
					break;
					}
				}

				if (isModel)
				{
					model->BiasesFiller = biasesFiller;
					switch (biasesFiller)
					{
					case dnn::Fillers::Constant:
					case dnn::Fillers::Normal:
					case dnn::Fillers::TruncatedNormal:
					case dnn::Fillers::Uniform:
					{
						model->BiasesScale = value.size() > 2 ? biasesScale : defaultBiasesScale;
					}
					break;

					case dnn::Fillers::HeNormal:
					case dnn::Fillers::HeUniform:
					case dnn::Fillers::LeCunNormal:
					case dnn::Fillers::LeCunUniform:
					{
						model->BiasesFillerMode = biasesFillerMode;
						model->BiasesGain = value.size() > 2 ? biasesGain : defaultBiasesGain;
					}
					break;
						
					case dnn::Fillers::XavierNormal:
					case dnn::Fillers::XavierUniform:
					{
						model->BiasesGain = value.size() > 2 ? biasesGain : defaultBiasesGain;
					}
					break;
					}
				}
				else
					useDefaultParams = false;
			}
			else if (strLine.rfind("BiasesLRM=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("BiasesLRM cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("BiasesLRM contains illegal characters."));
					goto FAIL;
				}

				try
				{
					biasesLRM = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("BiasesLRM value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (isModel)
					model->BiasesLRM = biasesLRM;

				useDefaultParams = false;
			}
			else if (strLine.rfind("BiasesWDM=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("BiasesWDM cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("BiasesWDM contains illegal characters."));
					goto FAIL;
				}

				try
				{
					biasesWDM = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("BiasesWDM value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (isModel)
					model->BiasesWDM = biasesWDM;

				useDefaultParams = false;
			}
			else if (strLine.rfind("Biases=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("Biases cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);

				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("Biases value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				biases = StringToBool(params);

				if (isModel)
					model->HasBias = biases;
			}
			else if (strLine.rfind("Dropout=") == 0)
			{
				if (layerType != LayerTypes::Input
					&& layerType != LayerTypes::BatchNormActivationDropout
					&& layerType != LayerTypes::Dropout)
				{
					msg = CheckMsg(line, col, std::string("Dropout cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 8);

				if (params.find_first_not_of(".0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Dropout contains illegal characters."));
					goto FAIL;
				}

				try
				{
					dropout = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Dropout value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (dropout < 0 || dropout >= 1)
				{
					msg = CheckMsg(line, col, std::string("Dropout out of range [0-1["));
					goto FAIL;
				}

				if (isModel)
					model->Dropout = dropout;
			}
			else if (strLine.rfind("Alpha=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::PRelu && layerType != LayerTypes::Activation && layerType != LayerTypes::LocalResponseNorm)
				{
					msg = CheckMsg(line, col, std::string("Alpha cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 6);

				if (params.find_first_not_of(".-0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Alpha contains illegal characters."));
					goto FAIL;
				}

				try
				{
					alpha = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Alpha value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
						
				if (isModel)
					model->AlphaFiller = alpha;
			}
			else if (strLine.rfind("Beta=") == 0)
			{
				if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::Activation && layerType != LayerTypes::LocalResponseNorm)
				{
					msg = CheckMsg(line, col, std::string("Beta cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 5);

				if (params.find_first_not_of(".-0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Beta contains illegal characters."));
					goto FAIL;
				}

				try
				{
					beta = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Beta value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
						
				if (isModel)
					model->BetaFiller = beta;
			}
			else if (strLine.rfind("AcrossChannels=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("AcrossChannels cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::LocalResponseNorm)
				{
					msg = CheckMsg(line, col, std::string("AcrossChannels cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 15);

				if (!IsStringBool(params))
				{
					msg = CheckMsg(line, col, std::string("AcrossChannels value must be boolean (Yes/No or True/False)."));
					goto FAIL;
				}

				acrossChannels = StringToBool(params);
			}
			else if (strLine.rfind("K=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("K cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::LocalResponseNorm)
				{
					msg = CheckMsg(line, col, std::string("K cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 2);

				if (params.find_first_not_of(".-0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("K contains illegal characters."));
					goto FAIL;
				}

				try
				{
					k = std::stof(Trim(params));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("K value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
			}
			else if (strLine.rfind("LocalSize=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("LocalSize cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::LocalResponseNorm)
				{
					msg = CheckMsg(line, col, std::string("LocalSize cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				try
				{
					localSize = std::stoull(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("LocalSize value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (localSize == 0)
				{
					msg = CheckMsg(line, col, std::string("LocalSize value cannot be zero."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Multiplier=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Multiplier cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("Multiplier cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
						
				try
				{
					multiplier = std::stoull(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Multiplier value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (multiplier == 0)
				{
					msg = CheckMsg(line, col, std::string("Multiplier value cannot be zero."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Group=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Group cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::ChannelSplit && layerType != LayerTypes::PartialDepthwiseConvolution)
				{
					msg = CheckMsg(line, col, std::string("Group cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 6);
						
				try
				{
					group = std::stoull(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Group value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (group == 0)
				{
					msg = CheckMsg(line, col, std::string("Group value cannot be zero."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Groups=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Groups cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Shuffle && layerType != LayerTypes::ChannelSplit && layerType != LayerTypes::Convolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::GroupNorm)
				{
					msg = CheckMsg(line, col, std::string("Groups cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);
						
				try
				{
					groups = std::stoull(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Groups value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (groups == 0)
				{
					msg = CheckMsg(line, col, std::string("Groups value cannot be zero."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Factor=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Factor cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Resampling)
				{
					msg = CheckMsg(line, col, std::string("Factor cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);
				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<float>();
				try
				{
					while (std::getline(list, item, ','))
					{
						if (item.find_first_not_of(".0123456789") != std::string::npos)
						{
							msg = CheckMsg(line, col, std::string("Factor contains illegal characters."));
							goto FAIL;
						}

						values.push_back(std::stof(item));
					}
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Factor value(s) not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("Factor must have two floaing point values."));
					goto FAIL;
				}

				factorH = values[0];
				factorW = values[1];
			}
			else if (strLine.rfind("Algorithm=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Algorithm cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Resampling)
				{
					msg = CheckMsg(line, col, std::string("Algorithm cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				auto ok = false;
				for (const auto& algo : magic_enum::enum_names<Algorithms>())
					if (params == std::string(algo))
						ok = true;
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Algorithm is not recognized."));
					goto FAIL;
				}

				if (magic_enum::enum_cast<Algorithms>(params).has_value())
					algorithm = magic_enum::enum_cast<Algorithms>(params).value();
				else
				{
					msg = CheckMsg(line, col, std::string("Algorithm unknown."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("GroupIndex=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("GroupIndex cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("GroupIndex cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
						
				try
				{
					groupIndex = std::stoul(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("GroupIndex value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
			}
			else if (strLine.rfind("LabelIndex=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("LabelIndex cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("LabelIndex cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
						
				try
				{
					labelIndex = std::stoul(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("LabelIndex value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Weight=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Weight cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("Weight cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);
						
				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("Weight contains illegal characters."));
					goto FAIL;
				}

				try
				{
					weight = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Weight value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (weight < -10.0f || weight > 10.0f)
				{
					msg = CheckMsg(line, col, std::string("Weight value must be int the range [-10-10]"));
					goto FAIL;
				}
			}
			else if (strLine.rfind("LabelTrue=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("LabelTrue cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("LabelTrue cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);
						
				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("LabelTrue contains illegal characters."));
					goto FAIL;
				}

				try
				{
					labelTrue = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("LabelTrue value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (labelTrue < -10.0f || labelTrue > 10.0f)
				{
					msg = CheckMsg(line, col, std::string("LabelTrue value must be int the range [-10-10]"));
					goto FAIL;
				}
			}
			else if (strLine.rfind("LabelFalse=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("LabelFalse cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("LabelFalse cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);
						
				if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
				{
					msg = CheckMsg(line, col, std::string("LabelFalse contains illegal characters."));
					goto FAIL;
				}

				try
				{
					labelFalse = std::stof(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("LabelFalse value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;

				}

				if (labelFalse < -10.0f || labelFalse > 10.0f)
				{
					msg = CheckMsg(line, col, std::string("LabelFalse value must be int the range [-10-10]"));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Cost=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Cost cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("Cost cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 5);

				auto ok = false;
				auto costs = magic_enum::enum_names<Costs>();
				for (const auto& cost : costs)
					if (params == std::string(cost))
						ok = true;
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Cost is not recognized."));
					goto FAIL;
				}

				auto cost = magic_enum::enum_cast<Costs>(params);
				if (cost.has_value())
					costFunction = cost.value();
				else
				{
					msg = CheckMsg(line, col, std::string("Cost is not recognized."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Activation=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Activation cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::BatchNormActivation && layerType != LayerTypes::BatchNormActivationDropout && layerType != LayerTypes::Activation)
				{
					msg = CheckMsg(line, col, std::string("Activation cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 11);

				auto ok = false;
				auto activations = magic_enum::enum_names<Activations>();
				for (const auto& activation : activations)
					if (params == std::string(activation))
						ok = true;
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Activation is not recognized."));
					goto FAIL;
				}

				auto activation = magic_enum::enum_cast<Activations>(params);
				if (activation.has_value())
					activationFunction = activation.value();
				else
				{
					msg = CheckMsg(line, col, std::string("Activation is not recognized."));
					goto FAIL;
				}

				switch (activationFunction)
				{
				case Activations::HardSigmoid:
				case Activations::Log:
				case Activations::Sigmoid:
				case Activations::LogSigmoid:
					labelTrue = Float(1);
					labelFalse = Float(0);
					break;
				default:
					break;
				}
			}
			else if (strLine.rfind("Channels=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Channels cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::ChannelZeroPad && layerType != LayerTypes::Dense && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Cost)
				{
					msg = CheckMsg(line, col, std::string("Channels cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 9);

				try
				{
					c = std::stoul(params);
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Channels value not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (c == 0)
				{
					msg = CheckMsg(line, col, std::string("Channels value cannot be zero."));
					goto FAIL;
				}
			}
			else if (strLine.rfind("Kernel=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Kernel cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
				{
					msg = CheckMsg(line, col, std::string("Kernel cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Kernel not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("Kernel must have two values."));
					goto FAIL;
				}

				if (values[0] == 0 || values[1] == 0)
				{
					msg = CheckMsg(line, col, std::string("Kernel values cannot be zero."));
					goto FAIL;
				}

				kernelH = values[0];
				kernelW = values[1];
			}
			else if (strLine.rfind("Dilation=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Dilation cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
				{
					msg = CheckMsg(line, col, std::string("Dilation cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 9);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Dilation not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("Dilation must have two values."));
					goto FAIL;
				}

				if (values[0] == 0 || values[1] == 0)
				{
					msg = CheckMsg(line, col, std::string("Dilation values cannot be zero."));
					goto FAIL;
				}

				dilationH = values[0];
				dilationW = values[1];
			}
			else if (strLine.rfind("Stride=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Stride cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
				{
					msg = CheckMsg(line, col, std::string("Stride cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 7);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Stride not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}

				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("Stride must have two values."));
					goto FAIL;
				}

				if (values[0] == 0 || values[1] == 0)
				{
					msg = CheckMsg(line, col, std::string("Stride values cannot be zero."));
					goto FAIL;
				}

				strideH = values[0];
				strideW = values[1];
			}
			else if (strLine.rfind("Pad=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Pad cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
				{
					msg = CheckMsg(line, col, std::string("Pad cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 4);

				auto list = std::istringstream(params);
				std::string item;
				auto values = std::vector<UInt>();

				try
				{
					while (std::getline(list, item, ','))
						values.push_back(std::stoull(item));
				}
				catch (std::exception exception)
				{
					msg = CheckMsg(line, col, std::string("Pad not recognized.") + nwl + std::string(exception.what()));
					goto FAIL;
				}
					
				if (values.size() != 2)
				{
					msg = CheckMsg(line, col, std::string("Pad must have two values."));
					goto FAIL;
				}

				padD = 0;
				padH = values[0];
				padW = values[1];
			}
			else if (strLine.rfind("Operation=") == 0)
			{
				if (isModel)
				{
					msg = CheckMsg(line, col, std::string("Operation cannot be specified in a model."));
					goto FAIL;
				}

				if (layerType != LayerTypes::Reduction)
				{
					msg = CheckMsg(line, col, std::string("Operation cannot be specified in a ") + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + std::string(" layer."));
					goto FAIL;
				}

				params = strLine.erase(0, 10);

				auto ok = false;
				auto ops = magic_enum::enum_names<ReduceOperations>();
				for (const auto& op : ops)
					if (params == std::string(op))
						ok = true;
				if (!ok)
				{
					msg = CheckMsg(line, col, std::string("Operation is not recognized."));
					goto FAIL;
				}

				auto op = magic_enum::enum_cast<ReduceOperations>(params);
				if (op.has_value())
					reduceOp = op.value();
				else
				{
					msg = CheckMsg(line, col, std::string("Operation is not recognized."));
					goto FAIL;
				}
			}
			else
			{
				msg = CheckMsg(line, col, std::string("Unrecognized tokens: ") + strLine);
				goto FAIL;
			}
		}
	
		if (layerType == LayerTypes::Cost)
		{
			if (c != classes)
			{
				msg = CheckMsg(line, col, std::string("Cost layers has not the same number of channels as the dataset: ") + std::to_string(classes));
				goto FAIL;
			}

			model->Layers.push_back(std::make_unique<Cost>(model->Device, model->Format, layerNames[model->Layers.size()].first, costFunction, groupIndex, labelIndex, c, model->GetLayerInputs(inputsStr), labelTrue, labelFalse, weight, epsSpecified ? eps : Float(0)));
			model->CostLayers.push_back(dynamic_cast<Cost*>(model->Layers[model->Layers.size() - 1].get()));
			model->CostFunc = costFunction;
		}

		{
			auto unreferencedLayers = model->SetRelations();

			if (unreferencedLayers.size() > 0)
			{
				auto l = unreferencedLayers[0];
				for (const auto& t : layerNames)
					if (t.first == l->Name)
						line = t.second;

				msg = CheckMsg(line, col, std::string("Layer ") + l->Name + std::string(" never referenced."));
				goto FAIL;
			}
		}

		if (model && !model->CostLayers.empty())
		{
			model->CostIndex = model->CostLayers.size() - 1ull;
			model->GroupIndex = model->CostLayers[model->CostIndex]->GroupIndex;
			model->LabelIndex = model->CostLayers[model->CostIndex]->LabelIndex;
		}
		else
		{
			msg = CheckMsg(line, col, std::string("A Cost layer is missing in the model"));
			goto FAIL;
		}

		for (auto l : model->CostLayers)
			if (model->GetLayerOutputs(l).size() > 0)
			{
				for (const auto& t : layerNames)
					if (t.first == l->Name)
						line = t.second;

				msg = CheckMsg(line, col, std::string("Cost Layer ") + l->Name + std::string(" is referenced."));
				goto FAIL;
			}

		if (model->Layers.back()->LayerType != LayerTypes::Cost)
		{
			msg = CheckMsg(line, col, std::string("Last layer must of type Cost."));
			goto FAIL;
		}

        // ToDo:
        // when skipping layers, check if it is compatible
        // check model definition is a welformed Directed Acyclic Graph
        // check parameters
           
		if (onlyCheck)
		{
			if (model != nullptr)
			{
				model->~Model();
				model = nullptr;
			}
		}
		else
			model->ResetWeights();
			
		msg = CheckMsg(0, 0, std::string("No issues found"), false);	// All checks have passed
		
		std::locale::global(loc);
		return model;

	FAIL:
		if (model != nullptr)
		{
			model->~Model();
			model = nullptr;
		}

		std::locale::global(loc);
		return nullptr;
	}

	bool Check(std::string& definition, CheckMsg& checkMsg)
	{
		definition = NormalizeDefinition(definition);

		Parse(definition, checkMsg, true);

		return checkMsg.Error;
	}

	Model* Read(const std::string& definition, Dataprovider* dataprovider, CheckMsg& checkMsg)
	{
		Model* model = Parse(NormalizeDefinition(definition), checkMsg, false, dataprovider);

		if (checkMsg.Error)
		{

		}
			
		return model;
	}

	Model* Load(const std::string& fileName, Dataprovider* dataprovider, CheckMsg& checkMsg)
	{
		Model* model = nullptr;

		auto file = std::ifstream(fileName);
		if (!file.bad() && file.is_open())
		{
			std::stringstream stream;
			stream << file.rdbuf();
			const auto buffer = stream.str();
			file.close();
			model = Read(buffer.c_str(), dataprovider, checkMsg);
		}

		return model;
	}	
}