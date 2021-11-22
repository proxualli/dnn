#pragma once
#include "Model.h"

namespace dnn
{
	class Definition final
	{
	public:
		static std::string Normalize(const std::string& definition)
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
			
			auto types = magic_enum::enum_names<LayerTypes>();
			for (auto type : types)
			{
				auto text = "Type=" + std::string(type);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
			}

			auto activations = magic_enum::enum_names<Activations>();
			for (auto activation : activations)
			{
				auto text = "Activation=" + std::string(activation);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
			}
			
			auto costs = magic_enum::enum_names<Costs>();
			for (auto cost : costs)
			{
				auto text = "Cost=" + std::string(cost);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
			}
			
			auto fillers = magic_enum::enum_names<Fillers>();
			for (auto filler : fillers)
			{
				auto textFillerWeights = "WeightsFiller=" + std::string(filler);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerWeights, textFillerWeights);
				
				auto textFillerBiases = "BiasesFiller=" + std::string(filler);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerBiases, textFillerBiases);
			}

			auto fillerModes = magic_enum::enum_names<FillerModes>();
			for (auto fillerMode : fillerModes)
			{
				auto textFillerMode = "(" + std::string(fillerMode) + ")";
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), textFillerMode, textFillerMode);
			}

			auto datasets = magic_enum::enum_names<Datasets>();
			for (auto dataset : datasets)
			{
				auto text = "Dataset=" + std::string(dataset);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
			}
			
			auto algorithms = magic_enum::enum_names<Algorithms>();
			for (auto algorithm : algorithms)
			{
				auto text = "Algorithm=" + std::string(algorithm);
				defNorm = CaseInsensitiveReplace(defNorm.begin(), defNorm.end(), text, text);
			}

			return defNorm;
		}

		static Model* Read(const std::string& definition, CheckMsg& msg, const bool onlyCheck = false, Dataprovider* dataprovider = nullptr)
		{
			auto userLocale = std::setlocale(LC_ALL, "C");

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
			auto weightsFillerMode = FillerModes::Auto;
			auto defaultWeightsGain = Float(1);
			auto weightsGain = Float(1);
			auto defaultWeightsScale = Float(0.05);
			auto weightsScale = Float(0.05);
			auto weightsLRM = Float(1);
			auto weightsWDM = Float(1);
			auto biasesFiller = Fillers::Constant;
			auto biasesFillerMode = FillerModes::Auto;
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
			bool acrossChannels = false;
			auto localSize = UInt(5);
			auto k = Float(1);
			auto multiplier = UInt(1);
			auto group = UInt(1);
			auto groups = UInt(1);
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

			auto iss = std::istringstream(definition);
			std::string strLine = "", modelName = "", layerName = "", params = "";
			auto layerNames = std::vector<std::pair<std::string, UInt>>();
			UInt line = 0, col = 0, modelMandatory = 0, layerMandatory = 0;
			auto isModel = true;
			
			while (SafeGetline(iss, strLine))
			{
				line++;
				
				if (strLine == "")
					continue;

				col = strLine.find_first_not_of("[],.=()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
				if (col != std::string::npos)
				{
					col++;
					msg =  CheckMsg(line, col, "Line contains illegal characters.");
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
						msg = CheckMsg(line, col, "Model or layer name contains illegal characters.");
						goto FAIL;
					}
					col = strLine.length() + 1;

					if (isModel)
					{
						if (modelName.empty())
						{
							modelName = layerName;
							model = new Model(modelName, dataprovider);
							
							layerNames.push_back(std::make_pair("Input", line));
						}
						else
						{
							if (modelMandatory != 129)
							{
								msg = CheckMsg(line, col, "Model doesn't have the Dataset and Dim specifiers.");
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

							model->Layers.push_back(std::make_unique<Input>(model->Device, model->Format, "Input", c, model->RandomCrop ? d : d + padD, model->RandomCrop ? h : h + padH, model->RandomCrop ? w : w + padW));

							isModel = false;

							auto exists = false;
							for (auto layer : layerNames)
								if (layer.first == layerName)
									exists = true;

							if (exists)
							{
								msg = CheckMsg(line, col, "Name already in use, must be unique.");
								goto FAIL;
							}

							layerNames.push_back(std::make_pair(layerName, line));
						}
					}
					else
					{
						if (layerMandatory != 129)
						{
							msg = CheckMsg(line, col, "Layer doesn't have Type and Inputs specifiers.");
							goto FAIL;
						}

						auto exists = false;
						for (auto layer : layerNames)
							if (layer.first == layerName)
								exists = true;

						if (exists)
						{
							msg = CheckMsg(line, col, "Layer name already in use, must be unique.");
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
							case Activations::FTS:
							case Activations::Linear:
							case Activations::Swish:
								break;

							case Activations::PRelu:
								if (alpha == 0)
								{
									msg = CheckMsg(line - 1, col, "Activation used without Alpha parameter.");
									goto FAIL;
								}
								break;

							case Activations::Mish:
							case Activations::Abs:
							case Activations::Exp:
							case Activations::Gelu:
							case Activations::GeluErf:
							case Activations::HardLogistic:
							case Activations::HardSwish:
							case Activations::Log:
							case Activations::Logistic:
							case Activations::LogLogistic:
							case Activations::LogSoftmax:
							case Activations::Pow:
							case Activations::Round:
							case Activations::Softmax:
							case Activations::SoftRelu:
							case Activations::Sqrt:
							case Activations::Square:
							case Activations::Tanh:
							case Activations::TanhExp:
								if (alpha != 0 || beta != 0)
								{
									msg = CheckMsg(line - 1, col, "This Activation cannot have an Alpha or Beta parameter.");
									goto FAIL;
								}
								break;

							case Activations::Clip:
							case Activations::ClipV2:
							    if (alpha == 0 && beta == 0)
								{
									msg = CheckMsg(line - 1, col, "Activation used without Alpha and Beta parameter.");
									goto FAIL;
								}
								break;
															
							case Activations::Relu:
								if (alpha < 0)
								{
									msg = CheckMsg(line - 1, col, "This Activation Alpha parameter must be positive.");
									goto FAIL;
								}
								if (beta != 0)
								{
									msg = CheckMsg(line - 1, col, "This Activation doesn't have a Beta parameter.");
									goto FAIL;
								}
								break;

							}
						}

						if (layerType == LayerTypes::Convolution || layerType == LayerTypes::DepthwiseConvolution || layerType == LayerTypes::ConvolutionTranspose || layerType == LayerTypes::PartialDepthwiseConvolution)
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
								msg = CheckMsg(line - 1, col, "Kernel, Stride, Dilation or Pad invalid in layer " + layerNames[model->Layers.size()].first);
								goto FAIL;
							}
						}

						if (layerType == LayerTypes::MaxPooling || layerType == LayerTypes::AvgPooling)
						{
							auto x = w / (double)strideH;
							auto y = h / (double)strideW;

							auto ok = true;
							if (x - std::floor(x) != 0.0)
								ok = false;
							if (y - std::floor(y) != 0.0)
								ok = false;
							if (x != y)
								ok = false;

							if (!ok)
							{
								msg = CheckMsg(line - 1, col, "Stride invalid in pooling layer " + layerNames[model->Layers.size()].first);
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
								msg = CheckMsg(line - 1, col, "Factor invalid in Resampling layer " + layerNames[model->Layers.size()].first);
								goto FAIL;
							}
						}

						if (layerType == LayerTypes::Cost)
						{
							if (c != classes)
							{
								msg = CheckMsg(line - 1, col, "Cost layers hasn't the same number of channels as the dataset (" + std::to_string(classes) + ").");
								goto FAIL;
							}
						}
					}

					if (!isModel)
					{
						const auto &name = layerNames[model->Layers.size()].first;
						const auto inputs = model->GetLayerInputs(inputsStr);

						switch (layerType)
						{
							case LayerTypes::Add:
							case LayerTypes::Substract:
							case LayerTypes::Max:
							case LayerTypes::Min:
							case LayerTypes::Multiply:
							case LayerTypes::ChannelMultiply:
							case LayerTypes::Divide:
							case LayerTypes::Average:
							{
								if (inputs.size() < 2)
								{
									msg = CheckMsg(line, col, "Layer " + name + " has not enough inputs.");
									goto FAIL;
								}

								for (auto i = 1ull; i < inputs.size(); i++)
									if (inputs[i]->C != inputs[0]->C)
									{
										msg = CheckMsg(line, col, "Layer " + name + " has uneven channels in the input " + inputs[i]->Name + ", must have " + std::to_string(inputs[0]->C) + " channels.");
										goto FAIL;
									}
							}
							break;
							
							case LayerTypes::Concat:
							{
								if (inputs.size() < 2)
								{
									msg = CheckMsg(line, col, "Layer " + name + " has not enough inputs.");
									goto FAIL;
								}
							}
							break;

							default:
							{
								if (inputs.size() > 1)
								{
									msg = CheckMsg(line, col, "Layer " + name + " must have only one input.");
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
								switch (activationFunction)
								{
								case Activations::BoundedRelu:
									if (alpha == 0)
										alpha = 6;
									break;
								case Activations::Elu:
								case Activations::Linear:
								case Activations::Swish:
									if (alpha == 0)
										alpha = 1;
									break;
								default:
									break;
								}
								model->Layers.push_back(std::make_unique<Activation>(model->Device, model->Format, name, activationFunction, inputs, alpha, beta));
								break;
							case LayerTypes::Add:
								model->Layers.push_back(std::make_unique<Add>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::Average:
								model->Layers.push_back(std::make_unique<Average>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::AvgPooling:
								model->Layers.push_back(std::make_unique<AvgPooling>(model->Device, model->Format, name, inputs, kernelH, kernelW, strideH, strideW, padH, padW));
								break;
							case LayerTypes::BatchNorm:
								model->Layers.push_back(std::make_unique<BatchNorm>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormHardLogistic:
								model->Layers.push_back(std::make_unique<BatchNormActivation<HardLogistic, LayerTypes::BatchNormHardLogistic>>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormHardSwish:
								model->Layers.push_back(std::make_unique<BatchNormActivation<HardSwish, LayerTypes::BatchNormHardSwish>>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormHardSwishDropout:
								model->Layers.push_back(std::make_unique<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>>(model->Device, model->Format, name, inputs, dropout, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormMish:
								model->Layers.push_back(std::make_unique<BatchNormActivation<Mish, LayerTypes::BatchNormMish>>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormMishDropout:
								model->Layers.push_back(std::make_unique<BatchNormActivationDropout<Mish, LayerTypes::BatchNormMishDropout>>(model->Device, model->Format, name, inputs, dropout, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormRelu:
								model->Layers.push_back(std::make_unique<BatchNormRelu>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormReluDropout:
								model->Layers.push_back(std::make_unique<BatchNormActivationDropout<Relu, LayerTypes::BatchNormReluDropout>>(model->Device, model->Format, name, inputs, dropout, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormSwish:
								model->Layers.push_back(std::make_unique<BatchNormActivation<Swish, LayerTypes::BatchNormSwish>>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormSwishDropout:
								model->Layers.push_back(std::make_unique<BatchNormActivationDropout<Swish, LayerTypes::BatchNormSwishDropout>>(model->Device, model->Format, name, inputs, dropout, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormTanhExp:
								model->Layers.push_back(std::make_unique<BatchNormActivation<TanhExp, LayerTypes::BatchNormTanhExp>>(model->Device, model->Format, name, inputs, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::BatchNormTanhExpDropout:
								model->Layers.push_back(std::make_unique<BatchNormActivationDropout<TanhExp, LayerTypes::BatchNormTanhExpDropout>>(model->Device, model->Format, name, inputs, dropout, scaling, momentum, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::ChannelMultiply:
								model->Layers.push_back(std::make_unique<ChannelMultiply>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::ChannelShuffle:
								model->Layers.push_back(std::make_unique<ChannelShuffle>(model->Device, model->Format, name, inputs, groups));
								break;
							case LayerTypes::ChannelSplit:
								model->Layers.push_back(std::make_unique<ChannelSplit>(model->Device, model->Format, name, inputs, group, groups));
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
								model->CostFuction = costFunction;
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
								model->Layers.push_back(std::make_unique<Dropout>(model->Device, model->Format, name, inputs, dropout));
								break;
							case LayerTypes::GlobalAvgPooling:
								model->Layers.push_back(std::make_unique<GlobalAvgPooling>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::GlobalMaxPooling:
								model->Layers.push_back(std::make_unique<GlobalMaxPooling>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::LayerNorm:
								model->Layers.push_back(std::make_unique<LayerNorm>(model->Device, model->Format, name, inputs, scaling, eps, biases));
								model->Layers[model->Layers.size() - 1]->SetParameters(useDefaultParams, weightsFiller, weightsFillerMode, weightsGain, weightsScale, weightsLRM, weightsWDM, biasesFiller, biasesFillerMode, biasesGain, biasesScale, biasesLRM, biasesWDM);
								break;
							case LayerTypes::LocalResponseNorm:
								model->Layers.push_back(std::make_unique<LocalResponseNorm>(model->Device, model->Format, name, inputs, acrossChannels, localSize, alpha, beta, k));
								break;
							case LayerTypes::Max:
								model->Layers.push_back(std::make_unique<Max>(model->Device, model->Format, name, inputs));
								break;
							case LayerTypes::MaxPooling:
								model->Layers.push_back(std::make_unique<MaxPooling>(model->Device, model->Format, name, inputs, kernelH, kernelW, strideH, strideW, padH, padW));
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
							case LayerTypes::Resampling:
								model->Layers.push_back(std::make_unique<Resampling>(model->Device, model->Format, name, inputs, algorithm, factorH, factorW));
								break;
							case LayerTypes::Substract:
								model->Layers.push_back(std::make_unique<Substract>(model->Device, model->Format, name, inputs));
								break;
							}
						}
						catch (std::exception exception)
						{
							msg = CheckMsg(line, col, "Exception occured when creating layer " + name + nwl + nwl + exception.what());
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
						dropout = Float(0);
						weight = Float(1);
						groupIndex = 0;
						labelIndex = 0;
						activationFunction = Activations::Linear;
						multiplier = 1;
						useDefaultParams = true;
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
						dropout = model->Dropout;
						acrossChannels = false;
						localSize = 5;
						k = Float(1);
					}
				}
				else if (strLine.find("Dataset=") == 0)
				{
					if (!isModel)
					{
						msg = CheckMsg(line, col, "Dataset cannot be specified in a layer.");
						goto FAIL;
					}
					if (modelMandatory > 0)
					{
						msg = CheckMsg(line, col, "Dataset must be specified first and only once in a model.");
						goto FAIL;
					}

					params = strLine.erase(0, 8);

					auto ok = false;
					auto datasets = magic_enum::enum_names<Datasets>();
					for (auto set : datasets)
						if (params == std::string(set))
							ok = true;
					if (!ok)
					{
						msg = CheckMsg(line, col, "Dataset is not recognized.");
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
						msg = CheckMsg(line, col, "Dataset is not recognized.");
						goto FAIL;
					}

					modelMandatory += 1;
				}
				else if (strLine.rfind("Dim=") == 0)
				{
					if (!isModel)
					{
						msg = CheckMsg(line, col, "Dim cannot be specified in a layer.");
						goto FAIL;
					}
					if (modelMandatory == 0)
					{
						msg = CheckMsg(line, col, "Dim must be specified second in a model.");
						goto FAIL;
					}
					if (modelMandatory > 1)
					{
						msg = CheckMsg(line, col, "Dim must be specified only once in a model.");
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
						msg = CheckMsg(line, col, "Dim not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 3)
					{
						msg = CheckMsg(line, col, "Dim must have three values.");
						goto FAIL;
					}

					c = values[0];
					d = 1;
					h = values[1];
					w = values[2];

					if (values[0] != 1 && values[0] != 3)
					{
						msg = CheckMsg(line, col, "First Dim (Channels) value must be 1 or 3.");
						goto FAIL;
					}
					if (values[1] < 28 || values[1] > 4096)
					{
						msg = CheckMsg(line, col, "Second Dim (Height) value must be in the range [28-4096].");
						goto FAIL;
					}
					if (values[2] < 28 || values[2] > 4096)
					{
						msg = CheckMsg(line, col, "Third Dim (Width) value must be in the range [28-4096].");
					}

					model->SampleC = values[0];
					model->SampleD = 1;
					model->SampleH = values[1];
					model->SampleW = values[2];

					modelMandatory += 128;
				}	
				else if (strLine.rfind("MeanStd=") == 0)
				{
					if (!isModel)
					{
						msg = CheckMsg(line, col, "MeanStd cannot be specified in a layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 8);

					if (!IsStringBool(params))
					{
						msg = CheckMsg(line, col, "MeanStd value must be boolean (Yes/No or True/False).");
						goto FAIL;
					}

					model->MeanStdNormalization = StringToBool(params);
				}
				else if (strLine.rfind("MirrorPad=") == 0)
				{
					if (!isModel)
					{
						msg = CheckMsg(line, col, "MirrorPad cannot be specified in a layer.");
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
						msg = CheckMsg(line, col, "MirrorPad not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "MirrorPad must have two values.");
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
						msg = CheckMsg(line, col, "ZeroPad cannot be specified in a layer.");
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
						msg = CheckMsg(line, col, "ZeroPad not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "ZeroPad must have two values.");
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
						msg = CheckMsg(line, col, "RandomCrop cannot be specified in a layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
						
					if (!IsStringBool(params))
					{
						msg = CheckMsg(line, col, "RandomCrop value must be boolean (Yes/No or True/False).");
						goto FAIL;
					}

					model->RandomCrop = StringToBool(params);
				}
				else if (strLine.rfind("Type=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Type cannot be specified in a model.");
						goto FAIL;
					}
					if (layerMandatory > 0)
					{
						msg = CheckMsg(line, col, "Type must be specified first and only once in a layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 5);

					auto ok = false;
					auto types = magic_enum::enum_names<LayerTypes>();
					for (auto type : types)
						if (params == std::string(type))
							ok = true;
						
					if (params == "Input")
					{
						msg = CheckMsg(line, col, "Type Input cannot be used.");
						goto FAIL;
					}
					if (!ok)
					{
						msg = CheckMsg(line, col, "Type is not recognized.");
						goto FAIL;
					}

					auto type = magic_enum::enum_cast<LayerTypes>(params);
					if (type.has_value())
						layerType = type.value();
					else
					{
						msg = CheckMsg(line, col, "Type is not recognized.");
						goto FAIL;
					}

					switch (layerType)
					{
						case LayerTypes::BatchNorm:
						case LayerTypes::BatchNormMish:
						case LayerTypes::BatchNormMishDropout:
						case LayerTypes::BatchNormHardLogistic:
						case LayerTypes::BatchNormHardSwish:
						case LayerTypes::BatchNormHardSwishDropout:
						case LayerTypes::BatchNormRelu:
						case LayerTypes::BatchNormReluDropout:
						case LayerTypes::BatchNormSwish:
						case LayerTypes::BatchNormSwishDropout:
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
						msg = CheckMsg(line, col, "Inputs cannot be specified in a model.");
						goto FAIL;
					}
					if (layerMandatory == 0)
					{
						msg = CheckMsg(line, col, "Inputs must be specified second in a layer.");
						goto FAIL;
					}
					if (layerMandatory > 1)
					{
						msg = CheckMsg(line, col, "Inputs must be specified only once in a layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 7);
						
					inputsStr = std::vector<std::string>();
					auto list = std::istringstream(params);
					std::string item;
					while (std::getline(list, item, ','))
						inputsStr.push_back(item);

					for (auto input : inputsStr)
					{
						auto ok = false;
						for (auto name : layerNames)
							if (name.first.compare(input) == 0)
								ok = true;
						if (!ok)
						{
							msg = CheckMsg(line, col, "Inputs " + input + " doesn't exists.");
							goto FAIL;
						}

						if (input == layerNames.back().first)
						{
							msg = CheckMsg(line, col, "Inputs " + input + " is circular and isn't allowed.");
							goto FAIL;
						}
					}

					layerMandatory += 128;
				}
				else if (strLine.rfind("Momentum=") == 0)
				{
					if (layerType != LayerTypes::Input && !isNormalizationLayer)
					{
						msg = CheckMsg(line, col, "Eps cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 9);

					if (params.find_first_not_of(".0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Momentum contains illegal characters.");
						goto FAIL;
					}

					try 
					{
						momentum = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Momentum value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (momentum <= 0 || momentum >= 1)
					{
						msg = CheckMsg(line, col, "Momentum value must be in the range ]0-1[");
						goto FAIL;
					}

					if (isModel)
						model->BatchNormMomentum = momentum;
				}
				else if (strLine.rfind("Scaling=") == 0)
				{
					if (layerType != LayerTypes::Input && !isNormalizationLayer)
					{
						msg = CheckMsg(line, col, "Eps cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 8);

					if (!IsStringBool(params))
					{
						msg = CheckMsg(line, col, "Scaling value must be boolean (Yes/No or True/False).");
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
						msg = CheckMsg(line, col, "Eps cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 4);
					
					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Eps contains illegal characters.");
						goto FAIL;
					}

					try
					{
						eps = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Eps value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (eps <= 0.0f || eps > 1.0f)
					{
						msg = CheckMsg(line, col, "Eps value must be in the range ]0-1]");
						goto FAIL;
					}

					if (isModel)
						model->BatchNormEps = eps;

					epsSpecified = true;
				}
				else if (strLine.rfind("WeightsFiller=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "WeightsFiller cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 14);

					std::string value;
					auto ok = false;
					auto fillers = magic_enum::enum_names<Fillers>();
					for (auto filler : fillers)
						if (params.rfind(std::string(filler)) == 0 && magic_enum::enum_cast<Fillers>(filler).has_value())
						{
							weightsFiller = magic_enum::enum_cast<Fillers>(filler).value();
							value = params.erase(0, filler.size());
							ok = true;
							break;
						}
					
					if (!ok)
					{
						msg = CheckMsg(line, col, "WeightsFiller not recognized.");
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
								msg = CheckMsg(line, col, "WeightsScale contains illegal characters.");
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
									msg = CheckMsg(line, col, "WeightsScale value not recognized." + nwl + exception.what());
									goto FAIL;
								}

								if (!ok)
								{
									msg = CheckMsg(line, col, "WeightsScale value not recognized.");
									goto FAIL;
								}
							}
							else
							{
								msg = CheckMsg(line, col, "WeightsScale value not recognized.");
								goto FAIL;
							}
						}
						break;
						default:
							auto fillerModes = magic_enum::enum_names<FillerModes>();
							for (auto fillerMode : fillerModes)
								if (value.rfind(std::string(fillerMode)) == 0 && magic_enum::enum_cast<FillerModes>(fillerMode).has_value())
								{
									weightsFillerMode = magic_enum::enum_cast<FillerModes>(fillerMode).value();
									value = params.erase(0, fillerMode.size());
									if (value.size() > 0)
									{
										if (value.find_first_not_of("(),.-eE0123456789") != std::string::npos)
										{
											msg = CheckMsg(line, col, "WeightsGain contains illegal characters.");
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
												msg = CheckMsg(line, col, "WeightsGain value not recognized." + nwl + exception.what());
												goto FAIL;
											}

											if (!ok)
											{
												msg = CheckMsg(line, col, "WeightsGain value not recognized.");
												goto FAIL;
											}
										}
										else
										{
											msg = CheckMsg(line, col, "WeightsGain value not recognized.");
											goto FAIL;
										}
									}
									else
										ok = true;
								}
							break;
						}
					}

					useDefaultParams = false;

					if (isModel)
					{
						model->WeightsFiller = weightsFiller;
						switch (weightsFiller)
						{
						case dnn::Fillers::Constant:
						case dnn::Fillers::Normal:
						case dnn::Fillers::TruncatedNormal:
						case dnn::Fillers::Uniform:
							model->WeightsScale = value.size() > 2 ? weightsScale : defaultWeightsScale;
							break;
						
						case dnn::Fillers::HeNormal:
						case dnn::Fillers::HeUniform:
						case dnn::Fillers::LeCunNormal:
						case dnn::Fillers::LeCunUniform:
						case dnn::Fillers::XavierNormal:
						case dnn::Fillers::XavierUniform:
							model->WeightsFillerMode = weightsFillerMode;
							model->WeightsGain = value.size() > 2 ? weightsGain : defaultWeightsGain;
							break;
						}
					}
				}
				else if (strLine.rfind("WeightsLRM=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "WeightsLRM cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
					
					if (params.find_first_not_of("-.eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "WeightsLRM contains illegal characters.");
						goto FAIL;
					}

					try
					{
						weightsLRM = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "WeightsLRM value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (isModel)
						model->WeightsLRM = weightsLRM;

					useDefaultParams = false;
				}
				else if (strLine.rfind("WeightsWDM=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "WeightsWDM cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);

					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "WeightsWDM contains illegal characters.");
						goto FAIL;
					}

					try
					{
						weightsWDM = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "WeightsWDM value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (isModel)
						model->WeightsWDM = weightsWDM;

					useDefaultParams = false;
				}
				else if (strLine.rfind("BiasesFiller=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "BiasesFiller cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 13);

					std::string value;
					auto ok = false;
					auto fillers = magic_enum::enum_names<Fillers>();
					for (auto filler : fillers)
						if (params.rfind(std::string(filler)) == 0 && magic_enum::enum_cast<Fillers>(filler).has_value())
						{
							biasesFiller = magic_enum::enum_cast<Fillers>(filler).value();
							value = params.erase(0, filler.size());
							ok = true;
							break;
						}
					
					if (!ok)
					{
						msg = CheckMsg(line, col, "BiasesFiller not recognized.");
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
								msg = CheckMsg(line, col, "BiasesScale contains illegal characters.");
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
									msg = CheckMsg(line, col, "BiasesScale value not recognized." + nwl + exception.what());
									goto FAIL;
								}

								if (!ok)
								{
									msg = CheckMsg(line, col, "BiasesScale value not recognized.");
									goto FAIL;
								}
							}
							else
							{
								msg = CheckMsg(line, col, "BiasesScale value not recognized.");
								goto FAIL;
							}
						}
						break;

						default:
							auto fillerModes = magic_enum::enum_names<FillerModes>();
							for (auto fillerMode : fillerModes)
								if (value.rfind(std::string(fillerMode)) == 0 && magic_enum::enum_cast<FillerModes>(fillerMode).has_value())
								{
									biasesFillerMode = magic_enum::enum_cast<FillerModes>(fillerMode).value();
									value = params.erase(0, fillerMode.size());
									if (value.size() > 0)
									{
										if (value.find_first_not_of("(),.-eE0123456789") != std::string::npos)
										{
											msg = CheckMsg(line, col, "BiasesGain contains illegal characters.");
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
												msg = CheckMsg(line, col, "BiasesGain value not recognized." + nwl + exception.what());
												goto FAIL;
											}

											if (!ok)
											{
												msg = CheckMsg(line, col, "BiasesGain value not recognized.");
												goto FAIL;
											}
										}
										else
										{
											msg = CheckMsg(line, col, "BiasesGain value not recognized.");
											goto FAIL;
										}
									}
									else
										ok = true;
								}
							break;
						}
					}

					useDefaultParams = false;

					if (isModel)
					{
						model->BiasesFiller = biasesFiller;
						switch (biasesFiller)
						{
						case dnn::Fillers::Constant:
						case dnn::Fillers::Normal:
						case dnn::Fillers::TruncatedNormal:
						case dnn::Fillers::Uniform:
							model->BiasesScale = value.size() > 2 ? biasesScale : defaultBiasesScale;
							break;

						case dnn::Fillers::HeNormal:
						case dnn::Fillers::HeUniform:
						case dnn::Fillers::LeCunNormal:
						case dnn::Fillers::LeCunUniform:
						case dnn::Fillers::XavierNormal:
						case dnn::Fillers::XavierUniform:
							model->BiasesFillerMode = biasesFillerMode;
							model->BiasesGain = value.size() > 2 ? biasesGain : defaultBiasesGain;
							break;
						}
					}
				}
				else if (strLine.rfind("BiasesLRM=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "BiasesLRM cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 10);

					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "BiasesLRM contains illegal characters.");
						goto FAIL;
					}

					try
					{
						biasesLRM = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "BiasesLRM value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (isModel)
						model->BiasesLRM = biasesLRM;

					useDefaultParams = false;
				}
				else if (strLine.rfind("BiasesWDM=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "BiasesWDM cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 10);

					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "BiasesWDM contains illegal characters.");
						goto FAIL;
					}

					try
					{
						biasesWDM = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "BiasesWDM value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (isModel)
						model->BiasesWDM = biasesWDM;

					useDefaultParams = false;
				}
				else if (strLine.rfind("Biases=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Dense && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "Biases cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 7);

					if (!IsStringBool(params))
					{
						msg = CheckMsg(line, col, "Biases value must be boolean (Yes/No or True/False).");
						goto FAIL;
					}

					biases = StringToBool(params);

					if (isModel)
						model->HasBias = biases;
				}
				else if (strLine.rfind("Dropout=") == 0)
				{
					if (!isNormalizationLayer && layerType != LayerTypes::Input && layerType != LayerTypes::Dropout && layerType != LayerTypes::BatchNormReluDropout && layerType != LayerTypes::BatchNormHardSwishDropout && layerType != LayerTypes::BatchNormMishDropout)
					{
						msg = CheckMsg(line, col, "Dropout cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 8);

					if (params.find_first_not_of(".0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Dropout contains illegal characters.");
						goto FAIL;
					}

					try
					{
						dropout = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Dropout value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (dropout < 0 || dropout >= 1)
					{
						msg = CheckMsg(line, col, "Dropout value must be int the range [0-1[");
						goto FAIL;
					}

					if (isModel)
						model->Dropout = dropout;
				}
				else if (strLine.rfind("Alpha=") == 0)
				{
					if (layerType != LayerTypes::Input && layerType != LayerTypes::Activation && layerType != LayerTypes::LocalResponseNorm)
					{
						msg = CheckMsg(line, col, "Alpha cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 6);

					if (params.find_first_not_of(".-0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Alpha contains illegal characters.");
						goto FAIL;
					}

					try
					{
						alpha = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Alpha value not recognized." + nwl + exception.what());
						goto FAIL;
					}
						
					if (isModel)
						model->AlphaFiller = alpha;
				}
				else if (strLine.rfind("Beta=") == 0)
				{
					if (layerType != LayerTypes::Input && layerType != LayerTypes::Activation && layerType != LayerTypes::LocalResponseNorm)
					{
						msg = CheckMsg(line, col, "Beta cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 5);

					if (params.find_first_not_of(".-0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Beta contains illegal characters.");
						goto FAIL;
					}

					try
					{
						beta = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Beta value not recognized." + nwl + exception.what());
						goto FAIL;
					}
						
					if (isModel)
						model->BetaFiller = beta;
				}
				else if (strLine.rfind("AcrossChannels=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "AcrossChannels cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::LocalResponseNorm)
					{
						msg = CheckMsg(line, col, "AcrossChannels cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 15);

					if (!IsStringBool(params))
					{
						msg = CheckMsg(line, col, "AcrossChannels value must be boolean (Yes/No or True/False).");
						goto FAIL;
					}

					acrossChannels = StringToBool(params);
				}
				else if (strLine.rfind("K=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "K cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::LocalResponseNorm)
					{
						msg = CheckMsg(line, col, "K cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 2);

					if (params.find_first_not_of(".-0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "K contains illegal characters.");
						goto FAIL;
					}

					try
					{
						k = std::stof(Trim(params));
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "K value not recognized." + nwl + exception.what());
						goto FAIL;
					}
				}
				else if (strLine.rfind("LocalSize=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "LocalSize cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::LocalResponseNorm)
					{
						msg = CheckMsg(line, col, "LocalSize cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 10);

					try
					{
						localSize = std::stoull(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "LocalSize value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (localSize == 0)
					{
						msg = CheckMsg(line, col, "LocalSize value cannot be zero.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Multiplier=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Multiplier cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "Multiplier cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
						
					try
					{
						multiplier = std::stoull(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Multiplier value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (multiplier == 0)
					{
						msg = CheckMsg(line, col, "Multiplier value cannot be zero.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Group=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Group cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::ChannelSplit && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "Group cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 6);
						
					try
					{
						group = std::stoull(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Group value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (group == 0)
					{
						msg = CheckMsg(line, col, "Group value cannot be zero.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Groups=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Groups cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::ChannelShuffle && layerType != LayerTypes::ChannelSplit && layerType != LayerTypes::Convolution && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "Groups cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 7);
						
					try
					{
						groups = std::stoull(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Groups value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (groups == 0)
					{
						msg = CheckMsg(line, col, "Groups value cannot be zero.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Factor=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Factor cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Resampling)
					{
						msg = CheckMsg(line, col, "Factor cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
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
								msg = CheckMsg(line, col, "Factor contains illegal characters.");
								goto FAIL;
							}

							values.push_back(std::stof(item));
						}
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Factor value(s) not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "Factor must have two floaing point values.");
						goto FAIL;
					}

					factorH = values[0];
					factorW = values[1];
				}
				else if (strLine.rfind("Algorithm=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Algorithm cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Resampling)
					{
						msg = CheckMsg(line, col, "Algorithm cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 10);

					auto ok = false;
					for (auto algo : magic_enum::enum_names<Algorithms>())
						if (params == std::string(algo))
							ok = true;
					if (!ok)
					{
						msg = CheckMsg(line, col, "Algorithm is not recognized.");
						goto FAIL;
					}

					if (magic_enum::enum_cast<Algorithms>(params).has_value())
						algorithm = magic_enum::enum_cast<Algorithms>(params).value();
					else
					{
						msg = CheckMsg(line, col, "Algorithm unknown.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("GroupIndex=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "GroupIndex cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "GroupIndex cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
						
					try
					{
						groupIndex = std::stoul(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "GroupIndex value not recognized." + nwl + exception.what());
						goto FAIL;
					}
				}
				else if (strLine.rfind("LabelIndex=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "LabelIndex cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "LabelIndex cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
						
					try
					{
						labelIndex = std::stoul(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "LabelIndex value not recognized." + nwl + exception.what());
						goto FAIL;
					}
				}
				else if (strLine.rfind("Weight=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Weight cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "Weight cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 7);
						
					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "Weight contains illegal characters.");
						goto FAIL;
					}

					try
					{
						weight = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Weight value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (weight < -10.0f || weight > 10.0f)
					{
						msg = CheckMsg(line, col, "Weight value must be int the range [-10-10]");
						goto FAIL;
					}
				}
				else if (strLine.rfind("LabelTrue=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "LabelTrue cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "LabelTrue cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 10);
						
					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "LabelTrue contains illegal characters.");
						goto FAIL;
					}

					try
					{
						labelTrue = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "LabelTrue value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (labelTrue < -10.0f || labelTrue > 10.0f)
					{
						msg = CheckMsg(line, col, "LabelTrue value must be int the range [-10-10]");
						goto FAIL;
					}
				}
				else if (strLine.rfind("LabelFalse=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "LabelFalse cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "LabelFalse cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);
						
					if (params.find_first_not_of(".-eE0123456789") != std::string::npos)
					{
						msg = CheckMsg(line, col, "LabelFalse contains illegal characters.");
						goto FAIL;
					}

					try
					{
						labelFalse = std::stof(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "LabelFalse value not recognized." + nwl + exception.what());
						goto FAIL;

					}

					if (labelFalse < -10.0f || labelFalse > 10.0f)
					{
						msg = CheckMsg(line, col, "LabelFalse value must be int the range [-10-10]");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Cost=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Cost cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "Cost cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 5);

					auto ok = false;
					auto costs = magic_enum::enum_names<Costs>();
					for (auto cost : costs)
						if (params == std::string(cost))
							ok = true;
					if (!ok)
					{
						msg = CheckMsg(line, col, "Cost is not recognized.");
						goto FAIL;
					}

					auto cost = magic_enum::enum_cast<Costs>(params);
					if (cost.has_value())
						costFunction = cost.value();
					else
					{
						msg = CheckMsg(line, col, "Cost is not recognized.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Activation=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Activation cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Activation)
					{
						msg = CheckMsg(line, col, "Activation cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 11);

					auto ok = false;
					auto activations = magic_enum::enum_names<Activations>();
					for (auto activation : activations)
						if (params == std::string(activation))
							ok = true;
					if (!ok)
					{
						msg = CheckMsg(line, col, "Activation is not recognized.");
						goto FAIL;
					}

					auto activation = magic_enum::enum_cast<Activations>(params);
					if (activation.has_value())
						activationFunction = activation.value();
					else
					{
						msg = CheckMsg(line, col, "Activation is not recognized.");
						goto FAIL;
					}

					switch (activationFunction)
					{
					case Activations::FTS:
					    alpha = Float(-0.2);
						break;
					case Activations::BoundedRelu:
						alpha = Float(6);
						break;
					case Activations::Clip:
					case Activations::Linear:
						alpha = Float(1);
						beta = Float(0);
						break;
					case Activations::Elu:
					case Activations::Swish:
						alpha = Float(1);
						break;
					case Activations::HardLogistic:
					case Activations::Log:
					case Activations::LogSoftmax:
					case Activations::Softmax:
						labelTrue = Float(1);
						labelFalse = Float(0);
						break;
					case Activations::Pow:
						alpha = Float(1);
						beta = Float(1);
						break;
					case Activations::PRelu:
						alpha = Float(0.25);
						break;
					case Activations::Relu:
						alpha = Float(0);
						break;
					default:
						alpha = Float(0);
						beta = Float(0);
					}
				}
				else if (strLine.rfind("Channels=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Channels cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::ChannelZeroPad && layerType != LayerTypes::Dense && layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::Cost)
					{
						msg = CheckMsg(line, col, "Channels cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
						goto FAIL;
					}

					params = strLine.erase(0, 9);

					try
					{
						c = std::stoul(params);
					}
					catch (std::exception exception)
					{
						msg = CheckMsg(line, col, "Channels value not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (c == 0)
					{
						msg = CheckMsg(line, col, "Channels value cannot be zero.");
						goto FAIL;
					}
				}
				else if (strLine.rfind("Kernel=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Kernel cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
					{
						msg = CheckMsg(line, col, "Kernel cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
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
						msg = CheckMsg(line, col, "Kernel not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "Kernel must have two values.");
						goto FAIL;
					}

					if (values[0] == 0 || values[1] == 0)
					{
						msg = CheckMsg(line, col, "Kernel values cannot be zero.");
						goto FAIL;
					}

					kernelH = values[0];
					kernelW = values[1];
				}
				else if (strLine.rfind("Dilation=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Dilation cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution)
					{
						msg = CheckMsg(line, col, "Stride cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
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
						msg = CheckMsg(line, col, "Dilation not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "Dilation must have two values.");
						goto FAIL;
					}

					if (values[0] == 0 || values[1] == 0)
					{
						msg = CheckMsg(line, col, "Dilation values cannot be zero.");
						goto FAIL;
					}

					dilationH = values[0];
					dilationW = values[1];
				}
				else if (strLine.rfind("Stride=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Stride cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
					{
						msg = CheckMsg(line, col, "Stride cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
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
						msg = CheckMsg(line, col, "Stride not recognized." + nwl + exception.what());
						goto FAIL;
					}

					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "Stride must have two values.");
						goto FAIL;
					}

					if (values[0] == 0 || values[1] == 0)
					{
						msg = CheckMsg(line, col, "Stride values cannot be zero.");
						goto FAIL;
					}

					strideH = values[0];
					strideW = values[1];
				}
				else if (strLine.rfind("Pad=") == 0)
				{
					if (isModel)
					{
						msg = CheckMsg(line, col, "Pad cannot be specified in a model.");
						goto FAIL;
					}

					if (layerType != LayerTypes::Convolution && layerType != LayerTypes::ConvolutionTranspose && layerType != LayerTypes::DepthwiseConvolution && layerType != LayerTypes::PartialDepthwiseConvolution && layerType != LayerTypes::AvgPooling && layerType != LayerTypes::MaxPooling)
					{
						msg = CheckMsg(line, col, "Pad cannot be specified in a " + std::string(magic_enum::enum_name<LayerTypes>(layerType)) + " layer.");
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
						msg = CheckMsg(line, col, "Pad not recognized." + nwl + exception.what());
						goto FAIL;
					}
					
					if (values.size() != 2)
					{
						msg = CheckMsg(line, col, "Pad must have two values.");
						goto FAIL;
					}

					padD = 0;
					padH = values[0];
					padW = values[1];
				}
				else
				{
					msg = CheckMsg(line, col, "Unrecognized tokens: " + strLine);
					goto FAIL;
				}
			}
				
			if (layerType == LayerTypes::Cost)
			{
				if (c != classes)
				{
					msg = CheckMsg(line, col, "Cost layers has not the same number of channels as the dataset: " + std::to_string(classes));
					goto FAIL;
				}

				model->Layers.push_back(std::make_unique<Cost>(model->Device, model->Format, layerNames[model->Layers.size()].first, costFunction, groupIndex, labelIndex, c, model->GetLayerInputs(inputsStr), labelTrue, labelFalse, weight, epsSpecified ? eps : Float(0)));
				model->CostLayers.push_back(dynamic_cast<Cost*>(model->Layers[model->Layers.size() - 1].get()));
				model->CostFuction = costFunction;
			}

			{
				auto unreferencedLayers = model->SetRelations();

				if (unreferencedLayers.size() > 0)
				{
					auto l = unreferencedLayers[0];
					for (auto t : layerNames)
						if (t.first == l->Name)
							line = t.second;

					msg = CheckMsg(line, col, "Layer " + l->Name + " never referenced.");
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
				msg = CheckMsg(line, col, "A Cost layer is missing in the model");
				goto FAIL;
			}

			for (auto l : model->CostLayers)
				if (model->GetLayerOutputs(l).size() > 0)
				{
					for (auto t : layerNames)
						if (t.first == l->Name)
							line = t.second;

					msg = CheckMsg(line, col, "Cost Layer " + l->Name + " is referenced.");
					goto FAIL;
				}

			if (model->Layers.back()->LayerType != LayerTypes::Cost)
			{
				msg = CheckMsg(line, col, "Last layer must of type Cost.");
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

			std::setlocale(LC_ALL, userLocale);
            
			msg = CheckMsg(0, 0, "No issues found", false);	// All checks have passed

			return model;

		FAIL:
			if (model != nullptr)
			{
				model->~Model();
				model = nullptr;
			}

			std::setlocale(LC_ALL, userLocale);
           
			return nullptr;
		}

		static bool CheckDefinition(std::string& definition, CheckMsg& checkMsg)
		{
			definition = Normalize(definition);

			Read(definition, checkMsg, true);

			return checkMsg.Error;
		}

		static Model* ReadDefinition(const std::string& definition, Dataprovider* dataprovider, CheckMsg& checkMsg)
		{
			Model* model = Read(Normalize(definition), checkMsg, false, dataprovider);

			if (checkMsg.Error)
			{

			}

			return model;
		}

		static Model* LoadDefinition(const std::string& fileName, Dataprovider* dataprovider, CheckMsg& checkMsg)
		{
			Model* model = nullptr;

			auto file = std::ifstream(fileName);
			if (!file.bad() && file.is_open())
			{
				std::stringstream stream;
				stream << file.rdbuf();
				const auto buffer = stream.str();
				file.close();
				model = ReadDefinition(buffer, dataprovider, checkMsg);
			}

			return model;
		}
	};
}