#pragma once
#include "Layer.h"
#include <stdexcept>

namespace dnn
{
	class ChannelSplit final : public Layer
	{
	public:
		const size_t Group;
		const size_t Groups;

		ChannelSplit(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t group, const size_t groups) :
			Layer(device, format, name, LayerTypes::ChannelSplit, 0, 0, inputs[0]->PaddedC / groups, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Group(group),
			Groups(groups)
		{
			assert(Inputs.size() == 1);
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Groups:") + tab + std::to_string(Groups));
			description.append(nwl + std::string(" Group:") + dtab + std::to_string(Group));

			return description;
		}

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			if (InputLayer->PaddedC % Groups != 0)
				throw std::runtime_error("input not splittable in ChannelSplit");

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				chosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (GetDataFmt(*InputLayer->DstMemDesc) != BlockedFmt)
					throw std::runtime_error("Blocked format expected in ChannelSplit");

				chosenFormat = BlockedFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto groupC = (Group - 1) * (plain ? C  : PaddedC);
			const auto strideH = HW * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (training)
				{
					if (!plain)
					{
						const auto vecZero = VecFloat(0);
						VecFloat In;

						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							inputOffset = (c + groupC) * HW;
							outputOffset = c * HW;
							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[w + inputOffset]);
								In.store_a(&Neurons[w + outputOffset]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w + outputOffset]);
#endif // DNN_LEAN
							}
						}
					}
					else
					{
						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < C; c++)
						{
							inputOffset = (c + groupC) * HW;
							outputOffset = c * HW;
							for (auto w = 0ull; w < HW; w++)
							{
								Neurons[w + outputOffset] = InputLayer->Neurons[w + inputOffset]);
#ifndef DNN_LEAN
 								NeuronsD1[w + outputOffset] = Float(0);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
					if (!plain)
					{
						const auto vecZero = VecFloat(0);
						VecFloat In;

						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							inputOffset = (c + groupC) * HW;
							outputOffset = c * HW;
							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[w + inputOffset]);
								In.store_a(&Neurons[w + outputOffset]);
							}
						}
					}
					else
					{
						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < C; c++)
						{
							inputOffset = (c + groupC) * HW;
							outputOffset = c * HW;
							for (auto w = 0ull; w < HW; w++)
								Neurons[w + outputOffset] = InputLayer->Neurons[w + inputOffset]);
						}
					}
				}				
			}
			else
			{
#endif
                if (training)
				{
					if (!plain)
					{
						for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
						{
							const auto vecZero = VecFloat(0); 
							VecFloat In;
														
 							size_t inputOffset, outputOffset;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								inputOffset = n * InputLayer->PaddedCDHW + (c + groupC) * HW;
								outputOffset = n * PaddedCDHW + c * HW;

								for (auto w = 0ull; w < strideH; w += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[w + inputOffset]);
									In.store_a(&Neurons[w + outputOffset]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w + outputOffset]);
#endif // DNN_LEAN
								}
							}
						});
					}
					else
					{
						for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
						{
							size_t inputOffset, outputOffset;
							for (auto c = 0ull; c < C; c ++)
							{
								inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
								outputOffset = n * CDHW + c * HW;

								for (auto w = 0ull; w < HW; w++)
								{
									Neurons[w + outputOffset] = InputLayer->Neurons[w + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[w + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
						});
					}
				}
				else
				{
					if (!plain)
					{
						for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
						{
							VecFloat In;
							size_t inputOffset, outputOffset;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								inputOffset = n * InputLayer->PaddedCDHW + (c + groupC) * HW;
								outputOffset = n * PaddedCDHW + c * HW;

								for (auto w = 0ull; w < strideH; w += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[w + inputOffset]); 
									In.store_a(&Neurons[w + outputOffset]);
								}
							}
						});
					}
					else
					{
						for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
						{
							size_t inputOffset, outputOffset;
							for (auto c = 0ull; c < C; c ++)
							{
								inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
								outputOffset = n * CDHW + c * HW;

								for (auto w = 0ull; w < HW; w++)
									Neurons[w + outputOffset] = InputLayer->Neurons[w + inputOffset];
							}
						});
					}
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto groupC = (Group - 1) * (plain ? C : PaddedC);
			const auto strideH = HW * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					size_t inputOffset, outputOffset;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						inputOffset = (c + groupC) * HW;
						outputOffset = c * HW;

						for (auto w = 0ull; w < strideH; w += VectorSize)
							(VecFloat().load_a(&InputLayer->NeuronsD1[w + inputOffset]) += VecFloat().load_a(&NeuronsD1[w + outputOffset])).store_a(&InputLayer->NeuronsD1[w + inputOffset]);
					}
				}
				else
				{
					size_t inputOffset, outputOffset;
					for (auto c = 0ull; c < C; c++)
					{
						inputOffset = (c + groupC) * HW;
						outputOffset = c * HW;

						for (auto w = 0ull; w < HW; w++)
							InputLayer->NeuronsD1[w + inputOffset] += NeuronsD1[w + outputOffset];
					}
				}
			}
			else
			{
#endif
				if (!plain)
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
							outputOffset = n * PaddedCDHW + c * HW;

							for (auto w = 0ull; w < strideH; w += VectorSize)
								(VecFloat().load_a(&InputLayer->NeuronsD1[w + inputOffset]) += VecFloat().load_a(&NeuronsD1[w + outputOffset])).store_a(&InputLayer->NeuronsD1[w + inputOffset]);
						}
					});
				}
				else
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						size_t inputOffset, outputOffset;
						for (auto c = 0ull; c < C; c++)
						{
							inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
							outputOffset = n * CDHW + c * HW;

							for (auto w = 0ull; w < HW; w++)
								InputLayer->NeuronsD1[w + inputOffset] += NeuronsD1[w + outputOffset];
						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}
