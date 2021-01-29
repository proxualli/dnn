#pragma once
#include "Layer.h"

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
				throw std::invalid_argument("input not splittable in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				chosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					chosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
					if (chosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					chosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto groupC = (Group - 1) * (plain ? C  : PaddedC);
			const auto strideHW = HW * VectorSize;
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (training)
				{
					if (!plain)
					{
						const auto vecZero = VecFloat(0);
						VecFloat In;		
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = (c + groupC) * HW;
							const auto outputOffset = c * HW;
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[hw + inputOffset]);
								In.store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif // DNN_LEAN
							}
						}
					}
					else
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (c + groupC) * HW;
							const auto outputOffset = c * HW;
							for (auto hw = 0ull; hw < HW; hw++)
							{
								Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
 								NeuronsD1[hw + outputOffset] = Float(0);
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
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = (c + groupC) * HW;
							const auto outputOffset = c * HW;
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[hw + inputOffset]);
								In.store_a(&Neurons[hw + outputOffset]);
							}
						}
					}
					else
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (c + groupC) * HW;
							const auto outputOffset = c * HW;
							for (auto hw = 0ull; hw < HW; hw++)
								Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
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
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto vecZero = VecFloat(0); 
							VecFloat In;						
 							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto inputOffset = n * InputLayer->PaddedCDHW + (c + groupC) * HW;
								const auto outputOffset = n * PaddedCDHW + c * HW;
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[hw + inputOffset]);
									In.store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif // DNN_LEAN
								}
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](size_t n)
						{
							for (auto c = 0ull; c < C; c ++)
							{
								const auto inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
								const auto outputOffset = n * CDHW + c * HW;
								for (auto hw = 0ull; hw < HW; hw++)
								{
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[hw + outputOffset] = Float(0);
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
						for_i(batchSize, threads, [=](size_t n)
						{
							VecFloat In;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto inputOffset = n * InputLayer->PaddedCDHW + (c + groupC) * HW;
								const auto outputOffset = n * PaddedCDHW + c * HW;
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[hw + inputOffset]); 
									In.store_a(&Neurons[hw + outputOffset]);
								}
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](size_t n)
						{
							for (auto c = 0ull; c < C; c ++)
							{
								const auto inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
								const auto outputOffset = n * CDHW + c * HW;
								for (auto hw = 0ull; hw < HW; hw++)
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
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
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto groupC = (Group - 1) * (plain ? C : PaddedC);
			const auto strideHW = HW * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat In, D1;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						const auto inputOffset = (c + groupC) * HW;
						const auto outputOffset = c * HW;
						for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
						{
							D1.load_a(&NeuronsD1[hw + outputOffset]);
							In.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							In += D1;
							In.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
						}
					}
				}
				else
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto inputOffset = (c + groupC) * HW;
						const auto outputOffset = c * HW;
						for (auto hw = 0ull; hw < HW; hw++)
							InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
					}
				}
			}
			else
			{
#endif
				if (!plain)
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						VecFloat In, D1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
							const auto outputOffset = n * PaddedCDHW + c * HW;
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								D1.load_a(&NeuronsD1[hw + outputOffset]);
								In.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								In += D1;
								In.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
							const auto outputOffset = n * CDHW + c * HW;
							for (auto hw = 0ull; hw < HW; hw++)
								InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
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
