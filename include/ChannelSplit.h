#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelSplit final : public Layer
	{
	public:
		const UInt Group;
		const UInt Groups;

		ChannelSplit(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt group, const UInt groups) :
			Layer(device, format, name, LayerTypes::ChannelSplit, 0, 0, inputs[0]->C / groups, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Group(group),
			Groups(groups)
		{
			assert(Inputs.size() == 1);
			assert(InputLayer->C % Groups == 0);

			if (InputLayer->C % Groups != 0)
				throw std::invalid_argument("input not splittable in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Groups:") + tab + std::to_string(Groups));
			description.append(nwl + std::string(" Group:") + dtab + std::to_string(Group));

			return description;
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
					if (ChosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto groupC = (Group - 1) * C;
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
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (c + groupC) * HW;
							const auto outputOffset = c * HW;
							for (auto hw = 0ull; hw < HW; hw++)
								Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
						}
				}				
			}
			else
			{
#endif
				if (training)
				{
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
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
					else
						for_i(batchSize, threads, [=](UInt n)
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
				else
				{
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
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
					else
						for_i(batchSize, threads, [=](UInt n)
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
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto groupC = (Group - 1) * C;
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
					for (auto c = 0ull; c < C; c++)
					{
						const auto inputOffset = (c + groupC) * HW;
						const auto outputOffset = c * HW;
						for (auto hw = 0ull; hw < HW; hw++)
							InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
					}
			}
			else
			{
#endif
				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat In, D1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = n * InputLayer->PaddedCDHW + (c + groupC) * HW;
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
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = n * InputLayer->CDHW + (c + groupC) * HW;
							const auto outputOffset = n * CDHW + c * HW;
							for (auto hw = 0ull; hw < HW; hw++)
								InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
						}
					});
#ifdef DNN_STOCHASTIC
			}
#endif

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}