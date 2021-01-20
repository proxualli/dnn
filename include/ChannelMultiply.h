#pragma once
#include "Layer.h"
#include <stdexcept>

namespace dnn
{
	class ChannelMultiply final : public Layer
	{
	private:
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif

	public:
		ChannelMultiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::ChannelMultiply, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() == 2);

			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[1]->H == 1);
			assert(Inputs[1]->W == 1);
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
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
			if (Format == dnnl::memory::format_tag::any)
			{
				chosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
				if (chosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
					throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_mul, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *DstMemDesc), Device.engine));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const size_t strideHW = HW * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat neuronsD1;
					size_t outputOffset;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						outputOffset = c * HW;
						for (auto w = 0ull; w < strideHW; w += VectorSize)
						{
							neuronsD1.load_a(&NeuronsD1[w + outputOffset]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[1]->Neurons[c]), VecFloat().load_a(&Inputs[0]->NeuronsD1[w + outputOffset])).store_a(&Inputs[0]->NeuronsD1[w + outputOffset]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[0]->Neurons[w + outputOffset]), VecFloat().load_a(&Inputs[1]->NeuronsD1[c])).store_a(&Inputs[1]->NeuronsD1[c]);
						}
					}
				}
				else
				{
					size_t outputOffset;
					for (auto c = 0ull; c < C; c++)
					{
						outputOffset = c * HW;
						for (auto w = 0ull; w < HW; w++)
						{
							Inputs[0]->NeuronsD1[w + outputOffset] += NeuronsD1[w + outputOffset] * Inputs[1]->Neurons[c];
							Inputs[1]->NeuronsD1[c] += NeuronsD1[w + outputOffset] * Inputs[0]->Neurons[w + outputOffset];
						}
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
						VecFloat neuronsD1;
						size_t outputOffset, channelOffset;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							outputOffset = n * PaddedCDHW + c * HW;
							channelOffset = n * PaddedC + c;
							for (auto w = 0ull; w < strideHW; w += VectorSize)
							{
								neuronsD1.load_a(&NeuronsD1[w + outputOffset]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[1]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[0]->NeuronsD1[w + outputOffset])).store_a(&Inputs[0]->NeuronsD1[w + outputOffset]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[0]->Neurons[w + outputOffset]), VecFloat().load_a(&Inputs[1]->NeuronsD1[channelOffset])).store_a(&Inputs[1]->NeuronsD1[channelOffset]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						size_t outputOffset, channelOffset;
						for (auto c = 0ull; c < C; c++)
						{
							outputOffset = n * CDHW + c * HW;
							channelOffset = n * C + c;
							for (auto w = 0ull; w < HW; w++)
							{
								Inputs[0]->NeuronsD1[w + outputOffset] += NeuronsD1[w + outputOffset] * Inputs[1]->Neurons[channelOffset];
								Inputs[1]->NeuronsD1[channelOffset] += NeuronsD1[w + outputOffset] * Inputs[0]->Neurons[w + outputOffset];
							}
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
