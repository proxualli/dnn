#pragma once
#include "Layer.h"
#include <stdexcept>

namespace dnn
{
	class ChannelMultiply final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::binary> fwd;
		std::unordered_map<int, dnnl::memory> fwdArgs;

	public:
		ChannelMultiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, dnn::Inputs inputs) :
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
			chosenFormat = BlockedFmt;

			if (!IsBlockedDataFmt(*Inputs[0]->DstMemDesc) || !IsBlockedDataFmt(*Inputs[1]->DstMemDesc))
				throw std::runtime_error("incompatible format used");

			dnnl::memory::desc memDesc = dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat);

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_mul, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, memDesc), Device.engine));
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			fwd->execute(Device.stream, fwdArgs);
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

			const size_t strideH = W * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto c = 0ull; c < PaddedC; c += VectorSize)
				{
					const auto offsetC = c * HW;

					for (auto h = 0ull; h < H; h++)
					{
						const auto offsetH = offsetC + h * strideH;

						for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
						{
							const auto& neuronsD1 = VecFloat().load_a(&NeuronsD1[w]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[1]->Neurons[c]), VecFloat().load_a(&Inputs[0]->NeuronsD1[w])).store_a(&Inputs[0]->NeuronsD1[w]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[0]->Neurons[w]), VecFloat().load_a(&Inputs[1]->NeuronsD1[c])).store_a(&Inputs[1]->NeuronsD1[c]);
						}
					}
				}
			}
			else
			{
#endif
				for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
				{
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						const auto offsetC = n * PaddedCDHW + c * HW;
						const auto channelOffset = n * PaddedC + c;

						for (auto h = 0ull; h < H; h++)
						{
							const auto offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								const auto& neuronsD1 = VecFloat().load_a(&NeuronsD1[w]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[1]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[0]->NeuronsD1[w])).store_a(&Inputs[0]->NeuronsD1[w]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[0]->Neurons[w]), VecFloat().load_a(&Inputs[1]->NeuronsD1[channelOffset])).store_a(&Inputs[1]->NeuronsD1[channelOffset]);
							}
						}
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
