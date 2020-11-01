#include "Model.h"

namespace dnn
{
	ChannelMultiply::ChannelMultiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
		Layer(device, format, name, LayerTypes::ChannelMultiply, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
	{
		assert(Inputs.size() == 2);

		assert(Inputs[0]->C == Inputs[1]->C);
		assert(Inputs[1]->H == 1);
		assert(Inputs[1]->W == 1);
	}

	std::string ChannelMultiply::GetDescription() const
	{
		return GetDescriptionHeader();
	}

	size_t ChannelMultiply::FanIn() const
	{
		return 1;
	}

	size_t ChannelMultiply::FanOut() const
	{
		return 1;
	}

	void ChannelMultiply::InitializeDescriptors(const size_t batchSize)
	{
		if (!IsBlockedDataFmt(*Inputs[0]->DstMemDesc) || !IsBlockedDataFmt(*Inputs[1]->DstMemDesc))
			throw std::exception("incompatible format used");

		dnnl::memory::desc memDesc = dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format);

		fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_mul, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, memDesc), Device.first));
		fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
		
		DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
		DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

		fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.first, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.first, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.first, Neurons.data()) } };
	}

	void ChannelMultiply::ForwardProp(const size_t batchSize, const bool training)
	{
		fwd->execute(Device.second, fwdArgs);
		Device.second.wait();

#ifndef DNN_LEAN
		if (training)
			ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
		DNN_UNREF_PAR(batchSize);
#endif
	}

	void ChannelMultiply::BackwardProp(const size_t batchSize)
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
						const auto neuronsD1 = VecFloat().load_a(&NeuronsD1[w]);
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
				for (auto c = 0ull; c < PaddedC; c+=VectorSize)
				{
					const auto offsetC = n * PaddedCDHW + c * HW;
					const auto channelOffset = n * PaddedC + c;

					for (auto h = 0ull; h < H; h++)
					{
						const auto offsetH = offsetC + h * strideH;

						for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
						{
							const auto neuronsD1 = VecFloat().load_a(&NeuronsD1[w]);
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
}