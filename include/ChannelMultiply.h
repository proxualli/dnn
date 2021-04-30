#pragma once
#include "Layer.h"

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
		auto GetFirt(const std::vector<Layer*>& inputs)
		{
			return (inputs[0]->H == 1 && inputs[0]->W == 1) ? Byte(1) : Byte(0);
		}
		auto GetSecond(const std::vector<Layer*>& inputs)
		{
			return (inputs[0]->H == 1 && inputs[0]->W == 1) ? Byte(0) : Byte(1);
		}

	public:
		const Byte first, second;

		ChannelMultiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::ChannelMultiply, 0, 0, inputs[GetFirt(inputs)]->C, inputs[GetFirt(inputs)]->D, inputs[GetFirt(inputs)]->H, inputs[GetFirt(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirt(inputs)),
			second(GetSecond(inputs))
		{
			assert(Inputs.size() == 2);

			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->H == 1 || Inputs[1]->H == 1);
			assert(Inputs[0]->W == 1 || Inputs[1]->W == 1);
			assert(Inputs[0]->H != 1 || Inputs[1]->H != 1);
			assert(Inputs[0]->W != 1 || Inputs[1]->W != 1);
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
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
			if (Format == dnnl::memory::format_tag::any)
			{
				ChosenFormat = GetDataFmt(*Inputs[first]->DstMemDesc);
				if (ChosenFormat != GetDataFmt(*Inputs[first]->DiffDstMemDesc))
					throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}
			else
				ChosenFormat = PlainFmt;

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_mul, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc), Device.engine));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroArray(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto strideHW = HW * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat neuronsD1;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						const auto outputOffset = c * HW;
						for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
						{
							neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[second]->Neurons[c]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
							mul_add(neuronsD1, VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[c])).store_a(&Inputs[second]->NeuronsD1[c]);
						}
					}
				}
				else
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto outputOffset = c * HW;
						for (auto hw = 0ull; hw < HW; hw++)
						{
							Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * Inputs[second]->Neurons[c];
							Inputs[second]->NeuronsD1[c] += NeuronsD1[hw + outputOffset] * Inputs[first]->Neurons[hw + outputOffset];
						}
					}
				}
			}
			else
			{
#endif
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat neuronsD1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = n * PaddedCDHW + c * HW;
							const auto channelOffset = n * PaddedC + c;
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
								mul_add(neuronsD1, VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[channelOffset])).store_a(&Inputs[second]->NeuronsD1[channelOffset]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = n * CDHW + c * HW;
							const auto channelOffset = n * C + c;
							for (auto hw = 0ull; hw < HW; hw++)
							{
								Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * Inputs[second]->Neurons[channelOffset];
								Inputs[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * Inputs[first]->Neurons[hw + outputOffset];
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
