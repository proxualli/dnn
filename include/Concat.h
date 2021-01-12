#pragma once
#include "Layer.h"

namespace dnn
{
	class Concat final : public Layer
	{
	private:
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::concat::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::concat> fwd;
#endif

	public:
		static auto InputChannels(const std::vector<Layer*>& inputs)
		{
			auto channels = 0ull;
			for (auto layer : inputs)
				channels += layer->C;

			return channels;
		}

		Concat(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Concat, 0, 0, InputChannels(inputs), inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);
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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				if (Format == dnnl::memory::format_tag::any)
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

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer");
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>();
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				if (Inputs[i]->DstMemDesc->data.ndims == 2)
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, chosenFormat));
				else
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::concat::primitive_desc>(dnnl::concat::primitive_desc(*DstMemDesc, 1, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::concat>(dnnl::concat(*fwdDesc));
#endif
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
			DNN_UNREF_PAR(training);
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				auto channelOffset = 0ull;
				for (auto input = 0ull; input < Inputs.size(); input++)
				{
					for (auto c = channelOffset; c < channelOffset + Inputs[input]->C; c++)
					{
						const auto inputIndex = (c - channelOffset) * HW;
						const auto outputIndex = c * HW;
						for (auto hw = 0ull; hw < HW; hw++)
							Inputs[input]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
					}
					channelOffset += Inputs[input]->C;
				}
			}
			else
			{
#endif
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto outputSampleOffset = b * (plain ? CDHW : PaddedCDHW);
					auto channelOffset = 0ull;
					for (auto input = 0ull; input < Inputs.size(); input++)
					{
						const auto inputSampleOffset = b * (plain ? Inputs[input]->CDHW : Inputs[input]->PaddedCDHW);
						for (auto c = channelOffset; c < channelOffset + Inputs[input]->C; c++)
						{
							const auto inputIndex = ((c - channelOffset) * HW) + inputSampleOffset;
							const auto outputIndex = (c * HW) + outputSampleOffset;
							for (auto hw = 0ull; hw < HW; hw++)
								Inputs[input]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
						}
						channelOffset += Inputs[input]->C;
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
