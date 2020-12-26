#pragma once
#include "Layer.h"

namespace dnn
{
	class Min final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::binary> fwd;

	public:
		Min(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Min, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() == 2);

			for (size_t i = 0; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}
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
			DNN_UNREF_PAR(batchSize);

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

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}
		
			for (size_t i = 1; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in Min layer");
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_min, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *InputLayer->DstMemDesc), Device.engine));
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
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
#endif

			const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto n = 0ull; n < size; n++)
				{
					Inputs[0]->NeuronsD1[n] += Inputs[0]->Neurons[n] <= Inputs[1]->Neurons[n] ? NeuronsD1[n] : 0;
					Inputs[1]->NeuronsD1[n] += Inputs[0]->Neurons[n] <= Inputs[1]->Neurons[n] ? 0 : NeuronsD1[n];
				}
			}
			else
			{
#endif
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * size;
					const auto end = start + size;

					VecFloat InA, InB, D1;
					for (auto n = start; n < end; n += VectorSize)
					{
						InA = VecFloat().load_a(&Inputs[0]->Neurons[n]);
						InB = VecFloat().load_a(&Inputs[1]->Neurons[n]);
						D1 = VecFloat().load_a(&NeuronsD1[n]);

						if_add(InA <= InB, VecFloat().load_a(&Inputs[0]->NeuronsD1[n]), D1).store_a(&Inputs[0]->NeuronsD1[n]);
						if_add(InA > InB, VecFloat().load_a(&Inputs[1]->NeuronsD1[n]), D1).store_a(&Inputs[1]->NeuronsD1[n]);
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
