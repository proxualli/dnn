#pragma once
#include "Layer.h"

namespace dnn
{
	class Min final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		
	public:
		Min(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Min, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() == 2);

			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
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

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_min, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *InputLayer->DstMemDesc), Device.engine));
#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (training)
			{
#ifdef DNN_LEAN
				DNN_UNREF_PAR(batchSize);

#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#endif
				Device.stream.wait();
#else
				const auto size = IsPlainFormat() ? CDHW() : PaddedCDHW();
				const auto part = GetVectorPart(size);
				const auto threads = GetThreads(batchSize * size);

				for_i(batchSize, threads, [=](UInt n)
				{
					const auto start = n * size;
					const auto end = start + part;
					const VecFloat vecZero = VecFloat(0);
					VecFloat In0, In1;
					for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
					{
						In0.load_a(&Inputs[0]->Neurons[cdhw]);
						In1.load_a(&Inputs[1]->Neurons[cdhw]);
						min(In0, In1).store_a(&Neurons[cdhw]);
						vecZero.store_nt(&NeuronsD1[cdhw]);
					}
					for (auto cdhw = end; cdhw < start + size; cdhw++)
					{
						Neurons[cdhw] = std::min(Inputs[0]->Neurons[cdhw], Inputs[1]->Neurons[cdhw]);
						NeuronsD1[cdhw] = 0;
					}
				});
#endif
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, Inputs[0]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, Inputs[1]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto size = IsPlainFormat() ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);
			const auto threads = GetThreads(batchSize * size);

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				VecFloat In0, In1, D1;
				for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
				{
					In0.load_a(&InputsOriginal[0]->Neurons[cdhw]);
					In1.load_a(&InputsOriginal[1]->Neurons[cdhw]);
					D1.load_a(&NeuronsD1[cdhw]);

					if_add(In0 <= In1, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw]), D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);
					if_add(In0 > In1, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw]), D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
				}
				for (auto cdhw = part; cdhw < size; cdhw++)
				{
					Inputs[0]->NeuronsD1[cdhw] += InputsOriginal[0]->Neurons[cdhw] <= InputsOriginal[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : 0;
					Inputs[1]->NeuronsD1[cdhw] += InputsOriginal[0]->Neurons[cdhw] <= InputsOriginal[1]->Neurons[cdhw] ? 0 : NeuronsD1[cdhw];
				}
			}
			else
			{
#endif
				for_i(batchSize, threads, [=](UInt b)
				{
					const auto start = b * size;
					const auto end = start + part;

					VecFloat In0, In1, D1;
					for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
					{
						In0.load_a(&InputsOriginal[0]->Neurons[cdhw]);
						In1.load_a(&InputsOriginal[1]->Neurons[cdhw]);
						D1.load_a(&NeuronsD1[cdhw]);

						if_add(In0 <= In1, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw]), D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);
						if_add(In0 > In1, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw]), D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					for (auto cdhw = part; cdhw < size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += InputsOriginal[0]->Neurons[cdhw] <= InputsOriginal[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : 0;
						Inputs[1]->NeuronsD1[cdhw] += InputsOriginal[0]->Neurons[cdhw] <= InputsOriginal[1]->Neurons[cdhw] ? 0 : NeuronsD1[cdhw];
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