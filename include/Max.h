#pragma once
#include "Layer.h"

namespace dnn
{
	class Max final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		
	public:
		Max(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Max, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
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

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_max, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *InputLayer->DstMemDesc), Device.engine));
#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			if (training)
			{
#ifdef DNN_LEAN
				DNN_UNREF_PAR(batchSize);

#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#endif
				Device.stream.wait();
#else
				const auto plain = IsPlainFormat();
				const auto size = plain ? CDHW : PaddedCDHW;
				const auto part = (size / VectorSize) * VectorSize;
				const auto elements = size * batchSize;
				const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
						
				for_i(batchSize, threads, [=](size_t n)
				{
					const auto start = n * size;
					const auto end = start + part;
					const VecFloat vecZero = VecFloat(0);
					VecFloat In0, In1;
					for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
					{
						In0.load_a(&Inputs[0]->Neurons[cdhw]);
						In1.load_a(&Inputs[1]->Neurons[cdhw]);
						max(In0,In1).store_a(&Neurons[cdhw]);
						vecZero.store_nt(&NeuronsD1[cdhw]);
					}
					for (auto cdhw = end; cdhw < start + size; cdhw++)
					{
						Neurons[cdhw] = std::max(Inputs[0]->Neurons[cdhw], Inputs[1]->Neurons[cdhw]);
						NeuronsD1[cdhw] = 0;
					}
				});
#endif
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = (size / VectorSize) * VectorSize;
			const auto elements = size * batchSize;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;		
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				VecFloat In0, In1, D1;
				for (auto cdhw = 0ull; cdhw < part; cdhw+=VectorSize)
				{
					In0.load_a(&Inputs[0]->Neurons[cdhw]);
					In1.load_a(&Inputs[1]->Neurons[cdhw]);
					D1.load_a(&NeuronsD1[cdhw]);
					if_add(In0 >= In1, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw]), D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);
					if_add(In0 < In1, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw]), D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
				}
				for (auto cdhw = part; cdhw < size; cdhw++)
				{
					Inputs[0]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : 0;
					Inputs[1]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? 0 : NeuronsD1[cdhw];
				}
			}
			else
			{
#endif
				for_i(batchSize, threads, [=](size_t n)
				{
					const auto start = n * size;
					const auto end = start + part;

					VecFloat In0, In1, D1;
					for (auto cdhw = start; cdhw < end; cdhw+=VectorSize)
					{
						In0.load_a(&Inputs[0]->Neurons[cdhw]);
						In1.load_a(&Inputs[1]->Neurons[cdhw]);
						D1.load_a(&NeuronsD1[cdhw]);

						if_add(In0 >= In1, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw]), D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);
						if_add(In0 < In1, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw]), D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					for (auto cdhw = end; cdhw < start + size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : 0;
						Inputs[1]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? 0 : NeuronsD1[cdhw];
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
