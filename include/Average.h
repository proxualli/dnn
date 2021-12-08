#pragma once
#include "Layer.h"

namespace dnn
{
	class Average final : public Layer
	{
	private:
		std::vector<Float> Scales;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::sum::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::sum> fwd;
#endif

	public:
		const Float Scale;

		Average(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Average, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Scale(Float(1) / Inputs.size())
		{
			assert(Inputs.size() > 1);

			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}

			Scales = std::vector<Float>(Inputs.size(), Scale);
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

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

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(ChosenFormat == GetDataFmt(*Inputs[i]->DstMemDesc));
				if (ChosenFormat != GetDataFmt(*Inputs[i]->DstMemDesc))
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Inputs[i]->Name);
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>();
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				if (Inputs[i]->DstMemDesc->data.ndims == 2)
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, ChosenFormat));
				else
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::sum(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW);
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
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = (size / VectorSize) * VectorSize;
			const auto elements = batchSize * size;
			const auto threads = elements < 2097152ull ? ULTRA_LIGHT_COMPUTE : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
					for (auto i = 0ull; i < inputs; i++)
						mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[i]->NeuronsD1[cdhw])).store_a(&Inputs[i]->NeuronsD1[cdhw]);

				for (auto cdhw = part; cdhw < size; cdhw++)
					for (auto i = 0ull; i < inputs; i++)
						Inputs[i]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
			}
			else
			{
#endif
				switch (inputs)
				{
				case 2:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						for (auto cdhw = start; cdhw < end; cdhw+=VectorSize)
						{
							mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 3:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[2]->NeuronsD1[cdhw])).store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
						}
					});
				}
				break;

				default:
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							for (auto i = 0ull; i < inputs; i++)
								mul_add(VecFloat().load_a(&NeuronsD1[cdhw]), Scale, VecFloat().load_a(&Inputs[i]->NeuronsD1[cdhw])).store_a(&Inputs[i]->NeuronsD1[cdhw]);
							
						for (auto cdhw = end; cdhw < start + size; cdhw++)
							for (auto i = 0ull; i < inputs; i++)
								Inputs[i]->NeuronsD1[cdhw] += Scale * NeuronsD1[cdhw];
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