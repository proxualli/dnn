#pragma once
#include "Layer.h"

namespace dnn
{
	class Add final : public Layer
	{
	private:
		std::vector<Float> Scales;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::sum::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::sum> fwd;

	public:
		Add(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Add, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);

			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}

			Scales = std::vector<Float>(Inputs.size(), Float(1));
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

			for (size_t i = 1; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in Add layer");
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>(Inputs.size());
			for (size_t i = 0; i < Inputs.size(); i++)
				srcsMemsDesc[i] = *Inputs[i]->DstMemDesc;

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (size_t i = 0; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			if (training)
			{
				const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;
				const auto part = (size / VectorSize) * VectorSize;
				const auto inputs = Inputs.size();

				switch (inputs)
				{
				case 2:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;
						VecFloat zero = VecFloat(0);
						VecFloat InA, InB;
						for (auto n = start; n < part; n += VectorSize)
						{
							InA.load_a(&Inputs[0]->Neurons[n]);
							InB.load_a(&Inputs[1]->Neurons[n]);
							(InA + InB).store_a(&Neurons[n]);
							zero.store_nt(&NeuronsD1[n]);
						}
						for (auto n = part; n < start + size; n++)
						{
							Neurons[n] = Inputs[0]->Neurons[n] + Inputs[1]->Neurons[n];
							NeuronsD1[n] = 0;
						}
					});
				}
				break;

				case 3:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat zero = VecFloat(0);
						VecFloat InA, InB, InC;
						for (auto n = start; n < part; n += VectorSize)
						{
							InA.load_a(&Inputs[0]->Neurons[n]);
							InB.load_a(&Inputs[1]->Neurons[n]);
							InC.load_a(&Inputs[2]->Neurons[n]);
							(InA + InB + InC).store_a(&Neurons[n]);
							zero.store_nt(&NeuronsD1[n]);
						}
						for (auto n = part; n < start + size; n++)
						{
							Neurons[n] = Inputs[0]->Neurons[n] + Inputs[1]->Neurons[n] + Inputs[2]->Neurons[n];
							NeuronsD1[n] = 0;
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat zero = VecFloat(0);
						VecFloat InA, InB, InC, InD;
						for (auto n = start; n < part; n += VectorSize)
						{
							InA.load_a(&Inputs[0]->Neurons[n]);
							InB.load_a(&Inputs[1]->Neurons[n]);
							InC.load_a(&Inputs[2]->Neurons[n]);
							InD.load_a(&Inputs[3]->Neurons[n]);
							(InA + InB + InC + InD).store_a(&Neurons[n]);
							zero.store_nt(&NeuronsD1[n]);
						}
						for (auto n = part; n < start + size; n++)
						{
							Neurons[n] = Inputs[0]->Neurons[n] + Inputs[1]->Neurons[n] + Inputs[2]->Neurons[n] + Inputs[3]->Neurons[n];
							NeuronsD1[n] = 0;
						}
					});
				}
				break;

				default:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;
							
#pragma omp simd
						for (auto n = start; n < end; n++)
						{
							NeuronsD1[n] = 0;
							for (auto y = 0ull; y < inputs; y++)
								Neurons[n] += Inputs[y]->Neurons[n];
						}
					});
				}
				break;
				}
			}
			else
			{
				fwd->execute(Device.stream, fwdArgs);
				Device.stream.wait();

#ifndef DNN_LEAN
				if (training)
					ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
				DNN_UNREF_PAR(batchSize);
				DNN_UNREF_PAR(training);
#endif
			}
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;
			const auto part = (size / VectorSize) * VectorSize;
			const auto inputs = Inputs.size();
			

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				switch (inputs)
				{
				case 2:
				{
					VecFloat In, D1;
					for (auto n = 0; n < part; n += VectorSize)
					{
						D1.load_a(&NeuronsD1[n]);

						In.load_a(&Inputs[0]->NeuronsD1[n]);
						In += D1;
						In.store_a(&Inputs[0]->NeuronsD1[n]);

						In.load_a(&Inputs[1]->NeuronsD1[n]);
						In += D1;
						In.store_a(&Inputs[1]->NeuronsD1[n]);
					}
					for (auto n = part; n < size; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
					}
				}
				break;

				default:
				{
					for (auto i = 0ull; i < inputs; i++)
						for (auto n = 0ull; n < size; n++)
							Inputs[i]->NeuronsD1[n] += NeuronsD1[n];

				}
				}
			}
			else
			{
#endif
				switch (Inputs.size())
				{
				case 2:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto n = start; n < end; n += VectorSize)
						{
							D1.load_a(&NeuronsD1[n]);

							In.load_a(&Inputs[0]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[n]);

							In.load_a(&Inputs[1]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[1]->NeuronsD1[n]);
						}
						for (auto n = end; n < start + size; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 3:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto n = start; n < end; n += VectorSize)
						{
							D1.load_a(&NeuronsD1[n]);

							In.load_a(&Inputs[0]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[n]);

							In.load_a(&Inputs[1]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[1]->NeuronsD1[n]);

							In.load_a(&Inputs[2]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[2]->NeuronsD1[n]);
						}
						for (auto n = end; n < start + size; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto n = start; n < end; n += VectorSize)
						{
							D1.load_a(&NeuronsD1[n]);

							In.load_a(&Inputs[0]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[n]);

							In.load_a(&Inputs[1]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[1]->NeuronsD1[n]);

							In.load_a(&Inputs[2]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[2]->NeuronsD1[n]);

							In.load_a(&Inputs[3]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[3]->NeuronsD1[n]);
						}
						for (auto n = end; n < start + size; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 5:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto n = start; n < end; n += VectorSize)
						{
							D1.load_a(&NeuronsD1[n]);

							In.load_a(&Inputs[0]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[n]);

							In.load_a(&Inputs[1]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[1]->NeuronsD1[n]);

							In.load_a(&Inputs[2]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[2]->NeuronsD1[n]);

							In.load_a(&Inputs[3]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[3]->NeuronsD1[n]);

							In.load_a(&Inputs[4]->NeuronsD1[n]);
							In += D1;
							In.store_a(&Inputs[4]->NeuronsD1[n]);
						}
						for (auto n = end; n < start + size; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 6:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 7:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				case 8:
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[7]->NeuronsD1[n] += NeuronsD1[n];
						}
					});
				}
				break;

				default:
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[7]->NeuronsD1[n] += NeuronsD1[n];
						}
						for (auto i = 8ull; i < inputs; i++)
							for (auto n = start; n < end; n++)
								Inputs[i]->NeuronsD1[n] += NeuronsD1[n];
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
