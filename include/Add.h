#pragma once
#include "Layer.h"

namespace dnn
{
	class Add final : public Layer
	{
	private:
		std::vector<Float> scales;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::sum::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::sum> fwd;
#endif

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

			scales = std::vector<Float>(Inputs.size(), Float(1));
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
						throw std::invalid_argument("Src and Diff format are different inputD1 " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
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
					throw std::invalid_argument("Incompatible memory formats inputD1 " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer");
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>();
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				if (Inputs[i]->DstMemDesc->data.ndims == 2)
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, ChosenFormat));
				else
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, scales, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (training)
			{
#ifdef DNN_LEAN
				DNN_UNREF_PAR(batchSize);

#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::sum(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
#else
				const auto plain = IsPlainFormat();
				const auto size = plain ? CDHW() : PaddedCDHW();
				const auto part = GetVectorPart(size);
				const auto elements = batchSize * size;
				const auto threads = GetThreads(elements);
				const auto inputs = Inputs.size();
				const auto vecZero = VecFloat(0);

				switch (inputs)
				{
				case 2:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;
						
						VecFloat In0, In1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							(In0 + In1).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
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

						VecFloat In0, In1, In2;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							(In0 + In1 + In2).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw] + Inputs[2]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat In0, In1, In2, In3;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							In3.load_a(&Inputs[3]->Neurons[cdhw]);
							(In0 + In1 + In2 + In3).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw] + Inputs[2]->Neurons[cdhw] + Inputs[3]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				default:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat inputD1, sum;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							sum = vecZero;
							for (auto i = 0ull; i < inputs; i++)
							{
								inputD1.load_a(&Inputs[i]->Neurons[cdhw]);
								sum += inputD1;
							}
							sum.store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							NeuronsD1[cdhw] = 0;
							Neurons[cdhw] = 0;
							for (auto i = 0ull; i < inputs; i++)
								Neurons[cdhw] += Inputs[i]->Neurons[cdhw];
						}
					});
				}
				break;
				}
#endif
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::sum(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);
			const auto elements = batchSize * size;
			const auto threads = GetThreads(elements);
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				switch (inputs)
				{
				case 2:
				{
					VecFloat inputD1, D1;
					for (auto cdhw = 0; cdhw < part; cdhw += VectorSize)
					{
						D1.load_a(&NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
						inputD1 += D1;
						inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
						inputD1 += D1;
						inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					PRAGMA_OMP_SIMD()
					for (auto cdhw = part; cdhw < size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
					}
				}
				break;

				default:
				{
					for (auto i = 0ull; i < inputs; i++)
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
							Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
				}
				}
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

						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
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

						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 5:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[4]->NeuronsD1[cdhw]);
							inputD1 += D1;
							inputD1.store_a(&Inputs[4]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 6:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 7:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 8:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
					});
				}
				break;

				default:
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						}
						for (auto i = 8ull; i < inputs; i++)
							for (auto cdhw = start; cdhw < end; cdhw++)
								Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
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
