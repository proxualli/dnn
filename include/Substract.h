#pragma once
#include "Layer.h"

namespace dnn
{
	class Substract final : public Layer
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
		Substract(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Substract, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);

			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}

			Scales = std::vector<Float>(Inputs.size(), Float(-1));
			Scales[0] = Float(1);
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
					throw std::invalid_argument("Incompatible memory formats in Substract layer");
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>(Inputs.size());
			for (auto i = 0ull; i < Inputs.size(); i++)
				srcsMemsDesc[i] = *Inputs[i]->DstMemDesc;

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
#endif
		}

		void ForwardProp(const size_t batchSize, const bool training)  final override
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
				const auto size = plain ? CDHW : PaddedCDHW;
				const auto part = (size / VectorSize) * VectorSize;
				const auto elements = batchSize * size;
				const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
				const auto inputs = Inputs.size();
				
				switch (inputs)
				{
				case 2:
				{
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
							(In0 - In1).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] - Inputs[1]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				case 3:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						const VecFloat vecZero = VecFloat(0);
						VecFloat In0, In1, In2;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							(In0 - In1 - In2).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] - Inputs[1]->Neurons[cdhw] - Inputs[2]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						const VecFloat vecZero = VecFloat(0);
						VecFloat In0, In1, In2, In3;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							In3.load_a(&Inputs[3]->Neurons[cdhw]);
							(In0 - In1 - In2 - In3).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw] - Inputs[1]->Neurons[cdhw] - Inputs[2]->Neurons[cdhw] - Inputs[3]->Neurons[cdhw];
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				default:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						const VecFloat vecZero = VecFloat(0);
						VecFloat In; VecFloat sum;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							sum.load_a(&Inputs[0]->Neurons[cdhw]);;
							for (auto i = 1ull; i < inputs; i++)
							{
								In.load_a(&Inputs[i]->Neurons[cdhw]);
								sum -= In;
							}
							sum.store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							NeuronsD1[cdhw] = 0;
							Neurons[cdhw] = Inputs[0]->Neurons[cdhw];
							for (auto i = 1ull; i < inputs; i++)
								Neurons[cdhw] -= Inputs[i]->Neurons[cdhw];
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

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = (size / VectorSize) * VectorSize;
			const auto elements = batchSize * size;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				switch (inputs)
				{
				case 2:
				{
					VecFloat In, D1;
					for (auto cdhw = 0; cdhw < part; cdhw += VectorSize)
					{
						D1.load_a(&NeuronsD1[cdhw]);

						In.load_a(&Inputs[0]->NeuronsD1[cdhw]);
						In += D1;
						In.store_a(&Inputs[0]->NeuronsD1[cdhw]);

						In.load_a(&Inputs[1]->NeuronsD1[cdhw]);
						In -= D1;
						In.store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					PRAGMA_OMP_SIMD()
					for (auto cdhw = part; cdhw < size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
						Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
					}
				}
				break;

				default:
				{
					PRAGMA_OMP_SIMD()
					for (auto cdhw = 0ull; cdhw < size; cdhw++)
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
					for (auto i = 1ull; i < inputs; i++)
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
							Inputs[i]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
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
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							In.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 3:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							In.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
						PRAGMA_OMP_SIMD()
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							In.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[3]->NeuronsD1[cdhw]);
						}
						PRAGMA_OMP_SIMD()
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 5:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + part;

						VecFloat In, D1;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							In.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							In += D1;
							In.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[3]->NeuronsD1[cdhw]);

							In.load_a(&Inputs[4]->NeuronsD1[cdhw]);
							In -= D1;
							In.store_a(&Inputs[4]->NeuronsD1[cdhw]);
						}
						PRAGMA_OMP_SIMD()
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 6:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 7:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				case 8:
				{
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[7]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
					});
				}
				break;

				default:
					for_i(batchSize, threads, [=](size_t n)
					{
						const auto start = n * size;
						const auto end = start + size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < end; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							Inputs[1]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[2]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[3]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[4]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[5]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[6]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
							Inputs[7]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
						}
						for (auto i = 8ull; i < inputs; i++)
							PRAGMA_OMP_SIMD()
							for (auto cdhw = start; cdhw < end; cdhw++)
								Inputs[i]->NeuronsD1[cdhw] -= NeuronsD1[cdhw];
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