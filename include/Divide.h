#pragma once
#include "Layer.h"

namespace dnn
{
	class Divide final : public Layer
	{
	public:
		Divide(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Divide, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);
			assert(Inputs.size() < 5);

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
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer");
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = (size / VectorSize) * VectorSize;
			const auto elements = batchSize * size;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat vecZero = VecFloat(0);
					VecFloat In, Out;
					for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
					{
						In.load_a(&Inputs[0]->Neurons[cdhw]);
						In.store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
						vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
					}
					for (auto i = 1ull; i < inputs; i++)
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
						{
							In.load_a(&Inputs[i]->Neurons[cdhw]);
							Out.load_a(&Neurons[cdhw]);
							Out /= In;
							Out.store_a(&Neurons[cdhw]);
						}
				}
				else
					for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
					{
						Neurons[cdhw] = Inputs[0]->Neurons[cdhw];
#ifndef DNN_LEAN
						NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
					}
				for (auto i = 1ull; i < inputs; i++)
					PRAGMA_OMP_SIMD()
					for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						Neurons[cdhw] /= Inputs[i]->Neurons[cdhw];
			}
			else
			{
#endif
				if (training)
				{
					if (!plain)
					{
						switch (inputs)
						{
						case 2:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									(In0 / In1).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
								}
							});
							break;

						case 3:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1, In2;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									(In0 / In1 / In2).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
								}
							});
							break;

						case 4:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1, In2, In3;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									In3.load_a(&Inputs[3]->Neurons[cdhw]);
									(In0 / In1 / In2 / In3).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
								}
							});
							break;

						default:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In, Out;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In.load_a(&Inputs[0]->Neurons[cdhw]);
									In.store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
								}
								for (auto i = 1ull; i < inputs; i++)
									for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
									{
										In.load_a(&Inputs[i]->Neurons[cdhw]);
										Out.load_a(&Neurons[cdhw]);
										Out /= In;
										Out.store_a(&Neurons[cdhw]);
									}
							});
							break;
						}
					}
					else
					{
						switch (inputs)
						{
						case 2:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
								}
							});
							break;

						case 3:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw] / Inputs[2]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
								}
							});
							break;

						case 4:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw] / Inputs[2]->Neurons[cdhw] / Inputs[3]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
								}
							});
							break;

						default:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
								}
								for (auto i = 1ull; i < inputs; i++)
									for (auto cdhw = start; cdhw < end; cdhw++)
										Neurons[cdhw] /= Inputs[i]->Neurons[cdhw];
							});
							break;
						}
					}
				}
				else
				{
					if (!plain)
					{
						switch (inputs)
						{
						case 2:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									(In0 / In1).store_a(&Neurons[cdhw]);
								}
							});
							break;

						case 3:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1, In2;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									(In0 / In1 / In2).store_a(&Neurons[cdhw]);
								}
							});
							break;

						case 4:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In0, In1, In2, In3;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									In3.load_a(&Inputs[3]->Neurons[cdhw]);
									(In0 / In1 / In2 / In3).store_a(&Neurons[cdhw]);
								}
							});
							break;

						default:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								const auto vecZero = VecFloat(0);
								VecFloat In, Out;
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In.load_a(&Inputs[0]->Neurons[cdhw]);
									In.store_a(&Neurons[cdhw]);
								}
								for (auto i = 1ull; i < inputs; i++)
									for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
									{
										In.load_a(&Inputs[i]->Neurons[cdhw]);
										Out.load_a(&Neurons[cdhw]);
										Out /= In;
										Out.store_a(&Neurons[cdhw]);
									}
							});
							break;
						}
					}
					else
					{
						switch (inputs)
						{
						case 2:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								for (auto cdhw = start; cdhw < end; cdhw++)
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw];
							});
							break;

						case 3:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw] / Inputs[2]->Neurons[cdhw];
							});
							break;

						case 4:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw] / Inputs[2]->Neurons[cdhw] / Inputs[3]->Neurons[cdhw];
							});
							break;

						default:
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * CDHW;
								const auto end = start + CDHW;
								for (auto cdhw = start; cdhw < end; cdhw++)
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw];
								for (auto i = 1ull; i < inputs; i++)
									for (auto cdhw = start; cdhw < end; cdhw++)
										Neurons[cdhw] /= Inputs[i]->Neurons[cdhw];
							});
							break;
						}
					}
				}
#ifdef DNN_STOCHASTIC
			}
#endif
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
				if (!plain)
				{
					switch (inputs)
					{
					case 2:
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
						{
							mul_add(approx_recipr(VecFloat().load_a(&Inputs[1]->Neurons[cdhw])), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&NeuronsD1[cdhw]), approx_recipr(square(VecFloat().load_a(&Inputs[1]->Neurons[cdhw]))), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
						break;

					case 3:
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw]);
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw]);
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]));
						}
						break;

					case 4:
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]) * Inputs[3]->Neurons[cdhw]);
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * FloatSquare(Inputs[3]->Neurons[cdhw]));
						}
						break;
					}
				}
				else
				{
					switch (inputs)
					{
					case 2:
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / Inputs[1]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / FloatSquare(Inputs[1]->Neurons[cdhw]);
						}
						break;

					case 3:
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw]);
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw]);
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]));
						}
						break;

					case 4:
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]) * Inputs[3]->Neurons[cdhw]);
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * FloatSquare(Inputs[3]->Neurons[cdhw]));
						}
						break;
					}
				}
			}
			else
			{
#endif
				if (!plain)
				{
					switch (inputs)
					{
					case 2:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								mul_add(approx_recipr(VecFloat().load_a(&Inputs[1]->Neurons[cdhw])), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&NeuronsD1[cdhw]), approx_recipr(square(VecFloat().load_a(&Inputs[1]->Neurons[cdhw]))), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
						});
						break;

					case 3:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / (Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n]);
								Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (FloatSquare(Inputs[1]->Neurons[n]) * Inputs[2]->Neurons[n]);
								Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (Inputs[1]->Neurons[n] * FloatSquare(Inputs[2]->Neurons[n]));
							}
						});
						break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							PRAGMA_OMP_SIMD()
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
								Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]) * Inputs[3]->Neurons[cdhw]);
								Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * FloatSquare(Inputs[3]->Neurons[cdhw]));
							}
						});
						break;
					}
				}
				else
				{
					switch (inputs)
					{
					case 2:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							PRAGMA_OMP_SIMD()
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / Inputs[1]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / FloatSquare(Inputs[1]->Neurons[cdhw]);
							}
						});
						break;

					case 3:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / (Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n]);
								Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (FloatSquare(Inputs[1]->Neurons[n]) * Inputs[2]->Neurons[n]);
								Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (Inputs[1]->Neurons[n] * FloatSquare(Inputs[2]->Neurons[n]));
							}
						});
						break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							PRAGMA_OMP_SIMD()
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (FloatSquare(Inputs[1]->Neurons[cdhw]) * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw]);
								Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * FloatSquare(Inputs[2]->Neurons[cdhw]) * Inputs[3]->Neurons[cdhw]);
								Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] / (Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * FloatSquare(Inputs[3]->Neurons[cdhw]));
							}
						});
						break;
					}
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
