#pragma once
#include "Layer.h"

namespace dnn
{
	class Multiply final : public Layer
	{
	public:
		Multiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Multiply, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
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

		size_t FanOut() const  final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize)  final override
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

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat In, Out;
					VecFloat vecZero = VecFloat(0);
					for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw +=VectorSize)
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
							Out.load_a(&Neurons[cdhw]);
							In.load_a(&Inputs[i]->Neurons[cdhw]);
							Out *= In;
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
					for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						Neurons[cdhw] *= Inputs[i]->Neurons[cdhw];
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
							VecFloat In0, In1;
							const auto vecZero = VecFloat(0);
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								(In0 * In1).store_a(&Neurons[cdhw]);
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
							VecFloat In0, In1, In2;
							const auto vecZero = VecFloat(0);
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								In2.load_a(&Inputs[2]->Neurons[cdhw]);
								(In0 * In1 * In2).store_a(&Neurons[cdhw]);
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
								VecFloat In0, In1, In2, In3;
								const auto vecZero = VecFloat(0);
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									In3.load_a(&Inputs[3]->Neurons[cdhw]);
									(In0 * In1 * In2 * In3).store_a(&Neurons[cdhw]);
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
							VecFloat In, Out;
							const auto vecZero = VecFloat(0);
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
									Out *= In;
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
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
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
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
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
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
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
									Neurons[cdhw] *= Inputs[i]->Neurons[cdhw];
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
								VecFloat In0, In1;
								const auto vecZero = VecFloat(0);
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									(In0 * In1).store_a(&Neurons[cdhw]);
								}
							});
						break;

					case 3:
						for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								VecFloat In0, In1, In2;
								const auto vecZero = VecFloat(0);
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									(In0 * In1 * In2).store_a(&Neurons[cdhw]);
								}
							});
						break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
							{
								const auto start = n * PaddedCDHW;
								const auto end = start + PaddedCDHW;
								VecFloat In0, In1, In2, In3;
								const auto vecZero = VecFloat(0);
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									In0.load_a(&Inputs[0]->Neurons[cdhw]);
									In1.load_a(&Inputs[1]->Neurons[cdhw]);
									In2.load_a(&Inputs[2]->Neurons[cdhw]);
									In3.load_a(&Inputs[3]->Neurons[cdhw]);
									(In0 * In1 * In2 * In3).store_a(&Neurons[cdhw]);
								}
							});
						break;

					default:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							VecFloat In, Out;
							const auto vecZero = VecFloat(0);
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
									Out *= In;
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
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
						});
						break;

					case 3:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
						});
						break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
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
									Neurons[cdhw] *= Inputs[i]->Neurons[cdhw];
						});
						break;
					}
				}
			}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const size_t batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
					switch (inputs)
					{
					case 2:
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
						{
							mul_add(VecFloat().load_a(&Inputs[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
					break;

					case 3:
					{
						VecFloat D1, In0, In1, In2;
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							mul_add(D1 * In1, In2, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(D1 * In0, In2, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							mul_add(D1 * In0, In1, VecFloat().load_a(&Inputs[2]->NeuronsD1[cdhw])).store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
					}
					break;

					case 4:
					{
						VecFloat D1, In0, In1, In2, In3;
						for (auto cdhw = 0ull; cdhw < PaddedCDHW; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							In3.load_a(&Inputs[3]->Neurons[cdhw]);
							mul_add(D1 * In1, In2 * In3, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(D1 * In0, In2 * In3, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							mul_add(D1 * In0, In1 * In3, VecFloat().load_a(&Inputs[2]->NeuronsD1[cdhw])).store_a(&Inputs[2]->NeuronsD1[cdhw]);
							mul_add(D1 * In0, In1 * In2, VecFloat().load_a(&Inputs[3]->NeuronsD1[cdhw])).store_a(&Inputs[3]->NeuronsD1[cdhw]);
						}
					}
					break;

					default:
						break;
					}
				else
					switch (inputs)
					{
					case 2:
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw];
						}
					break;

					case 3:
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
						}
					break;

					case 4:
						for (auto cdhw = 0ull; cdhw < CDHW; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
						}
					break;

					default:
						break;
					}
			}
			else
			{
#endif
				if (!plain)
					switch (inputs)
					{
					case 2:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								mul_add(VecFloat().load_a(&Inputs[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
						});
					break;

					case 3:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							VecFloat D1, In0, In1, In2;
							for (auto cdhw = start; cdhw < end; cdhw+= VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								In2.load_a(&Inputs[2]->Neurons[cdhw]);
								mul_add(D1 * In1, In2, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(D1 * In0, In2, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
								mul_add(D1 * In0, In1, VecFloat().load_a(&Inputs[2]->NeuronsD1[cdhw])).store_a(&Inputs[2]->NeuronsD1[cdhw]);
							}
						});
					break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							VecFloat D1, In0, In1, In2, In3;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								In2.load_a(&Inputs[2]->Neurons[cdhw]);
								In3.load_a(&Inputs[3]->Neurons[cdhw]);
								mul_add(D1 * In1, In2 * In3, VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(D1 * In0, In2 * In3, VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
								mul_add(D1 * In0, In1 * In3, VecFloat().load_a(&Inputs[2]->NeuronsD1[cdhw])).store_a(&Inputs[2]->NeuronsD1[cdhw]);
								mul_add(D1 * In0, In1 * In2, VecFloat().load_a(&Inputs[3]->NeuronsD1[cdhw])).store_a(&Inputs[3]->NeuronsD1[cdhw]);
							}
						});
					break;

					default:
						break;
					}
				else
					switch (inputs)
					{
					case 2:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw];
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
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
								Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
							}
						});
						break;

					case 4:
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
								Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[3]->Neurons[cdhw];
								Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw] * Inputs[2]->Neurons[cdhw];
							}
						});
						break;

					default:
						break;
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
