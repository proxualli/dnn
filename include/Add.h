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
		FloatVector SurvivalProbability;

		Add(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Add, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			SurvivalProbability(FloatVector(inputs.size(), Float(1)))
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
			const auto inputs = Inputs.size();
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);
#ifdef DNN_STOCHASTIC
			const auto threads = 1ull;
#else
			const auto threads = GetThreads(batchSize * size);
#endif
			auto fullDepth = true;
			auto skippedInputs = FloatVector(inputs, Float(1));
			for (auto i = 0ull; i < inputs; i++)
			{
				skippedInputs[i] = Inputs[i]->Skip ? Float(0) : Float(1);
				if (SurvivalProbability[i] != Float(1))
					fullDepth = false;
			}

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
				const auto vecZero = VecFloat(0);
				
				switch (inputs)
				{
				case 2:
				{
					if (fullDepth)
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
					else
					{	
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							const auto end = start + part;
							const Float survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
							const Float survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));

							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								((In0 * survivalProb0) + (In1 * survivalProb1)).store_a(&Neurons[cdhw]);
								vecZero.store_nt(&NeuronsD1[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
							{
								Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * survivalProb0) + (Inputs[1]->Neurons[cdhw] * survivalProb1);
								NeuronsD1[cdhw] = 0;
							}
						});
					}
				}
				break;

				case 3:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto end = start + part;
						const Float survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
						const Float survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));
						const Float survivalProb2 = SurvivalProbability[2] * (Inputs[2]->Skip ? Float(0) : Float(1));

						VecFloat In0, In1, In2;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							((In0 * survivalProb0) + (In1 * survivalProb1) + (In2 * survivalProb2)).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * survivalProb0) + (Inputs[1]->Neurons[cdhw] * survivalProb1) + (Inputs[2]->Neurons[cdhw] * survivalProb2);
							NeuronsD1[cdhw] = 0;
						}
					});
				}
				break;

				case 4:
				{
					const Float survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
					const Float survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));
					const Float survivalProb2 = SurvivalProbability[2] * (Inputs[2]->Skip ? Float(0) : Float(1));
					const Float survivalProb3 = SurvivalProbability[3] * (Inputs[3]->Skip ? Float(0) : Float(1));

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
							((In0 * survivalProb0) + (In1 * survivalProb1) + (In2 * survivalProb2) + (In3 * survivalProb3)).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * survivalProb0) + (Inputs[1]->Neurons[cdhw] * survivalProb1) + (Inputs[2]->Neurons[cdhw] * survivalProb2) + (Inputs[3]->Neurons[cdhw] * survivalProb3);
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

						VecFloat In, sum;
						for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
						{
							sum = vecZero;
							for (auto i = 0ull; i < inputs; i++)
							{
								In.load_a(&Inputs[i]->Neurons[cdhw]);
								sum += In * SurvivalProbability[i] * (Inputs[i]->Skip ? Float(0) : Float(1));
							}
							sum.store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							NeuronsD1[cdhw] = 0;
							Neurons[cdhw] = 0;
							for (auto i = 0ull; i < inputs; i++)
								Neurons[cdhw] += Inputs[i]->Neurons[cdhw] * SurvivalProbability[i] * (Inputs[i]->Skip ? Float(0) : Float(1));
						}
					});
				}
				break;
				}
#endif
			}
			else
			{
				if (fullDepth)
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::sum(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();
				}
				else
				{
					switch (inputs)
					{
					case 2:
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							const auto end = start + part;
							const auto survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
							const auto survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));

							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								((In0 * survivalProb0) + (In1 * survivalProb1)).store_a(&Neurons[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
								Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * survivalProb0) + (Inputs[1]->Neurons[cdhw] * survivalProb1);
						});
						
					}
					break;

					case 3:
					{
						const auto survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
						const auto survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));
						const auto survivalProb2 = SurvivalProbability[2] * (Inputs[2]->Skip ? Float(0) : Float(1));

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
								((In0 * survivalProb0) + (In1 * survivalProb1) + (In2 * survivalProb2)).store_a(&Neurons[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
								Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * SurvivalProbability[0]) + (Inputs[1]->Neurons[cdhw] * SurvivalProbability[1]) + (Inputs[2]->Neurons[cdhw] * SurvivalProbability[2]);
						});
					}
					break;

					case 4:
					{
						const auto survivalProb0 = SurvivalProbability[0] * (Inputs[0]->Skip ? Float(0) : Float(1));
						const auto survivalProb1 = SurvivalProbability[1] * (Inputs[1]->Skip ? Float(0) : Float(1));
						const auto survivalProb2 = SurvivalProbability[2] * (Inputs[2]->Skip ? Float(0) : Float(1));
						const auto survivalProb3 = SurvivalProbability[3] * (Inputs[3]->Skip ? Float(0) : Float(1));

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
								((In0 * survivalProb0) + (In1 * survivalProb1) + (In2 * survivalProb2) + (In3 * survivalProb3)).store_a(&Neurons[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
								Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * survivalProb0) + (Inputs[1]->Neurons[cdhw] * survivalProb1) + (Inputs[2]->Neurons[cdhw] * survivalProb2) + (Inputs[3]->Neurons[cdhw] * survivalProb3);
						});
					}
					break;

					default:
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							const auto end = start + part;

							VecFloat In, sum;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								sum = VecFloat(0);
								for (auto i = 0ull; i < inputs; i++)
								{
									In.load_a(&Inputs[i]->Neurons[cdhw]);
									sum += In * SurvivalProbability[i] * (Inputs[i]->Skip ? Float(0) : Float(1));
								}
								sum.store_a(&Neurons[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
							{
								Neurons[cdhw] = 0;
								for (auto i = 0ull; i < inputs; i++)
									Neurons[cdhw] += Inputs[i]->Neurons[cdhw] * SurvivalProbability[i] * (Inputs[i]->Skip ? Float(0) : Float(1));
							}
						});
					}
					break;
					}
				}
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif
			const auto inputs = Inputs.size();
			const auto size = IsPlainFormat() ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);
			auto fullDepth = true;
			auto skippedInputs = FloatVector(inputs, Float(1));
			for (auto i = 0ull; i < inputs; i++)
			{
				skippedInputs[i] = Inputs[i]->Skip ? Float(0) : Float(1);
				if (SurvivalProbability[i] != Float(1))
					fullDepth = false;
			}

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				switch (inputs)
				{
				case 2:
				{
					const auto skip0 = Inputs[0]->Skip ? Float(0) : Float(1);
					const auto skip1 = Inputs[1]->Skip ? Float(0) : Float(1);

					VecFloat inputD1, D1;
					for (auto cdhw = 0; cdhw < part; cdhw += VectorSize)
					{
						D1.load_a(&NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
						inputD1 += D1 * skip0;
						inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
						inputD1 += D1 * skip1;
						inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					PRAGMA_OMP_SIMD()
					for (auto cdhw = part; cdhw < size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skip0;
						Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skip1;
					}
				}
				break;

				default:
				{
					for (auto i = 0ull; i < inputs; i++)
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
							Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * (Inputs[i]->Skip ? Float(0) : Float(1));
				}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * size);
				
				switch (inputs)
				{
				case 2:
				{
					if (fullDepth)
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
								(inputD1 + D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
								(inputD1 + D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							const auto end = start + part;
							const auto skip0 = Inputs[0]->Skip ? Float(0) : Float(1);
							const auto skip1 = Inputs[1]->Skip ? Float(0) : Float(1);

							VecFloat inputD1, D1;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(D1, skip0, inputD1).store_a(&Inputs[0]->NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
								mul_add(D1, skip1, inputD1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
							for (auto cdhw = end; cdhw < start + size; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skip0;
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skip1;
							}
						});
					}
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
							inputD1 += D1 * skippedInputs[0];
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[1];
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[2];
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
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
							inputD1 += D1 * skippedInputs[0];
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[1];
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[2];
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[3];
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
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
							inputD1 += D1 * skippedInputs[0];
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[1];
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[2];
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[3];
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[4]->NeuronsD1[cdhw]);
							inputD1 += D1 * skippedInputs[4];
							inputD1.store_a(&Inputs[4]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = end; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[4];
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
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[5];
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
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[6];
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
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[6];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[7];
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
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[6];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[7];
						}
						for (auto i = 8ull; i < inputs; i++)
							for (auto cdhw = start; cdhw < end; cdhw++)
								Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * skippedInputs[i];
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