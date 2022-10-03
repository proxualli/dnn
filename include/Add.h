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
		//std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::sum> fwd;
		//std::unique_ptr<dnnl::binary> bwdAdd;
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
			if (InputLayer->DstMemDesc->get_ndims() == 2)
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
				if (Inputs[i]->DstMemDesc->get_ndims() == 2)
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, ChosenFormat));
				else
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(Device.engine , *DstMemDesc, scales, srcsMemsDesc));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

/*
			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));
			dnnl::primitive_attr binary_attr_ops;
			binary_attr_ops.set_scales(0, 0, scales);
*/

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
			//bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			auto fullDepth = true;
			for (auto i = 0ull; i < Inputs.size(); i++)
				if (SurvivalProbability[i] != Float(1))
					fullDepth = false;
			
			for (auto i = 0ull; i < Inputs.size(); i++)
				scales[i] = fullDepth ? Float(1) : SurvivalProbability[i] * (Inputs[i]->Skip ? Float(0) : Float(1));
		
			
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
				const auto size = IsPlainFormat() ? CDHW() : PaddedCDHW();
#ifdef DNN_STOCHASTIC
				const auto threads = 1ull;
#else
				const auto threads = GetThreads(batchSize * size);
#endif
				const auto part = GetVectorPart(size);
				const auto vecZero = VecFloat(0);
				
				switch (Inputs.size())
				{
				case 2:
				{
					if (fullDepth)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							
							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								(In0 + In1).store_a(&Neurons[cdhw]);
								vecZero.store_nt(&NeuronsD1[cdhw]);
							}
							for (auto cdhw = start + part; cdhw < start + size; cdhw++)
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
							const auto scales0 = scales[0];
							const auto scales1 = scales[1];

							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								((In0 * scales0) + (In1 * scales1)).store_a(&Neurons[cdhw]);
								vecZero.store_nt(&NeuronsD1[cdhw]);
							}
							for (auto cdhw = start + part; cdhw < start + size; cdhw++)
							{
								Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * scales0) + (Inputs[1]->Neurons[cdhw] * scales1);
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
						const auto scales0 = scales[0];
						const auto scales1 = scales[1];
						const auto scales2 = scales[2];
						
						VecFloat In0, In1, In2;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							((In0 * scales0) + (In1 * scales1) + (In2 * scales2)).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * scales0) + (Inputs[1]->Neurons[cdhw] * scales1) + (Inputs[2]->Neurons[cdhw] * scales2);
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
						const auto scales0 = scales[0];
						const auto scales1 = scales[1];
						const auto scales2 = scales[2];
						const auto scales3 = scales[3];

						VecFloat In0, In1, In2, In3;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							In0.load_a(&Inputs[0]->Neurons[cdhw]);
							In1.load_a(&Inputs[1]->Neurons[cdhw]);
							In2.load_a(&Inputs[2]->Neurons[cdhw]);
							In3.load_a(&Inputs[3]->Neurons[cdhw]);
							((In0 * scales0) + (In1 * scales1) + (In2 * scales2) + (In3 * scales3)).store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * scales0) + (Inputs[1]->Neurons[cdhw] * scales1) + (Inputs[2]->Neurons[cdhw] * scales2) + (Inputs[3]->Neurons[cdhw] * scales3);
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
						
						VecFloat In, sum;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							sum = vecZero;
							for (auto i = 0ull; i < Inputs.size(); i++)
							{
								In.load_a(&Inputs[i]->Neurons[cdhw]);
								sum += In * scales[i];
							}
							sum.store_a(&Neurons[cdhw]);
							vecZero.store_nt(&NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							NeuronsD1[cdhw] = 0;
							Neurons[cdhw] = 0;
							for (auto i = 0ull; i < Inputs.size(); i++)
								Neurons[cdhw] += Inputs[i]->Neurons[cdhw] * scales[i];
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
			const auto size = IsPlainFormat() ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);

			auto fullDepth = true;
			for (auto i = 0ull; i < Inputs.size(); i++)
				if (SurvivalProbability[i] != Float(1))
					fullDepth = false;

			for (auto i = 0ull; i < Inputs.size(); i++)
				scales[i] = fullDepth ? Float(1) : (Inputs[i]->Skip ? Float(0) : Float(1));
/*
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
#ifdef DNN_CACHE_PRIMITIVES
				bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[i]->DiffDstMemDesc, Device.engine, Inputs[i]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*Inputs[i]->DiffDstMemDesc, Device.engine, Inputs[i]->NeuronsD1.data()) } });
#else
				dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
				Device.stream.wait();
			}
*/

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				switch (Inputs.size())
				{
				case 2:
				{
					const auto scales0 = scales[0];
					const auto scales1 = scales[1];
										
					VecFloat inputD1, D1;
					for (auto cdhw = 0; cdhw < part; cdhw += VectorSize)
					{
						D1.load_a(&NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
						inputD1 += D1 * scales0;
						inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

						inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
						inputD1 += D1 * scales1;
						inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}
					PRAGMA_OMP_SIMD()
					for (auto cdhw = part; cdhw < size; cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales0;
						Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales1;
					}
				}
				break;

				default:
				{
					for (auto i = 0ull; i < Inputs.size(); i++)
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
							Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[i];
				}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * size);

				switch (Inputs.size())
				{
				case 2:
				{
					if (fullDepth)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * size;
							
							VecFloat inputD1, D1;
							for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
								(inputD1 + D1).store_a(&Inputs[0]->NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
								(inputD1 + D1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
							for (auto cdhw = start + part; cdhw < start + size; cdhw++)
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
							const auto scale0 = scales[0];
							const auto scale1 = scales[1];

							VecFloat inputD1, D1;
							for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
							{
								D1.load_a(&NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(D1, scale0, inputD1).store_a(&Inputs[0]->NeuronsD1[cdhw]);

								inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
								mul_add(D1, scale1, inputD1).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
							for (auto cdhw = start + part; cdhw < start + size; cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale0;
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale1;
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
						const auto scale0 = scales[0];
						const auto scale1 = scales[1];
						const auto scale2 = scales[2];
						
						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale0;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale2;
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale0;
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale1;
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale2;
						}
					});
				}
				break;

				case 4:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						const auto scale0 = scales[0];
						const auto scale1 = scales[1];
						const auto scale2 = scales[2];
						const auto scale3 = scales[3];
												
						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale0;
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale1;
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale2;
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1 * scale3;
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale0;
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale1;
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale2;
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale3;
						}
					});
				}
				break;

				case 5:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						
						VecFloat inputD1, D1;
						for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[0]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales[0];
							inputD1.store_a(&Inputs[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales[1];
							inputD1.store_a(&Inputs[1]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[2]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales[2];
							inputD1.store_a(&Inputs[2]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[3]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales[3];
							inputD1.store_a(&Inputs[3]->NeuronsD1[cdhw]);

							inputD1.load_a(&Inputs[4]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales[4];
							inputD1.store_a(&Inputs[4]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = start + part; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[4];
						}
					});
				}
				break;

				case 6:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[5];
						}
					});
				}
				break;

				case 7:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[6];
						}
					});
				}
				break;

				case 8:
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[6];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[7];
						}
					});
				}
				break;

				default:
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * size;
						PRAGMA_OMP_SIMD()
						for (auto cdhw = start; cdhw < start + size; cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[0];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[1];
							Inputs[2]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[2];
							Inputs[3]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[3];
							Inputs[4]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[4];
							Inputs[5]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[5];
							Inputs[6]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[6];
							Inputs[7]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[7];
						}
						for (auto i = 8ull; i < Inputs.size(); i++)
							for (auto cdhw = start; cdhw < start + size; cdhw++)
								Inputs[i]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales[i];
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