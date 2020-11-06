#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelSplit final : public Layer
	{
	public:
		const size_t Group;
		const size_t Groups;

		ChannelSplit(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t group, const size_t groups) :
			Layer(device, format, name, LayerTypes::ChannelSplit, 0, 0, inputs[0]->PaddedC / groups, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Group(group),
			Groups(groups)
		{
			assert(Inputs.size() == 1);
		}

		std::string ChannelSplit::GetDescription() const final override
		{
			std::string description = GetDescriptionHeader();

			description.append(nwl + " Groups:" + tab + std::to_string(Groups));
			description.append(nwl + " Group:" + dtab + std::to_string(Group));

			return description;
		}

		size_t ChannelSplit::FanIn() const final override
		{
			return 1;
		}

		size_t ChannelSplit::FanOut() const final override
		{
			return 1;
		}

		void ChannelSplit::InitializeDescriptors(const size_t batchSize) final override
		{
			if (InputLayer->PaddedC % Groups != 0)
				throw std::exception("input not splittable in ChannelSplit");


			if (GetDataFmt(*InputLayer->DstMemDesc) != BlockedFmt)
				throw std::exception("Blocked format expected in ChannelSplit");

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, BlockedFmt));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, BlockedFmt));
		}

		void ChannelSplit::ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto strideH = W * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				const auto vecZero = VecFloat(0);

				for (auto c = 0ull; c < PaddedC; c += VectorSize)
				{
					const auto offsetC = (c + ((Group - 1) * PaddedC)) * HW;
					const auto offsetCHalf = c * HW;

					for (auto h = 0ull; h < H; h++)
					{
						const auto offsetH = offsetC + h * strideH;
						const auto offsetHHalf = offsetCHalf + h * strideH;

						for (auto w = offsetH, x = offsetHHalf; w < offsetH + strideH; w += VectorSize, x += VectorSize)
						{
							(VecFloat().load_a(&InputLayer->NeuronsD1[w]) += VecFloat().load_a(&NeuronsD1[x])).store_a(&InputLayer->NeuronsD1[w]);
#ifndef DNN_LEAN
							vecZero.store_nt(&NeuronsD1[x]);
#endif // DNN_LEAN
						}
					}
				}
			}
			else
			{
#endif
				for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
				{
					const auto vecZero = VecFloat(0);

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						const auto offsetC = n * InputLayer->PaddedCDHW + (c + ((Group - 1) * PaddedC)) * HW;
						const auto offsetCHalf = n * PaddedCDHW + c * HW;

						for (auto h = 0ull; h < H; h++)
						{
							const auto offsetH = offsetC + h * strideH;
							const auto offsetHHalf = offsetCHalf + h * strideH;

							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								(VecFloat().load_a(&InputLayer->Neurons[w + offsetH])).store_a(&Neurons[w + offsetHHalf]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w + offsetHHalf]);
#endif // DNN_LEAN
							}
						}
					}
				});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void ChannelSplit::BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto strideH = W * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto c = 0ull; c < PaddedC; c += VectorSize)
				{
					const auto offsetC = (c + ((Group - 1) * PaddedC)) * HW;
					const auto offsetCHalf = c * HW;

					for (auto h = 0ull; h < H; h++)
					{
						const auto offsetH = offsetC + h * strideH;
						const auto offsetHHalf = offsetCHalf + h * strideH;

						for (auto w = offsetH, x = offsetHHalf; w < offsetH + strideH; w += VectorSize, x += VectorSize)
							(VecFloat().load_a(&InputLayer->NeuronsD1[w]) += VecFloat().load_a(&NeuronsD1[x])).store_a(&InputLayer->NeuronsD1[w]);
					}
				}
			}
			else
			{
#endif
				for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto offsetC = n * InputLayer->PaddedCDHW + (c + ((Group - 1) * PaddedC)) * HW;
							const auto offsetCHalf = n * PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								const auto offsetHHalf = offsetCHalf + h * strideH;

								for (auto w = 0ull; w < strideH; w += VectorSize)
									(VecFloat().load_a(&InputLayer->NeuronsD1[w + offsetH]) += VecFloat().load_a(&NeuronsD1[w + offsetHHalf])).store_a(&InputLayer->NeuronsD1[w + offsetH]);
							}
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
