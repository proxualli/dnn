#pragma once
#include "Utils.h"

#include "jpeglib.h"
#include "jerror.h"

#undef cimg_display
#define cimg_display 0
#include "CImg.h"

namespace dnn
{
	constexpr auto MaximumLevels = 10;		// number of levels in AutoAugment
	constexpr auto FloatLevel(const int level, const Float minValue = Float(0.1), const Float maxValue = Float(1.9)) noexcept { return (Float(level) * (maxValue - minValue) / MaximumLevels) + minValue; }
	constexpr auto IntLevel(const int level, const int minValue = 0, const int maxValue = MaximumLevels) noexcept { return (level * (maxValue - minValue) / MaximumLevels) + minValue; }

	enum class Interpolations
	{
		Cubic = 0,
		Linear = 1,
		Nearest = 2
	};

	enum class Positions
	{
		TopLeft = 0,
		TopRight = 1,
		BottomLeft = 2,
		BottomRight = 3,
		Center = 4
	};
	
	template<typename T>
	struct Image
	{
		typedef std::vector<T, AlignedAllocator<T, 64ull>> VectorT;

	private:
		VectorT Data;

	public:
		UInt C; // Channels
		UInt D;	// Depth
		UInt H;	// Height
		UInt W;	// Width

		Image() :
			C(0),
			D(0),
			H(0),
			W(0),
			Data(VectorT())
		{
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w, const VectorT& image) :
			C(c),
			D(d),
			H(h),
			W(w),
			Data(image)
		{
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w, const T* image) :
			C(c),
			D(d),
			H(h),
			W(w),
			Data(VectorT(c * d * h * w))
		{
			std::memcpy(Data.data(), image, c * d * h * w * sizeof(T));
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w) :
			C(c),
			D(d),
			H(h),
			W(w),
			Data(VectorT(c * d * h * w))
		{
		}

		~Image() = default;

		T& operator()(const UInt c, const UInt d, const UInt h, const UInt w)
		{
			return Data[w + (h * W) + (d * H * W) + (c * D * H * W)];
		}

		const T& operator()(const UInt c, const UInt d, const UInt h, const UInt w) const
		{
			return Data[w + (h * W) + (d * H * W) + (c * D * H * W)];
		}

		T* data()
		{
			return Data.data();
		}

		const T* data() const
		{
			return Data.data();
		}

		auto Area() const
		{
			return H * W;
		}

		auto ChannelSize() const
		{
			return D * H * W;
		}

		auto Size() const
		{
			return C * D * H * W;
		}
		
		static cimg_library::CImg<Float> ImageToCImgFloat(const Image& image)
		{
			auto img = cimg_library::CImg<Float>(static_cast<uint32_t>(image.W), static_cast<uint32_t>(image.H), static_cast<uint32_t>(image.D), static_cast<uint32_t>(image.C));

			for (auto c = 0u; c < image.C; c++)
				for (auto d = 0u; d < image.D; d++)
					for (auto h = 0u; h < image.H; h++)
						for (auto w = 0u; w < image.W; w++)
							img(w, h, d, c) = image(c, d, h, w);

				return img;
		}

		static cimg_library::CImg<T> ImageToCImg(const Image& image)
		{
			return cimg_library::CImg<T>(image.data(), static_cast<uint32_t>(image.W), static_cast<uint32_t>(image.H), static_cast<uint32_t>(image.D), static_cast<uint32_t>(image.C));
		}

		static Image CImgToImage(const cimg_library::CImg<T>& image)
		{
			return Image(image._spectrum, image._depth, image._height, image._width, image.data());
		}

		static Image AutoAugment(const Image& image, const UInt padD, const UInt padH, const UInt padW, const std::vector<Float>& mean, const bool mirrorPad)
		{
			Image img(image);

			const auto operation = UniformInt<UInt>(0, 24);

			switch (operation)
			{
			case 1:
			case 3:
			case 5:
				img = Padding(img, padD, padH, padW, mean, mirrorPad);
				break;
			}

			switch (operation)
			{
			case 0:
			{
				if (Bernoulli<bool>(Float(0.1)))
					img = Invert(img);

				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						img = Contrast(img, FloatLevel(6));
					else
						img = Contrast(img, FloatLevel(4));
				}
			}
			break;

			case 1:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Rotate(img, FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
					else
						img = Rotate(img, -FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, 0, IntLevel(7), mean);
					else
						img = Translate(img, 0, -IntLevel(7), mean);
				}
			}
			break;

			case 2:
			{
				if (Bernoulli<bool>(Float(0.8)))
				{
					if (Bernoulli<bool>())
						img = Sharpness(img, FloatLevel(2));
					else
						img = Sharpness(img, FloatLevel(8));
				}

				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						img = Sharpness(img, FloatLevel(3));
					else
						img = Sharpness(img, FloatLevel(7));
				}
			}
			break;

			case 3:
			{
				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						img = Rotate(img, FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
					else
						img = Rotate(img, -FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, IntLevel(7), 0, mean);
					else
						img = Translate(img, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 4:
			{
				if (Bernoulli<bool>())
					img = AutoContrast(img);

				if (Bernoulli<bool>(Float(0.9)))
					img = Equalize(img);
			}
			break;

			case 5:
			{
				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						img = Rotate(img, FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
					else
						img = Rotate(img, -FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						img = Posterize(img, 32);
					else
						img = Posterize(img, 64);
				}
			}
			break;

			case 6:
			{
				if (Bernoulli<bool>(Float(0.4)))
				{
					if (Bernoulli<bool>())
						img = Color(img, FloatLevel(3));
					else
						img = Color(img, FloatLevel(7));
				}

				if (Bernoulli<bool>(Float(0.6)))
				{
					if (Bernoulli<bool>())
						img = Brightness(img, FloatLevel(7));
					else
						img = Brightness(img, FloatLevel(3));
				}
			}
			break;

			case 7:
			{
				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						img = Sharpness(img, FloatLevel(9));
					else
						img = Sharpness(img, FloatLevel(1));
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Brightness(img, FloatLevel(8));
					else
						img = Brightness(img, FloatLevel(2));
				}
			}
			break;

			case 8:
			{
				if (Bernoulli<bool>(Float(0.6)))
					img = Equalize(img);

				if (Bernoulli<bool>())
					img = Equalize(img);
			}
			break;

			case 9:
			{
				if (Bernoulli<bool>(Float(0.6)))
				{
					if (Bernoulli<bool>())
						img = Contrast(img, FloatLevel(7));
					else
						img = Contrast(img, FloatLevel(3));
				}

				if (Bernoulli<bool>(Float(Float(0.6))))
				{
					if (Bernoulli<bool>(Float(0.5)))
						img = Sharpness(img, FloatLevel(6));
					else
						img = Sharpness(img, FloatLevel(4));
				}
			}
			break;

			case 10:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Color(img, FloatLevel(7));
					else
						img = Color(img, FloatLevel(3));
				}

				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						img = Translate(img, 0, IntLevel(8), mean);
					else
						img = Translate(img, 0, -IntLevel(8), mean);
				}
			}
			break;

			case 11:
			{
				if (Bernoulli<bool>(Float(0.3)))
					img = Equalize(img);

				if (Bernoulli<bool>(Float(0.4)))
					img = AutoContrast(img);
			}
			break;

			case 12:
			{
				if (Bernoulli<bool>(Float(0.4)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, IntLevel(3), 0, mean);
					else
						img = Translate(img, -IntLevel(3), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.2)))
					img = Sharpness(img, FloatLevel(6));
			}
			break;

			case 13:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						img = Brightness(img, FloatLevel(6));
					else
						img = Brightness(img, FloatLevel(4));
				}

				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						img = Color(img, FloatLevel(8));
					else
						img = Color(img, FloatLevel(2));
				}
			}
			break;

			case 14:
			{
				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						img = Solarize(img, IntLevel(2, 0, 256));
					else
						img = Solarize(img, IntLevel(8, 0, 256));
				}
			}
			break;

			case 15:
			{
				if (Bernoulli<bool>(Float(0.2)))
					img = Equalize(img);

				if (Bernoulli<bool>(Float(0.6)))
					img = AutoContrast(img);
			}
			break;

			case 16:
			{
				if (Bernoulli<bool>(Float(0.2)))
					img = Equalize(img);

				if (Bernoulli<bool>(Float(0.6)))
					img = Equalize(img);
			}
			break;

			case 17:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						img = Color(img, FloatLevel(8));
					else
						img = Color(img, FloatLevel(2));
				}

				if (Bernoulli<bool>(Float(0.6)))
					img = Equalize(img);
			}
			break;

			case 18:
			{
				if (Bernoulli<bool>(Float(0.8)))
					img = AutoContrast(img);

				if (Bernoulli<bool>(Float(0.2)))
					img = Solarize(img, IntLevel(8, 0, 256));
			}
			break;

			case 19:
			{
				if (Bernoulli<bool>(Float(0.1)))
					img = Brightness(img, FloatLevel(3));

				if (Bernoulli<bool>(Float(0.7)))
					img = Color(img, FloatLevel(4));
			}
			break;

			case 20:
			{
				if (Bernoulli<bool>(Float(0.4)))
					img = Solarize(img, IntLevel(5, 0, 256));

				if (Bernoulli<bool>(Float(0.9)))
					img = AutoContrast(img);
			}
			break;

			case 21:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, IntLevel(7), 0, mean);
					else
						img = Translate(img, -IntLevel(7), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, IntLevel(7), 0, mean);
					else
						img = Translate(img, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 22:
			{
				if (Bernoulli<bool>(Float(0.9)))
					img = AutoContrast(img);

				if (Bernoulli<bool>(Float(0.8)))
					img = Solarize(img, IntLevel(3, 0, 256));
			}
			break;

			case 23:
			{
				if (Bernoulli<bool>(Float(0.8)))
					img = Equalize(img);

				if (Bernoulli<bool>(Float(0.1)))
					img = Invert(img);
			}
			break;

			case 24:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						img = Translate(img, IntLevel(8), 0, mean);
					else
						img = Translate(img, -IntLevel(8), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.9)))
					img = AutoContrast(img);
			}
			break;
			}

			switch (operation)
			{
			case 1:
			case 3:
			case 5:
				break;

			default:
				img = Padding(img, padD, padH, padW, mean, mirrorPad);
				break;
			}

			return img;
		}

		static Image AutoContrast(const Image& image)
		{
			const T maximum = std::is_floating_point_v<T> ? static_cast<T>(1) : static_cast<T>(255);
			
			auto imgSource = ImageToCImg(image);

			imgSource.normalize(0, maximum);

			auto img = CImgToImage(imgSource);

			return img;
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Brightness(const Image& image, const Float magnitude)
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			const auto delta = (magnitude - Float(1)) / 2;

			for (auto d = 0u; d < image.D; d++)
				for (auto h = 0u; h < image.H; h++)
					for (auto w = 0u; w < image.W; w++)
						srcImage(w, h, d, 2u) = cimg_library::cimg::cut(srcImage(w, h, d, 2u) + delta, 0, 1);

			srcImage.HSLtoRGB();

			Image img(image.C, image.D, image.H, image.W);

			for (auto c = 0u; c < img.C; c++)
				for (auto d = 0u; d < img.D; d++)
					for (auto h = 0u; h < img.H; h++)
						for (auto w = 0u; w < img.W; w++)
							img(c, d, h, w) = Saturate<Float>(srcImage(w, h, d, c));

			return img;
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Color(const Image& image, const Float magnitude)
		{
			auto imgSource = ImageToCImgFloat(image);

			imgSource.RGBtoHSL();

			for (auto d = 0u; d < image.D; d++)
				for (auto h = 0u; h < image.H; h++)
					for (auto w = 0u; w < image.W; w++)
						imgSource(w, h, d, 0u) = cimg_library::cimg::cut(imgSource(w, h, d, 0u) * magnitude, 0, 360);

			imgSource.HSLtoRGB();

			Image img(image.C, image.D, image.H, image.W);
			for (auto c = 0u; c < img.C; c++)
				for (auto d = 0u; d < img.D; d++)
					for (auto h = 0u; h < img.H; h++)
						for (auto w = 0u; w < img.W; w++)
							img(c, d, h, w) = Saturate<Float>(imgSource(w, h, d, c));

			return img;
		}

		static Image ColorCast(const Image& image, const UInt angle)
		{
			auto imgSource = ImageToCImgFloat(image);

			imgSource.RGBtoHSL();

			const auto shift = Float(Bernoulli<bool>() ? static_cast<int>(UniformInt<UInt>(0, 2 * angle)) - static_cast<int>(angle) : 0);

			for (auto d = 0u; d < image.D; d++)
				for (auto h = 0u; h < image.H; h++)
					for (auto w = 0u; w < image.W; w++)
						imgSource(w, h, d, 0u) = cimg_library::cimg::cut(imgSource(w, h, d, 0u) + shift, 0, 360);
				
			imgSource.HSLtoRGB();

			Image img(image.C, image.D, image.H, image.W);
			for (auto c = 0u; c < img.C; c++)
				for (auto d = 0u; d < img.D; d++)
					for (auto h = 0u; h < img.H; h++)
						for (auto w = 0u; w < img.W; w++)
							img(c, d, h, w) = Saturate<Float>(imgSource(w, h, d, c));
					
			return img;
		}
		
		// magnitude = 0   // gray image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Contrast(const Image& image, const Float magnitude)
		{
			auto imgSource = ImageToCImgFloat(image);

			imgSource.RGBtoHSL();

			for (auto d = 0u; d < image.D; d++)
				for (auto h = 0u; h < image.H; h++)
					for (auto w = 0u; w < image.W; w++)
						imgSource(w, h, d, 1u) = cimg_library::cimg::cut(imgSource(w, h, d, 1u) * magnitude, 0, 1);

			imgSource.HSLtoRGB();

			Image img(image.C, image.D, image.H, image.W);
			for (auto c = 0u; c < img.C; c++)
				for (auto d = 0u; d < img.D; d++)
					for (auto h = 0u; h < img.H; h++)
						for (auto w = 0u; w < img.W; w++)
							img(c, d, h, w) = Saturate<Float>(imgSource(w, h, d, c));

			return img;
		}

		// magnitude = 0   // gray image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Crop(const Image& image, const Positions position, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image img(image.C, depth, height, width);

			for (auto c = 0ull; c < img.C; c++)
			{
				const T channelMean = std::is_floating_point_v<T> ? static_cast<T>(0) : static_cast<T>(mean[c]);
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = channelMean;
			}

			const auto minDepth = std::min(img.D, image.D);
			const auto minHeight = std::min(img.H, image.H);
			const auto minWidth = std::min(img.W, image.W);

			const auto srcDdelta = img.D < image.D ? (image.D - img.D) / 2 : 0ull;
			const auto dstDdelta = img.D > image.D ? (img.D - image.D) / 2 : 0ull;

			switch (position)
			{
			case Positions::Center:
			{
				const auto srcHdelta = img.H < image.H ? (image.H - img.H) / 2 : 0ull;
				const auto dstHdelta = img.H > image.H ? (img.H - image.H) / 2 : 0ull;
				const auto srcWdelta = img.W < image.W ? (image.W - img.W) / 2 : 0ull;
				const auto dstWdelta = img.W > image.W ? (img.W - image.W) / 2 : 0ull;

				for (auto c = 0ull; c < img.C; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::TopLeft:
			{
				for (auto c = 0ull; c < img.C; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								img(c, d + dstDdelta, h, w) = image(c, d + srcDdelta, h, w);
			}
			break;

			case Positions::TopRight:
			{
				const auto srcWdelta = img.W < image.W ? (image.W - img.W) : 0ull;
				const auto dstWdelta = img.W > image.W ? (img.W - image.W) : 0ull;

				for (auto c = 0ull; c < img.C; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								img(c, d + dstDdelta, h, w + dstWdelta) = image(c, d + srcDdelta, h, w + srcWdelta);
			}
			break;

			case Positions::BottomRight:
			{
				const auto srcHdelta = img.H < image.H ? (image.H - img.H) : 0ull;
				const auto dstHdelta = img.H > image.H ? (img.H - image.H) : 0ull;
				const auto srcWdelta = img.W < image.W ? (image.W- img.W) : 0ull;
				const auto dstWdelta = img.W > image.W ? (img.W - image.W) : 0ull;

				for (auto c = 0ull; c < img.C; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::BottomLeft:
			{
				const auto srcHdelta = img.H < image.H ? (image.H - img.H) : 0ull;
				const auto dstHdelta = img.H > image.H ? (img.H - image.H) : 0ull;

				for (auto c = 0ull; c < img.C; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w) = image(c, d + srcDdelta, h + srcHdelta, w);
			}
			break;
			}

			return img;
		}


		static Image Distorted(const Image& image, const Float scale, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean)
		{
			const auto zoom = scale / Float(100) * UniformReal<Float>( Float(-1), Float(1));
			const auto height = static_cast<UInt>(static_cast<int>(image.H) + static_cast<int>(std::round(static_cast<int>(image.H) * zoom)));
			const auto width = static_cast<UInt>(static_cast<int>(image.W) + static_cast<int>(std::round(static_cast<int>(image.W) * zoom)));

			return Image::Crop(Image::Rotate(Image::Resize(image, image.D, height, width, interpolation), angle * UniformReal<Float>( Float(-1), Float(1)), interpolation, mean), Positions::Center, image.D, image.H, image.W, mean);
		}

		static Image Dropout(const Image& image, const Float dropout, const std::vector<Float>& mean)
		{
			Image img(image);
			
			for (auto d = 0ull; d < img.D; d++)
				for (auto h = 0ull; h < img.H; h++)
					for (auto w = 0ull; w < img.W; w++)
						if (Bernoulli<bool>(dropout))
							for (auto c = 0ull; c < img.C; c++)
							{
								if constexpr (std::is_floating_point_v<T>)
									img(c, d, h, w) = static_cast<T>(0);
								else
									img(c, d, h, w) = static_cast<T>(mean[c]);
							}
			return img;
		}
		
		static Image Equalize(const Image& image)
		{
			auto imgSource = ImageToCImg(image);

			imgSource.equalize(256);

			auto img = CImgToImage(imgSource);

			return img;
		}

		static Float GetChannelMean(const Image& image, const UInt c)
		{
			auto mean = Float(0);

			for (auto d = 0ull; d < image.D; d++)
				for (auto h = 0ull; h < image.H; h++)
					for (auto w = 0ull; w < image.W; w++)
						mean += image(c, d, h, w);

			return mean /= image.ChannelSize();
		}

		static Float GetChannelVariance(const Image& image, const UInt c)
		{
			const auto mean = Image::GetChannelMean(image, c);

			auto variance = Float(0);

			for (auto d = 0ull; d < image.D; d++)
				for (auto h = 0ull; h < image.H; h++)
					for (auto w = 0ull; w < image.W; w++)
						variance += FloatSquare(image(c, d, h, w) - mean);

			return variance /= image.ChannelSize();
		}

		static Float GetChannelStdDev(const Image& image, const UInt c)
		{
			return std::max(std::sqrt(Image::GetChannelVariance(image, c)), Float(1) / std::sqrt(Float(image.ChannelSize())));
		}

		static Image HorizontalMirror(const Image& image)
		{
			Image img(image.C, image.D, image.H, image.W);

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = image(c, d, h, image.W - 1 - w);
			
			return img;
		}

		static Image Invert(const Image& image)
		{
			Image img(image.C, image.D, image.H, image.W);

			constexpr T maximum = std::is_floating_point_v<T> ? static_cast<T>(1) : static_cast<T>(255);

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = maximum - image(c, d, h, w);

			return img;
		}

		static Image LoadJPEG(const std::string& fileName, const bool forceColorFormat = false)
		{
			Image img = CImgToImage(cimg_library::CImg<T>().get_load_jpeg(fileName.c_str()));

			if (forceColorFormat && img.C == 1)
			{
				Image imgToColor = Image(3, img.D, img.W, img.H);

				for (auto c = 0ull; c < 3ull; c++)
					for (auto d = 0ull; d < img.D; d++)
						for (auto h = 0ull; h < img.H; h++)
							for (auto w = 0ull; w < img.W; w++)
								imgToColor(c, d, h, w) = img(0, d, h, w);

				return imgToColor;
			}
			else
				return img;
		}

		static Image LoadPNG(const std::string& fileName, const bool forceColorFormat = false)
		{
			auto bitsPerPixel = 0u;
			Image img = CImgToImage(cimg_library::CImg<T>().get_load_png(fileName.c_str(), &bitsPerPixel));

			if (forceColorFormat && img.C == 1)
			{
				Image imgToColor = Image(3, img.D, img.W, img.H);

				for (auto c = 0ull; c < 3ull; c++)
					for (auto d = 0ull; d < img.D; d++)
						for (auto h = 0ull; h < img.H; h++)
							for (auto w = 0ull; w < img.W; w++)
								imgToColor(c, d, h, w) = img(0, d, h, w);

				return imgToColor;
			}
			else
				return img;
		}

		static Image MirrorPad(const Image& image, const UInt depth, const UInt height, const UInt width)
		{
			Image img(image.C, image.D + (depth * 2), image.H + (height * 2), image.W + (width * 2));

			for (auto c = 0ull; c < image.C; c++)
			{
				for (auto d = 0ull; d < depth; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d, h, w) = image(c, d, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d, h, w + width) = image(c, d, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d, h, w + width + image.W) = image(c, d, height - (h + 1), image.W - (w + 1));
					}
					for (auto h = 0ull; h < image.H; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d, h + height, w) = image(c, d, h, width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth, h + height, w + width) = image(c, d, h, w);
						for (auto w = 0ull; w < width; w++)
							img(c, d, h + height, w + width + image.W) = image(c, d, h, image.W - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d, h + height + image.H, w) = image(c, d, image.H - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d, h + height + image.H, w + width) = image(c, d, image.H - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d, h + height + image.H, w + width + image.W) = image(c, d, image.H - (h + 1), image.W - (w + 1));
					}
				}
				for (auto d = 0ull; d < image.D; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h, w) = image(c, d + depth, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth, h, w + width) = image(c, d + depth, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h, w + width + image.W) = image(c, d + depth, height - (h + 1), image.W - (w + 1));
					}
					for (auto h = 0ull; h < image.H; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h + height, w) = image(c, d + depth, h, width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth, h + height, w + width) = image(c, d + depth, h, w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h + height, w + width + image.W) = image(c, d + depth, h, image.W - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h + height + image.H, w) = image(c, d + depth, image.H - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth, h + height + image.H, w + width) = image(c, d + depth, image.H - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth, h + height + image.H, w + width + image.W) = image(c, d + depth, image.H - (h + 1), image.W - (w + 1));
					}
				}
				for (auto d = 0ull; d < depth; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h, w) = image(c, d + depth + image.D, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth + image.D, h, w + width) = image(c, d + depth + image.D, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h, w + width + image.W) = image(c, d + depth + image.D, height - (h + 1), image.W - (w + 1));
					}
					for (auto h = 0ull; h < image.H; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h + height, w) = image(c, d + depth + image.D, h, width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth + image.D, h + height, w + width) = image(c, d + depth + image.D, h, w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h + height, w + width + image.W) = image(c, d + depth + image.D, h, image.W - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h + height + image.H, w) = image(c, d + depth + image.D, image.H - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth + image.D, h + height + image.H, w + width) = image(c, d + depth + image.D, image.H - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							img(c, d + depth + image.D, h + height + image.H, w + width + image.W) = image(c, d + depth + image.D, image.H - (h + 1), image.W - (w + 1));
					}
				}
			}

			return img;
		}


		inline static Image Padding(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean, const bool mirrorPad = false)
		{
			return mirrorPad ? Image::MirrorPad(image, depth, height, width) : Image::ZeroPad(image, depth, height, width, mean);
		}

		static Image Posterize(const Image& image, const UInt levels = 16)
		{
			Image img(image.C, image.D, image.H, image.W);

			auto palette = std::vector<Byte>(256);
			const auto q = 256ull / levels;
			for (auto c = 0ull; c < 255ull; c++)
				palette[c] = Saturate<UInt>((((c / q) * q) * levels) / (levels - 1));

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = palette[image(c, d, h, w)];

			return img;
		}
		
		static Image RandomCrop(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image img(image.C, depth, height, width);

			auto channelMean = static_cast<T>(0);
			for (auto c = 0ull; c < img.C; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = static_cast<T>(mean[c]);
				
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = channelMean;
			}
			
			const auto minD = std::min(img.D, image.D);
			const auto minH = std::min(img.H, image.H);
			const auto minW = std::min(img.W, image.W);
			
			const auto srcDdelta = img.D < image.D ? UniformInt<UInt>(0, image.D - img.D) : 0ull;
			const auto srcHdelta = img.H < image.H ? UniformInt<UInt>(0, image.H - img.H) : 0ull;
			const auto srcWdelta = img.W < image.W ? UniformInt<UInt>(0, image.W - img.W) : 0ull;
			
			const auto dstDdelta = img.D > image.D ? UniformInt<UInt>(0, img.D - image.D) : 0ull;
			const auto dstHdelta = img.H > image.H ? UniformInt<UInt>(0, img.H - image.H) : 0ull;
			const auto dstWdelta = img.W > image.W ? UniformInt<UInt>(0, img.W - image.W) : 0ull;

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < minD; d++)
					for (auto h = 0ull; h < minH; h++)
						for (auto w = 0ull; w < minW; w++)
							img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			
			return img;
		}

		static Image RandomCutout(const Image& image, const std::vector<Float>& mean)
		{
			Image img(image);

			const auto centerH = UniformInt<UInt>(0, img.H);
			const auto centerW = UniformInt<UInt>(0, img.W);
			const auto rangeH = UniformInt<UInt>(img.H / 8, img.H / 4);
			const auto rangeW = UniformInt<UInt>(img.W / 8, img.W / 4);
			const auto startH = static_cast<long>(centerH) - static_cast<long>(rangeH) > 0l ? centerH - rangeH : 0ull;
			const auto startW = static_cast<long>(centerW) - static_cast<long>(rangeW) > 0l ? centerW - rangeW : 0ull;
			const auto enheight = centerH + rangeH < img.H ? centerH + rangeH : img.H;
			const auto enwidth = centerW + rangeW < img.W ? centerW + rangeW : img.W;

			auto channelMean =  static_cast<T>(0);
			for (auto c = 0ull; c < img.C; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean =  static_cast<T>(mean[c]);
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = startH; h < enheight; h++)
						for (auto w = startW; w < enwidth; w++)
							img(c, d, h, w) = channelMean;
			}

			return img;
		}

		static Image RandomCutMix(const Image& image, const Image& imageMix, double* lambda)
		{
			Image img(image);
			Image imgMix(imageMix);

			const auto cutRate = std::sqrt(1.0 - *lambda);
			const auto cutH = static_cast<int>(static_cast<double>(img.H) * cutRate);
			const auto cutW = static_cast<int>(static_cast<double>(img.W) * cutRate);
			const auto cy = UniformInt<int>(0, static_cast<int>(img.H));
			const auto cx = UniformInt<int>(0, static_cast<int>(img.W));
			const auto bby1 = Clamp<int>(cy - cutH / 2, 0, static_cast<int>(img.H));
			const auto bby2 = Clamp<int>(cy + cutH / 2, 0, static_cast<int>(img.H));
			const auto bbx1 = Clamp<int>(cx - cutW / 2, 0, static_cast<int>(img.W));
			const auto bbx2 = Clamp<int>(cx + cutW / 2, 0, static_cast<int>(img.W));

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = bby1; h < bby2; h++)
						for (auto w = bbx1; w < bbx2; w++)
							img(c, d, h, w) = imgMix(c, d, h, w);

			*lambda = 1.0 - (static_cast<double>((bbx2 - bbx1) * (bby2 - bby1)) / static_cast<double>(img.H * img.W));

			return img;
		}

		static Image Resize(const Image& image, const UInt depth, const UInt height, const UInt width, const Interpolations interpolation)
		{
			auto imgSource = ImageToCImg(image);

			switch (interpolation)
			{
			case Interpolations::Cubic:
				imgSource.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C), 5, 0);
				break;
			case Interpolations::Linear:
				imgSource.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C), 3, 0);
				break;
			case Interpolations::Nearest:
				imgSource.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C), 1, 0);
				break;
			}
			
			auto img = CImgToImage(imgSource);

			return img;
		}

		static Image Rotate(const Image& image, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean)
		{
			auto imgSource = ImageToCImg(ZeroPad(image, image.D / 2, image.H / 2, image.W / 2, mean));

			switch (interpolation)
			{
			case Interpolations::Cubic:
				imgSource.rotate(angle, 2, 0);
				break;
			case Interpolations::Linear:
				imgSource.rotate(angle, 1, 0);
				break;
			case Interpolations::Nearest:
				imgSource.rotate(angle, 0, 0);
				break;
			}
			
			auto img = CImgToImage(imgSource);

			return Crop(img, Positions::Center, image.D, image.H, image.W, mean);
		}
			
		// magnitude = 0   // blurred image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Sharpness(const Image& image, const Float magnitude)
		{
			auto imgSource = ImageToCImg(image);

			imgSource.sharpen(magnitude, false);

			auto img = CImgToImage(imgSource);

			return img;
		}

		static Image Solarize(const Image& image, const T treshold = 128)
		{
			Image img(image.C, image.D, image.H, image.W);

			constexpr T maximum = std::is_floating_point_v<T> ?  static_cast<T>(1) :  static_cast<T>(255);

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = (image(c, d, h, w) < treshold) ? image(c, d, h, w) : (maximum - image(c, d, h, w));

			return img;
		}
		
		static Image Translate(const Image& image, const int height, const int width, const std::vector<Float>& mean)
		{
			if (height == 0 && width == 0)
				return image;

			if (width <= -static_cast<int>(image.W) || width >= static_cast<int>(image.W) || height <= -static_cast<int>(image.H) || height >= static_cast<int>(image.H))
			{
				Image img(image.C, image.D, image.H, image.W);
				
				T channelMean =  static_cast<T>(0);
				for (auto c = 0ull; c < image.C; c++)
				{
					if constexpr (!std::is_floating_point_v<T>)
						channelMean =  static_cast<T>(mean[c]);

					for (auto d = 0ull; d < image.D; d++)
						for (auto h = 0ull; h < image.H; h++)
							for (auto w = 0ull; w < image.W; w++)
								img(c, d, h, w) = channelMean;
				}

				return img;
			}

			auto imgSource = ImageToCImg(image);

			if (width != 0)
			{
				if (width < 0)
					cimg_forYZC(imgSource, y, z, c)
					{
						std::memmove(imgSource.data(0, y, z, c), imgSource.data(-width, y, z, c), static_cast<UInt>(imgSource._width + width) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(imgSource.data(imgSource._width + width, y, z, c), 0, -width * sizeof(T));
						else
							std::memset(imgSource.data(imgSource._width + width, y, z, c), (int)mean[c], -width * sizeof(T));
					}
				else
					cimg_forYZC(imgSource, y, z, c)
					{
						std::memmove(imgSource.data(width, y, z, c), imgSource.data(0, y, z, c), static_cast<UInt>(imgSource._width - width) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(imgSource.data(0, y, z, c), 0, width * sizeof(T));
						else
							std::memset(imgSource.data(0, y, z, c), (int)mean[c], width * sizeof(T));
					}
			}

			if (height != 0)
			{
				if (height < 0)
					cimg_forZC(imgSource, z, c)
					{
						std::memmove(imgSource.data(0, 0, z, c), imgSource.data(0, -height, z, c), static_cast<UInt>(imgSource._width) * static_cast<UInt>(imgSource._height + height) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(imgSource.data(0, imgSource._height + height, z, c), 0, -height * static_cast<UInt>(imgSource._width) * sizeof(T));
						else
							std::memset(imgSource.data(0, imgSource._height + height, z, c), (int)mean[c], -height * static_cast<UInt>(imgSource._width) * sizeof(T));
					}
				else
					cimg_forZC(imgSource, z, c)
					{
						std::memmove(imgSource.data(0, height, z, c), imgSource.data(0, 0, z, c), static_cast<UInt>(imgSource._width) * static_cast<UInt>(imgSource._height - height) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(imgSource.data(0, 0, z, c), 0, height * static_cast<UInt>(imgSource._width) * sizeof(T));
						else
							std::memset(imgSource.data(0, 0, z, c), (int)mean[c], height * static_cast<UInt>(imgSource._width) * sizeof(T));
					}
			}

			auto img = CImgToImage(imgSource);

			return img;
		}

		static Image VerticalMirror(const Image& image)
		{
			Image img(image.C, image.D, image.H, image.W);

			for (auto c = 0ull; c < img.C; c++)
				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = image(c, d, image.H - 1 - h, w);

			return img;
		}
		
		static Image ZeroPad(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image img(image.C, image.D + (depth * 2), image.H + (height * 2), image.W + (width * 2));

			T channelMean = static_cast<T>(0);
			for (auto c = 0ull; c < img.C; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = static_cast<T>(mean[c]);

				for (auto d = 0ull; d < img.D; d++)
					for (auto h = 0ull; h < img.H; h++)
						for (auto w = 0ull; w < img.W; w++)
							img(c, d, h, w) = channelMean;
			}

			for (auto c = 0ull; c < image.C; c++)
				for (auto d = 0ull; d < image.D; d++)
					for (auto h = 0ull; h < image.H; h++)
						for (auto w = 0ull; w < image.W; w++)
							img(c, d + depth, h + height, w + width) = image(c, d, h, w);

			return img;
		}
	};
}