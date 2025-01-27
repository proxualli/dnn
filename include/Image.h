#pragma once
#include "Utils.h"

#ifdef cimg_use_jpeg
#include "jpeglib.h"
#include "jerror.h"
#endif
#undef cimg_display
#ifndef NDEBUG
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#define cimg_display 2
#else
#define cimg_display 0
#endif
#else	
#define cimg_display 0
#endif
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
		UInt Channels;
		UInt Depth;
		UInt Height;
		UInt Width;
		
		Image() :
			Channels(0),
			Depth(0),
			Height(0),
			Width(0),
			Data(VectorT())
		{
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w, const VectorT& image) :
			Channels(c),
			Depth(d),
			Height(h),
			Width(w),
			Data(image)
		{
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w, const T* image) :
			Channels(c),
			Depth(d),
			Height(h),
			Width(w),
			Data(VectorT(c * d * h * w))
		{
			std::memcpy(Data.data(), image, c * d * h * w * sizeof(T));
		}

		Image(const UInt c, const UInt d, const UInt h, const UInt w) :
			Channels(c),
			Depth(d),
			Height(h),
			Width(w),
			Data(VectorT(c * d * h * w))
		{
		}

		~Image() = default;

		T& operator()(const UInt c, const UInt d, const UInt h, const UInt w)
		{
			return Data[w + (h * Width) + (d * Height * Width) + (c * Depth * Height * Width)];
		}

		const T& operator()(const UInt c, const UInt d, const UInt h, const UInt w) const
		{
			return Data[w + (h * Width) + (d * Height * Width) + (c * Depth * Height * Width)];
		}

		T* data()
		{
			return Data.data();
		}

		const T* data() const
		{
			return Data.data();
		}

		auto C() const
		{
			return Channels;
		}
		
		auto D() const
		{
			return Depth;
		}
		
		auto H() const
		{
			return Height;
		}
		
		auto W() const
		{
			return Width;
		}

		auto Area() const
		{
			return Height * Width;
		}

		auto ChannelSize() const
		{
			return Depth * Height * Width;
		}

		auto Size() const
		{
			return Channels * Depth * Height * Width;
		}
		
		static cimg_library::CImg<Float> ImageToCImgFloat(const Image& image)
		{
			auto dstImage = cimg_library::CImg<Float>(uint32_t(image.Width), uint32_t(image.Height), uint32_t(image.Depth), uint32_t(image.Channels));

			for (auto c = 0ull; c < image.Channels; c++)
				for (auto d = 0ull; d < image.Depth; d++)
					for (auto h = 0ull; h < image.Height; h++)
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(uint32_t(w), uint32_t(h), uint32_t(d), uint32_t(c)) = image(c, d, h, w);

				return dstImage;
		}

		static cimg_library::CImg<T> ImageToCImg(const Image& image)
		{
			return cimg_library::CImg<T>(image.data(), uint32_t(image.Width), uint32_t(image.Height), uint32_t(image.Depth), uint32_t(image.Channels));
		}

		static Image CImgToImage(const cimg_library::CImg<T>& image)
		{
			return Image(image._spectrum, image._depth, image._height, image._width, image.data());
		}

		static Image AutoAugment(const Image& image, const UInt padD, const UInt padH, const UInt padW, const std::vector<Float>& mean, const bool mirrorPad)
		{
			Image dstImage(image);

			auto operation = UniformInt<int>(0, 24);

			switch (operation)
			{
			case 1:
			case 3:
			case 5:
				dstImage = Padding(dstImage, padD, padH, padW, mean, mirrorPad);
				break;
			}

			switch (operation)
			{
			case 0:
			{
				if (Bernoulli<bool>(Float(0.1)))
					dstImage = Invert(dstImage);

				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						dstImage = Contrast(dstImage, FloatLevel(6));
					else
						dstImage = Contrast(dstImage, FloatLevel(4));
				}
			}
			break;

			case 1:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Rotate(dstImage, FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
					else
						dstImage = Rotate(dstImage, -FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, 0, IntLevel(7), mean);
					else
						dstImage = Translate(dstImage, 0, -IntLevel(7), mean);
				}
			}
			break;

			case 2:
			{
				if (Bernoulli<bool>(Float(0.8)))
				{
					if (Bernoulli<bool>())
						dstImage = Sharpness(dstImage, FloatLevel(2));
					else
						dstImage = Sharpness(dstImage, FloatLevel(8));
				}

				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						dstImage = Sharpness(dstImage, FloatLevel(3));
					else
						dstImage = Sharpness(dstImage, FloatLevel(7));
				}
			}
			break;

			case 3:
			{
				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						dstImage = Rotate(dstImage, FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
					else
						dstImage = Rotate(dstImage, -FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, IntLevel(7), 0, mean);
					else
						dstImage = Translate(dstImage, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 4:
			{
				if (Bernoulli<bool>())
					dstImage = AutoContrast(dstImage);

				if (Bernoulli<bool>(Float(0.9)))
					dstImage = Equalize(dstImage);
			}
			break;

			case 5:
			{
				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						dstImage = Rotate(dstImage, FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
					else
						dstImage = Rotate(dstImage, -FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						dstImage = Posterize(dstImage, 32);
					else
						dstImage = Posterize(dstImage, 64);
				}
			}
			break;

			case 6:
			{
				if (Bernoulli<bool>(Float(0.4)))
				{
					if (Bernoulli<bool>())
						dstImage = Color(dstImage, FloatLevel(3));
					else
						dstImage = Color(dstImage, FloatLevel(7));
				}

				if (Bernoulli<bool>(Float(0.6)))
				{
					if (Bernoulli<bool>())
						dstImage = Brightness(dstImage, FloatLevel(7));
					else
						dstImage = Brightness(dstImage, FloatLevel(3));
				}
			}
			break;

			case 7:
			{
				if (Bernoulli<bool>(Float(0.3)))
				{
					if (Bernoulli<bool>())
						dstImage = Sharpness(dstImage, FloatLevel(9));
					else
						dstImage = Sharpness(dstImage, FloatLevel(1));
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Brightness(dstImage, FloatLevel(8));
					else
						dstImage = Brightness(dstImage, FloatLevel(2));
				}
			}
			break;

			case 8:
			{
				if (Bernoulli<bool>(Float(0.6)))
					dstImage = Equalize(dstImage);

				if (Bernoulli<bool>())
					dstImage = Equalize(dstImage);
			}
			break;

			case 9:
			{
				if (Bernoulli<bool>(Float(0.6)))
				{
					if (Bernoulli<bool>())
						dstImage = Contrast(dstImage, FloatLevel(7));
					else
						dstImage = Contrast(dstImage, FloatLevel(3));
				}

				if (Bernoulli<bool>(Float(Float(0.6))))
				{
					if (Bernoulli<bool>(Float(0.5)))
						dstImage = Sharpness(dstImage, FloatLevel(6));
					else
						dstImage = Sharpness(dstImage, FloatLevel(4));
				}
			}
			break;

			case 10:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Color(dstImage, FloatLevel(7));
					else
						dstImage = Color(dstImage, FloatLevel(3));
				}

				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, 0, IntLevel(8), mean);
					else
						dstImage = Translate(dstImage, 0, -IntLevel(8), mean);
				}
			}
			break;

			case 11:
			{
				if (Bernoulli<bool>(Float(0.3)))
					dstImage = Equalize(dstImage);

				if (Bernoulli<bool>(Float(0.4)))
					dstImage = AutoContrast(dstImage);
			}
			break;

			case 12:
			{
				if (Bernoulli<bool>(Float(0.4)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, IntLevel(3), 0, mean);
					else
						dstImage = Translate(dstImage, -IntLevel(3), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.2)))
					dstImage = Sharpness(dstImage, FloatLevel(6));
			}
			break;

			case 13:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						dstImage = Brightness(dstImage, FloatLevel(6));
					else
						dstImage = Brightness(dstImage, FloatLevel(4));
				}

				if (Bernoulli<bool>(Float(0.2)))
				{
					if (Bernoulli<bool>())
						dstImage = Color(dstImage, FloatLevel(8));
					else
						dstImage = Color(dstImage, FloatLevel(2));
				}
			}
			break;

			case 14:
			{
				if (Bernoulli<bool>())
				{
					if (Bernoulli<bool>())
						dstImage = Solarize(dstImage, static_cast<T>(IntLevel(2, 0, 256)));
					else
						dstImage = Solarize(dstImage, static_cast<T>(IntLevel(8, 0, 256)));
				}
			}
			break;

			case 15:
			{
				if (Bernoulli<bool>(Float(0.2)))
					dstImage = Equalize(dstImage);

				if (Bernoulli<bool>(Float(0.6)))
					dstImage = AutoContrast(dstImage);
			}
			break;

			case 16:
			{
				if (Bernoulli<bool>(Float(0.2)))
					dstImage = Equalize(dstImage);

				if (Bernoulli<bool>(Float(0.6)))
					dstImage = Equalize(dstImage);
			}
			break;

			case 17:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						dstImage = Color(dstImage, FloatLevel(8));
					else
						dstImage = Color(dstImage, FloatLevel(2));
				}

				if (Bernoulli<bool>(Float(0.6)))
					dstImage = Equalize(dstImage);
			}
			break;

			case 18:
			{
				if (Bernoulli<bool>(Float(0.8)))
					dstImage = AutoContrast(dstImage);

				if (Bernoulli<bool>(Float(0.2)))
					dstImage = Solarize(dstImage, static_cast<T>(IntLevel(8, 0, 256)));
			}
			break;

			case 19:
			{
				if (Bernoulli<bool>(Float(0.1)))
					dstImage = Brightness(dstImage, FloatLevel(3));

				if (Bernoulli<bool>(Float(0.7)))
					dstImage = Color(dstImage, FloatLevel(4));
			}
			break;

			case 20:
			{
				if (Bernoulli<bool>(Float(0.4)))
					dstImage = Solarize(dstImage, static_cast<T>(IntLevel(5, 0, 256)));

				if (Bernoulli<bool>(Float(0.9)))
					dstImage = AutoContrast(dstImage);
			}
			break;

			case 21:
			{
				if (Bernoulli<bool>(Float(0.9)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, IntLevel(7), 0, mean);
					else
						dstImage = Translate(dstImage, -IntLevel(7), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, IntLevel(7), 0, mean);
					else
						dstImage = Translate(dstImage, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 22:
			{
				if (Bernoulli<bool>(Float(0.9)))
					dstImage = AutoContrast(dstImage);

				if (Bernoulli<bool>(Float(0.8)))
					dstImage = Solarize(dstImage, static_cast<T>(IntLevel(3, 0, 256)));
			}
			break;

			case 23:
			{
				if (Bernoulli<bool>(Float(0.8)))
					dstImage = Equalize(dstImage);

				if (Bernoulli<bool>(Float(0.1)))
					dstImage = Invert(dstImage);
			}
			break;

			case 24:
			{
				if (Bernoulli<bool>(Float(0.7)))
				{
					if (Bernoulli<bool>())
						dstImage = Translate(dstImage, IntLevel(8), 0, mean);
					else
						dstImage = Translate(dstImage, -IntLevel(8), 0, mean);
				}

				if (Bernoulli<bool>(Float(0.9)))
					dstImage = AutoContrast(dstImage);
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
				dstImage = Padding(dstImage, padD, padH, padW, mean, mirrorPad);
				break;
			}

			return dstImage;
		}

		static Image AutoContrast(const Image& image)
		{
			const T maximum = std::is_floating_point_v<T> ? T(1) : T(255);
			
			auto srcImage = ImageToCImg(image);

			srcImage.normalize(0, maximum);

			auto dstImage = CImgToImage(srcImage);

			return dstImage;
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Brightness(const Image& image, const Float magnitude)
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			const auto delta = (magnitude - Float(1)) / 2;

			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 2) = cimg_library::cimg::cut<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 2) + delta, 0, 1);

			srcImage.HSLtoRGB();

			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = Saturate<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), uint32_t(c)));

			return dstImage;
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Color(const Image& image, const Float magnitude)
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 0) = cimg_library::cimg::cut<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 0) * magnitude, 0, 360);

			srcImage.HSLtoRGB();

			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);
			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = Saturate<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), uint32_t(c)));

			return dstImage;
		}

		static Image ColorCast(const Image& image, const UInt angle)
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			const auto shift = Float(Bernoulli<bool>() ? UniformInt<int>( -int(angle), int(angle)) : 0);

			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 0) = cimg_library::cimg::cut<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 0) + shift, 0, 360);
				
			srcImage.HSLtoRGB();

			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);
			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = Saturate<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), uint32_t(c)));
					
			return dstImage;
		}
		
		// magnitude = 0   // gray image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Contrast(const Image& image, const Float magnitude)
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 1) = cimg_library::cimg::cut<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), 1) * magnitude, 0, 1);

			srcImage.HSLtoRGB();

			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);
			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = Saturate<Float>(srcImage(uint32_t(w), uint32_t(h), uint32_t(d), uint32_t(c)));

			return dstImage;
		}

		static Image Crop(const Image& image, const Positions position, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image dstImage(image.Channels, depth, height, width);

			for (auto c = 0ull; c < dstImage.Channels; c++)
			{
				const T channelMean = std::is_floating_point_v<T> ? T(0) : T(mean[c]);
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = channelMean;
			}

			const auto minDepth = std::min(dstImage.Depth, image.Depth);
			const auto minHeight = std::min(dstImage.Height, image.Height);
			const auto minWidth = std::min(dstImage.Width, image.Width);

			const auto srcDdelta = dstImage.Depth < image.Depth ? (image.Depth - dstImage.Depth) / 2: 0ull;
			const auto dstDdelta = dstImage.Depth > image.Depth ? (dstImage.Depth - image.Depth) / 2: 0ull;

			switch (position)
			{
			case Positions::Center:
			{
				const auto srcHdelta = dstImage.Height < image.Height ? (image.Height - dstImage.Height) / 2 : 0ull;
				const auto dstHdelta = dstImage.Height > image.Height ? (dstImage.Height - image.Height) / 2 : 0ull;
				const auto srcWdelta = dstImage.Width < image.Width ? (image.Width - dstImage.Width) / 2 : 0ull;
				const auto dstWdelta = dstImage.Width > image.Width ? (dstImage.Width - image.Width) / 2 : 0ull;

				for (auto c = 0ull; c < dstImage.Channels; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								dstImage(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::TopLeft:
			{
				for (auto c = 0ull; c < dstImage.Channels; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								dstImage(c, d + dstDdelta, h, w) = image(c, d + srcDdelta, h, w);
			}
			break;

			case Positions::TopRight:
			{
				const auto srcWdelta = dstImage.Width < image.Width ? (image.Width - dstImage.Width) : 0ull;
				const auto dstWdelta = dstImage.Width > image.Width ? (dstImage.Width - image.Width) : 0ull;

				for (auto c = 0ull; c < dstImage.Channels; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								dstImage(c, d + dstDdelta, h, w + dstWdelta) = image(c, d + srcDdelta, h, w + srcWdelta);
			}
			break;

			case Positions::BottomRight:
			{
				const auto srcHdelta = dstImage.Height < image.Height ? (image.Height - dstImage.Height) : 0ull;
				const auto dstHdelta = dstImage.Height > image.Height ? (dstImage.Height - image.Height) : 0ull;
				const auto srcWdelta = dstImage.Width < image.Width ? (image.Width - dstImage.Width) : 0ull;
				const auto dstWdelta = dstImage.Width > image.Width ? (dstImage.Width - image.Width) : 0ull;

				for (auto c = 0ull; c < dstImage.Channels; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								dstImage(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::BottomLeft:
			{
				const auto srcHdelta = dstImage.Height < image.Height ? (image.Height - dstImage.Height) : 0ull;
				const auto dstHdelta = dstImage.Height > image.Height ? (dstImage.Height - image.Height) : 0ull;

				for (auto c = 0ull; c < dstImage.Channels; c++)
					for (auto d = 0ull; d < minDepth; d++)
						for (auto h = 0ull; h < minHeight; h++)
							for (auto w = 0ull; w < minWidth; w++)
								dstImage(c, d + dstDdelta, h + dstHdelta, w) = image(c, d + srcDdelta, h + srcHdelta, w);
			}
			break;
			}

			return dstImage;
		}

		static Image Distorted(const Image& image, const Float scale, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean)
		{
			const auto zoom = Double(scale) / Double(100) * UniformReal<Double>(Double(-1), Double(1));
			const auto height = image.Height + UInt(Int(std::round(Double(image.Height) * zoom)));
			const auto width = image.Width + UInt(Int(std::round(Double(image.Width) * zoom)));

			return Image::Crop(Image::Rotate(Image::Resize(image, image.Depth, height, width, interpolation), angle * UniformReal<Float>( Float(-1), Float(1)), interpolation, mean), Positions::Center, image.Depth, image.Height, image.Width, mean);
		}

		static Image Dropout(const Image& image, const Float dropout, const std::vector<Float>& mean)
		{
			Image dstImage(image);
			
			for (auto d = 0ull; d < dstImage.Depth; d++)
				for (auto h = 0ull; h < dstImage.Height; h++)
					for (auto w = 0ull; w < dstImage.Width; w++)
						if (Bernoulli<bool>(dropout))
						{
							for (auto c = 0ull; c < dstImage.Channels; c++)
							{
								if constexpr (std::is_floating_point_v<T>)
									dstImage(c, d, h, w) = T(0);
								else
									dstImage(c, d, h, w) = T(mean[c]);
							}
						}

			return dstImage;
		}
		
		static Image Equalize(const Image& image)
		{
			auto srcImage = ImageToCImg(image);

			srcImage.equalize(256);

			auto dstImage = CImgToImage(srcImage);

			return dstImage;
		}

		static Float GetChannelMean(const Image& image, const UInt c)
		{
			auto mean = Float(0);
			auto correction = Float(0);

			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						KahanSum<Float>(Float(image(c, d, h, w)), mean, correction);
						
			mean /= Float(image.ChannelSize());

			return mean;
		}

		static Float GetChannelVariance(const Image& image, const UInt c)
		{
			const auto mean = Image::GetChannelMean(image, c);

			auto variance = Float(0);
			auto correction = Float(0);


			for (auto d = 0ull; d < image.Depth; d++)
				for (auto h = 0ull; h < image.Height; h++)
					for (auto w = 0ull; w < image.Width; w++)
						KahanSum<Float>(Square<Float>(Float(image(c, d, h, w)) - mean), variance, correction);

			variance /= Float(image.ChannelSize());
			variance = std::max(Float(0), variance);

			return variance;
		}

		static Float GetChannelStdDev(const Image& image, const UInt c)
		{
			return std::max(std::sqrt(GetChannelVariance(image, c)), Float(1) / std::sqrt(Float(image.ChannelSize())));
		}

		static Image HorizontalMirror(const Image& image)
		{
			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = image(c, d, h, image.Width - 1 - w);
			
			return dstImage;
		}

		static Image Invert(const Image& image)
		{
			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			constexpr T maximum = std::is_floating_point_v<T> ? T(1) : T(255);

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = maximum - image(c, d, h, w);

			return dstImage;
		}

#ifdef cimg_use_jpeg
		static Image LoadJPEG(const std::string& fileName, const bool forceColorFormat = false)
		{
			Image dstImage = CImgToImage(cimg_library::CImg<T>().get_load_jpeg(fileName.c_str()));

			if (forceColorFormat && dstImage.Channels == 1)
			{
				Image dstColorImage = Image(3, dstImage.Depth, dstImage.Width, dstImage.Height);

				for (auto c = 0ull; c < 3ull; c++)
					for (auto d = 0ull; d < dstImage.Depth; d++)
						for (auto h = 0ull; h < dstImage.Height; h++)
							for (auto w = 0ull; w < dstImage.Width; w++)
								dstColorImage(c, d, h, w) = dstImage(0, d, h, w);

				return dstColorImage;
			}
			else
				return dstImage;
		}
#endif

#ifdef cimg_use_png
		static Image LoadPNG(const std::string& fileName, const bool forceColorFormat = false)
		{
			auto bitsPerPixel = 0u;
			Image dstImage = CImgToImage(cimg_library::CImg<T>().get_load_png(fileName.c_str(), &bitsPerPixel));

			if (forceColorFormat && dstImage.Channels == 1)
			{
				Image dstColorImage = Image(3, dstImage.Depth, dstImage.Width, dstImage.Height);

				for (auto c = 0ull; c < 3ull; c++)
					for (auto d = 0ull; d < dstImage.Depth; d++)
						for (auto h = 0ull; h < dstImage.Height; h++)
							for (auto w = 0ull; w < dstImage.Width; w++)
								dstColorImage(c, d, h, w) = dstImage(0, d, h, w);

				return dstColorImage;
			}
			else
				return dstImage;
		}
#endif

		static Image MirrorPad(const Image& image, const UInt depth, const UInt height, const UInt width)
		{
			Image dstImage(image.Channels, image.Depth + (depth * 2), image.Height + (height * 2), image.Width + (width * 2));

			for (auto c = 0ull; c < image.Channels; c++)
			{
				for (auto d = 0ull; d < depth; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h, w) = image(c, d, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d, h, w + width) = image(c, d, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h, w + width + image.Width) = image(c, d, height - (h + 1), image.Width - (w + 1));
					}
					for (auto h = 0ull; h < image.Height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h + height, w) = image(c, d, h, width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth, h + height, w + width) = image(c, d, h, w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h + height, w + width + image.Width) = image(c, d, h, image.Width - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h + height + image.Height, w) = image(c, d, image.Height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d, h + height + image.Height, w + width) = image(c, d, image.Height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d, h + height + image.Height, w + width + image.Width) = image(c, d, image.Height - (h + 1), image.Width - (w + 1));
					}
				}
				for (auto d = 0ull; d < image.Depth; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h, w) = image(c, d + depth, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth, h, w + width) = image(c, d + depth, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h, w + width + image.Width) = image(c, d + depth, height - (h + 1), image.Width - (w + 1));
					}
					for (auto h = 0ull; h < image.Height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h + height, w) = image(c, d + depth, h, width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth, h + height, w + width) = image(c, d + depth, h, w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h + height, w + width + image.Width) = image(c, d + depth, h, image.Width - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h + height + image.Height, w) = image(c, d + depth, image.Height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth, h + height + image.Height, w + width) = image(c, d + depth, image.Height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth, h + height + image.Height, w + width + image.Width) = image(c, d + depth, image.Height - (h + 1), image.Width - (w + 1));
					}
				}
				for (auto d = 0ull; d < depth; d++)
				{
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h, w) = image(c, d + depth + image.Depth, height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth + image.Depth, h, w + width) = image(c, d + depth + image.Depth, height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h, w + width + image.Width) = image(c, d + depth + image.Depth, height - (h + 1), image.Width - (w + 1));
					}
					for (auto h = 0ull; h < image.Height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h + height, w) = image(c, d + depth + image.Depth, h, width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth + image.Depth, h + height, w + width) = image(c, d + depth + image.Depth, h, w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h + height, w + width + image.Width) = image(c, d + depth + image.Depth, h, image.Width - (w + 1));
					}
					for (auto h = 0ull; h < height; h++)
					{
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h + height + image.Height, w) = image(c, d + depth + image.Depth, image.Height - (h + 1), width - (w + 1));
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth + image.Depth, h + height + image.Height, w + width) = image(c, d + depth + image.Depth, image.Height - (h + 1), w);
						for (auto w = 0ull; w < width; w++)
							dstImage(c, d + depth + image.Depth, h + height + image.Height, w + width + image.Width) = image(c, d + depth + image.Depth, image.Height - (h + 1), image.Width - (w + 1));
					}
				}
			}

			return dstImage;
		}

		inline static Image Padding(const Image& image, const UInt padD, const UInt padH, const UInt padW, const std::vector<Float>& mean, const bool mirrorPad = false)
		{
			return mirrorPad ? Image::MirrorPad(image, padD, padH, padW) : Image::ZeroPad(image, padD, padH, padW, mean);
		}

		static Image Posterize(const Image& image, const UInt levels = 16)
		{
			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			auto palette = std::vector<Byte>(256);
			const auto q = 256ull / levels;
			for (auto c = 0ull; c < 255ull; c++)
				palette[c] = Saturate<UInt>((((c / q) * q) * levels) / (levels - 1));

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = palette[image(c, d, h, w)];

			return dstImage;
		}
		
		static Image RandomCrop(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image dstImage(image.Channels, depth, height, width);

			auto channelMean = T(0);
			for (auto c = 0ull; c < dstImage.Channels; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = T(mean[c]);
				
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = channelMean;
			}
			
			const auto minD = std::min(dstImage.Depth, image.Depth);
			const auto minH = std::min(dstImage.Height, image.Height);
			const auto minW = std::min(dstImage.Width, image.Width);
			
			const auto srcDdelta = dstImage.Depth < image.Depth ? UniformInt<UInt>(0, image.Depth - dstImage.Depth) : 0ull;
			const auto srcHdelta = dstImage.Height < image.Height ? UniformInt<UInt>(0, image.Height - dstImage.Height) : 0ull;
			const auto srcWdelta = dstImage.Width < image.Width ? UniformInt<UInt>(0, image.Width - dstImage.Width) : 0ull;
			
			const auto dstDdelta = dstImage.Depth > image.Depth ? UniformInt<UInt>(0, dstImage.Depth - image.Depth) : 0ull;
			const auto dstHdelta = dstImage.Height > image.Height ? UniformInt<UInt>(0, dstImage.Height - image.Height) : 0ull;
			const auto dstWdelta = dstImage.Width > image.Width ? UniformInt<UInt>(0, dstImage.Width - image.Width) : 0ull;

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < minD; d++)
					for (auto h = 0ull; h < minH; h++)
						for (auto w = 0ull; w < minW; w++)
							dstImage(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			
			return dstImage;
		}

		static Image RandomCutout(const Image& image, const std::vector<Float>& mean)
		{
			Image dstImage(image);

			const auto centerH = UniformInt<UInt>(0, dstImage.Height);
			const auto centerW = UniformInt<UInt>(0, dstImage.Width);
			const auto rangeH = UniformInt<UInt>(dstImage.Height / 8, dstImage.Height / 4);
			const auto rangeW = UniformInt<UInt>(dstImage.Width / 8, dstImage.Width / 4);
			const auto startH = long(centerH) - long(rangeH) > 0 ? centerH - rangeH : 0ull;
			const auto startW = long(centerW) - long(rangeW) > 0 ? centerW - rangeW : 0ull;
			const auto endH = centerH + rangeH < dstImage.Height ? centerH + rangeH : dstImage.Height;
			const auto endW = centerW + rangeW < dstImage.Width ? centerW + rangeW : dstImage.Width;

			auto channelMean = T(0);
			for (auto c = 0ull; c < dstImage.Channels; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = T(mean[c]);
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = startH; h < endH; h++)
						for (auto w = startW; w < endW; w++)
							dstImage(c, d, h, w) = channelMean;
			}

			return dstImage;
		}

		static Image RandomCutMix(const Image& image, const Image& imageMix, double* lambda)
		{
			Image dstImage(image);
			Image mixImage(imageMix);

			const auto cutRate = std::sqrt(1.0 - *lambda);
			const auto cutH = static_cast<int>(double(dstImage.Height) * cutRate);
			const auto cutW = static_cast<int>(double(dstImage.Width) * cutRate);
			const auto cy = UniformInt<int>(0, int(dstImage.Height));
			const auto cx = UniformInt<int>(0, int(dstImage.Width));
			const auto bby1 = Clamp<int>(cy - cutH / 2, 0, int(dstImage.Height));
			const auto bby2 = Clamp<int>(cy + cutH / 2, 0, int(dstImage.Height));
			const auto bbx1 = Clamp<int>(cx - cutW / 2, 0, int(dstImage.Width));
			const auto bbx2 = Clamp<int>(cx + cutW / 2, 0, int(dstImage.Width));

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = bby1; h < bby2; h++)
						for (auto w = bbx1; w < bbx2; w++)
							dstImage(c, d, h, w) = mixImage(c, d, h, w);

			*lambda = 1.0 - ((double(bbx2) - double(bbx1)) * (double(bby2) - double(bby1)) / double(dstImage.Height * dstImage.Width));

			return dstImage;
		}

		static Image Resize(const Image& image, const UInt depth, const UInt height, const UInt width, const Interpolations interpolation)
		{
			auto srcImage = ImageToCImg(image);

			switch (interpolation)
			{
			case Interpolations::Cubic:
				srcImage.resize(int(width), int(height), int(depth), int(image.Channels), 5, 0);
				break;
			case Interpolations::Linear:
				srcImage.resize(int(width), int(height), int(depth), int(image.Channels), 3, 0);
				break;
			case Interpolations::Nearest:
				srcImage.resize(int(width), int(height), int(depth), int(image.Channels), 1, 0);
				break;
			}
			
			auto dstImage = CImgToImage(srcImage);

			return dstImage;
		}

		static Image Rotate(const Image& image, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean)
		{
			auto srcImage = ImageToCImg(ZeroPad(image, image.Depth / 2, image.Height / 2, image.Width / 2, mean));

			switch (interpolation)
			{
			case Interpolations::Cubic:
				srcImage.rotate(angle, 2, 0);
				break;
			case Interpolations::Linear:
				srcImage.rotate(angle, 1, 0);
				break;
			case Interpolations::Nearest:
				srcImage.rotate(angle, 0, 0);
				break;
			}
			
			auto dstImage = CImgToImage(srcImage);

			return Crop(dstImage, Positions::Center, image.Depth, image.Height, image.Width, mean);
		}
			
		// magnitude = 0   // blurred image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Sharpness(const Image& image, const Float magnitude)
		{
			auto srcImage = ImageToCImg(image);

			srcImage.sharpen(magnitude, false);

			auto dstImage = CImgToImage(srcImage);

			return dstImage;
		}

		static Image Solarize(const Image& image, const T treshold = 128)
		{
			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			constexpr T maximum = std::is_floating_point_v<T> ? T(1) : T(255);

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = (image(c, d, h, w) < treshold) ? image(c, d, h, w) : (maximum - image(c, d, h, w));

			return dstImage;
		}
		
		static Image Translate(const Image& image, const int deltaH, const int deltaW, const std::vector<Float>& mean)
		{
			if (deltaH == 0 && deltaW == 0)
				return image;

			if (deltaW <= -int(image.Width) || deltaW >= int(image.Width) || deltaH <= -int(image.Height) || deltaH >= int(image.Height))
			{
				Image dstImage(image.Channels, image.Depth, image.Height, image.Width);
				
				T channelMean = T(0);
				for (auto c = 0ull; c < image.Channels; c++)
				{
					if constexpr (!std::is_floating_point_v<T>)
						channelMean = T(mean[c]);

					for (auto d = 0ull; d < image.Depth; d++)
						for (auto h = 0ull; h < image.Height; h++)
							for (auto w = 0ull; w < image.Width; w++)
								dstImage(c, d, h, w) = channelMean;
				}

				return dstImage;
			}

			auto srcImage = ImageToCImg(image);

			if (deltaW != 0)
			{
				if (deltaW < 0)
					cimg_forYZC(srcImage, y, z, c)
					{
						std::memmove(srcImage.data(0, y, z, c), srcImage.data(-deltaW, y, z, c), UInt(srcImage._width + deltaW) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(srcImage.data(srcImage._width + deltaW, y, z, c), 0, -deltaW * sizeof(T));
						else
							std::memset(srcImage.data(srcImage._width + deltaW, y, z, c), (int)mean[c], -deltaW * sizeof(T));
					}
				else
					cimg_forYZC(srcImage, y, z, c)
					{
						std::memmove(srcImage.data(deltaW, y, z, c), srcImage.data(0, y, z, c), UInt(srcImage._width - deltaW) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(srcImage.data(0, y, z, c), 0, deltaW * sizeof(T));
						else
							std::memset(srcImage.data(0, y, z, c), (int)mean[c], deltaW * sizeof(T));
					}
			}

			if (deltaH != 0)
			{
				if (deltaH < 0)
					cimg_forZC(srcImage, z, c)
					{
						std::memmove(srcImage.data(0, 0, z, c), srcImage.data(0, -deltaH, z, c), UInt(srcImage._width) * UInt(srcImage._height + deltaH) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(srcImage.data(0, srcImage._height + deltaH, z, c), 0, -deltaH * UInt(srcImage._width) * sizeof(T));
						else
							std::memset(srcImage.data(0, srcImage._height + deltaH, z, c), (int)mean[c], -deltaH * UInt(srcImage._width) * sizeof(T));
					}
				else
					cimg_forZC(srcImage, z, c)
					{
						std::memmove(srcImage.data(0, deltaH, z, c), srcImage.data(0, 0, z, c), UInt(srcImage._width) * UInt(srcImage._height - deltaH) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(srcImage.data(0, 0, z, c), 0, deltaH * UInt(srcImage._width) * sizeof(T));
						else
							std::memset(srcImage.data(0, 0, z, c), (int)mean[c], deltaH * UInt(srcImage._width) * sizeof(T));
					}
			}

			auto dstImage = CImgToImage(srcImage);

			return dstImage;
		}

		static Image VerticalMirror(const Image& image)
		{
			Image dstImage(image.Channels, image.Depth, image.Height, image.Width);

			for (auto c = 0ull; c < dstImage.Channels; c++)
				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = image(c, d, image.Height - 1 - h, w);

			return dstImage;
		}
		
		static Image ZeroPad(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean)
		{
			Image dstImage(image.Channels, image.Depth + (depth * 2), image.Height + (height * 2), image.Width + (width * 2));

			T channelMean = T(0);
			for (auto c = 0ull; c < dstImage.Channels; c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = T(mean[c]);

				for (auto d = 0ull; d < dstImage.Depth; d++)
					for (auto h = 0ull; h < dstImage.Height; h++)
						for (auto w = 0ull; w < dstImage.Width; w++)
							dstImage(c, d, h, w) = channelMean;
			}

			for (auto c = 0ull; c < image.Channels; c++)
				for (auto d = 0ull; d < image.Depth; d++)
					for (auto h = 0ull; h < image.Height; h++)
						for (auto w = 0ull; w < image.Width; w++)
							dstImage(c, d + depth, h + height, w + width) = image(c, d, h, w);

			return dstImage;
		}
	};
}