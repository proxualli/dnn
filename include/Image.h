#pragma once
#include "Utils.h"

#include "jpeglib.h"
#include "jerror.h"

#undef cimg_display
#define cimg_display 0
#include "CImg.h"

namespace dnn
{
	namespace image
	{
	constexpr auto MaximumLevels = 10;		// number of levels in AutoAugment
	constexpr auto FloatLevel(const int level, const Float minValue = Float(0.1), const Float maxValue = Float(1.9)) NOEXCEPT { return (Float(level) * (maxValue - minValue) / MaximumLevels) + minValue; }
	constexpr auto IntLevel(const int level, const int minValue = 0, const int maxValue = MaximumLevels) NOEXCEPT { return (level * (maxValue - minValue) / MaximumLevels) + minValue; }

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
	private:
		cimg_library::CImg<T> Data;

	public:
		Image() NOEXCEPT :
			Data(cimg_library::CImg<T>())
		{
		}

		Image(const cimg_library::CImg<T>& image) NOEXCEPT :
			Data(image)
		{
		}
		
		Image(const unsigned c, const unsigned d, const unsigned h, const unsigned w) NOEXCEPT :
			Data(cimg_library::CImg<T>(w, h, d, c))
		{
		}

		~Image() = default;

		T* data() NOEXCEPT
		{
			return Data.data();
		}

		const T* data() const NOEXCEPT
		{
			return Data.data();
		}

		auto C() const NOEXCEPT
		{
			return Data._spectrum;
		}
		
		auto D() const NOEXCEPT
		{
			return Data._depth;
		}
		
		auto H() const NOEXCEPT
		{
			return Data._height;
		}

		auto W() const NOEXCEPT
		{
			return Data._width;
		}

		auto Area() const NOEXCEPT
		{
			return Data._height * Data._width;
		}

		auto ChannelSize() const NOEXCEPT
		{
			return Data._depth * Data._height * Data._width;
		}

		auto Size() const NOEXCEPT
		{
			return Data.size();
		}

		T& operator()(const unsigned c, const unsigned d, const unsigned h, const unsigned w) NOEXCEPT
		{
			return Data[w + (h * Data._width) + (d * Data._height * Data._width) + (c * Data._depth * Data._height * Data._width)];
		}

		const T& operator()(const unsigned c, const unsigned d, const unsigned h, const unsigned w) const NOEXCEPT
		{
			return Data[w + (h * Data._width) + (d * Data._height * Data._width) + (c * Data._depth * Data._height * Data._width)];
		}
		
		static Float GetChannelMean(const Image& image, const unsigned c) NOEXCEPT
		{
			auto mean = Float(0);
			auto correction = Float(0);

			for (auto d = 0u; d < image.D(); d++)
				for (auto h = 0u; h < image.H(); h++)
					for (auto w = 0u; w < image.W(); w++)
						KahanSum(image(c, d, h, w), mean, correction);

			return mean /= image.ChannelSize();
		}

		static Float GetChannelVariance(const Image& image, const unsigned c) NOEXCEPT
		{
			const auto mean = GetChannelMean(image, c);

			auto variance = Float(0);
			auto correction = Float(0);

			for (auto d = 0u; d < image.D(); d++)
				for (auto h = 0u; h < image.H(); h++)
					for (auto w = 0u; w < image.W(); w++)
						KahanSum(Square<Float>(image(c, d, h, w) - mean), variance, correction);

			return variance /= image.ChannelSize();
		}

		static Float GetChannelStdDev(const Image& image, const unsigned c) NOEXCEPT
		{
			return std::max(std::sqrt(GetChannelVariance(image, c)), Float(1) / std::sqrt(Float(image.ChannelSize())));
		}

		inline static cimg_library::CImg<Float> ImageToCImgFloat(const Image& image) NOEXCEPT
		{
			auto img = cimg_library::CImg<Float>(image.W(), image.H(), image.D(), image.C());

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZC(img, w, h, d, c) { img(w, h, d, c) = image(c, d, h, w); }
#else
			cimg_forXYC(img, w, h, c) { img(w, h, 0, c) = image(c, 0, h, w); }
#endif
			
			return img;
		}

		static Image AutoAugment(const Image& image, const UInt padD, const UInt padH, const UInt padW, const std::vector<Float>& mean, const bool mirrorPad, std::mt19937& generator) NOEXCEPT
		{
			Image img(image);

			const auto operation = UniformInt<UInt>(generator, 0, 24);

			switch (operation)
			{
			case 1:
			case 3:
			case 5:
				img = Image::Padding(img, padD, padH, padW, mean, mirrorPad);
				break;
			}

			switch (operation)
			{
			case 0:
			{
				if (Bernoulli<bool>(generator, Float(0.1)))
					Image::Invert(img);

				if (Bernoulli<bool>(generator, Float(0.2)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Contrast(img, FloatLevel(6));
					else
						Image::Contrast(img, FloatLevel(4));
				}
			}
			break;

			case 1:
			{
				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Rotate(img, FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
					else
						Image::Rotate(img, -FloatLevel(2, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.3)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, 0, IntLevel(7), mean);
					else
						Image::Translate(img, 0, -IntLevel(7), mean);
				}
			}
			break;

			case 2:
			{
				if (Bernoulli<bool>(generator, Float(0.8)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Sharpness(img, FloatLevel(2));
					else
						Image::Sharpness(img, FloatLevel(8));
				}

				if (Bernoulli<bool>(generator, Float(0.9)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Sharpness(img, FloatLevel(3));
					else
						Image::Sharpness(img, FloatLevel(7));
				}
			}
			break;

			case 3:
			{
				if (Bernoulli<bool>(generator, Float(0.5)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Rotate(img, FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
					else
						Image::Rotate(img, -FloatLevel(6, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, IntLevel(7), 0, mean);
					else
						Image::Translate(img, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 4:
			{
				if (Bernoulli<bool>(generator, Float(0.5)))
					Image::AutoContrast(img);

				if (Bernoulli<bool>(generator, Float(0.9)))
					Image::Equalize(img);
			}
			break;

			case 5:
			{
				if (Bernoulli<bool>(generator, Float(0.2)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Rotate(img, FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
					else
						Image::Rotate(img, -FloatLevel(4, 0, 20), Interpolations::Cubic, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.3)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Posterize(img, 32);
					else
						Image::Posterize(img, 64);
				}
			}
			break;

			case 6:
			{
				if (Bernoulli<bool>(generator, Float(0.4)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Color(img, FloatLevel(3));
					else
						Image::Color(img, FloatLevel(7));
				}

				if (Bernoulli<bool>(generator, Float(0.6)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Brightness(img, FloatLevel(7));
					else
						Image::Brightness(img, FloatLevel(3));
				}
			}
			break;

			case 7:
			{
				if (Bernoulli<bool>(generator, Float(0.3)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Sharpness(img, FloatLevel(9));
					else
						Image::Sharpness(img, FloatLevel(1));
				}

				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Brightness(img, FloatLevel(8));
					else
						Image::Brightness(img, FloatLevel(2));
				}
			}
			break;

			case 8:
			{
				if (Bernoulli<bool>(generator, Float(0.6)))
					Image::Equalize(img);

				if (Bernoulli<bool>(generator, Float(0.5)))
					Image::Equalize(img);
			}
			break;

			case 9:
			{
				if (Bernoulli<bool>(generator, Float(0.6)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Contrast(img, FloatLevel(7));
					else
						Image::Contrast(img, FloatLevel(3));
				}

				if (Bernoulli<bool>(generator, Float(0.6)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Sharpness(img, FloatLevel(6));
					else
						Image::Sharpness(img, FloatLevel(4));
				}
			}
			break;

			case 10:
			{
				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Color(img, FloatLevel(7));
					else
						Image::Color(img, FloatLevel(3));
				}

				if (Bernoulli<bool>(generator, Float(0.5)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, 0, IntLevel(8), mean);
					else
						Image::Translate(img, 0, -IntLevel(8), mean);
				}
			}
			break;

			case 11:
			{
				if (Bernoulli<bool>(generator, Float(0.3)))
					Image::Equalize(img);

				if (Bernoulli<bool>(generator, Float(0.4)))
					Image::AutoContrast(img);
			}
			break;

			case 12:
			{
				if (Bernoulli<bool>(generator, Float(0.4)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, IntLevel(3), 0, mean);
					else
						Image::Translate(img, -IntLevel(3), 0, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.2)))
					Image::Sharpness(img, FloatLevel(6));
			}
			break;

			case 13:
			{
				if (Bernoulli<bool>(generator, Float(0.9)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Brightness(img, FloatLevel(6));
					else
						Image::Brightness(img, FloatLevel(4));
				}

				if (Bernoulli<bool>(generator, Float(0.2)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Color(img, FloatLevel(8));
					else
						Image::Color(img, FloatLevel(2));
				}
			}
			break;

			case 14:
			{
				if (Bernoulli<bool>(generator, Float(0.5)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Solarize(img, IntLevel(2, 0, 256));
					else
						Image::Solarize(img, IntLevel(8, 0, 256));
				}
			}
			break;

			case 15:
			{
				if (Bernoulli<bool>(generator, Float(0.2)))
					Image::Equalize(img);

				if (Bernoulli<bool>(generator, Float(0.6)))
					Image::AutoContrast(img);
			}
			break;

			case 16:
			{
				if (Bernoulli<bool>(generator, Float(0.2)))
					Image::Equalize(img);
				if (Bernoulli<bool>(generator, Float(0.6)))
					Image::Equalize(img);
			}
			break;

			case 17:
			{
				if (Bernoulli<bool>(generator, Float(0.9)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Color(img, FloatLevel(8));
					else
						Image::Color(img, FloatLevel(2));
				}

				if (Bernoulli<bool>(generator, Float(0.6)))
					Image::Equalize(img);
			}
			break;

			case 18:
			{
				if (Bernoulli<bool>(generator, Float(0.8)))
					Image::AutoContrast(img);

				if (Bernoulli<bool>(generator, Float(0.2)))
					Image::Solarize(img, IntLevel(8, 0, 256));
			}
			break;

			case 19:
			{
				if (Bernoulli<bool>(generator, Float(0.1)))
					Image::Brightness(img, FloatLevel(3));

				if (Bernoulli<bool>(generator, Float(0.7)))
					Image::Color(img, FloatLevel(4));
			}
			break;

			case 20:
			{
				if (Bernoulli<bool>(generator, Float(0.4)))
					Image::Solarize(img, IntLevel(5, 0, 256));

				if (Bernoulli<bool>(generator, Float(0.9)))
					Image::AutoContrast(img);
			}
			break;

			case 21:
			{
				if (Bernoulli<bool>(generator, Float(0.9)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, IntLevel(7), 0, mean);
					else
						Image::Translate(img, -IntLevel(7), 0, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, IntLevel(7), 0, mean);
					else
						Image::Translate(img, -IntLevel(7), 0, mean);
				}
			}
			break;

			case 22:
			{
				if (Bernoulli<bool>(generator, Float(0.9)))
					Image::AutoContrast(img);

				if (Bernoulli<bool>(generator, Float(0.8)))
					Image::Solarize(img, IntLevel(3, 0, 256));
			}
			break;

			case 23:
			{
				if (Bernoulli<bool>(generator, Float(0.8)))
					Image::Equalize(img);

				if (Bernoulli<bool>(generator, Float(0.1)))
					Image::Invert(img);
			}
			break;

			case 24:
			{
				if (Bernoulli<bool>(generator, Float(0.7)))
				{
					if (Bernoulli<bool>(generator, Float(0.5)))
						Image::Translate(img, IntLevel(8), 0, mean);
					else
						Image::Translate(img, -IntLevel(8), 0, mean);
				}

				if (Bernoulli<bool>(generator, Float(0.9)))
					Image::AutoContrast(img);
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
				img = Image::Padding(img, padD, padH, padW, mean, mirrorPad);
				break;
			}

			return img;
		}

		static void AutoContrast(Image& image) NOEXCEPT
		{
			constexpr T maximum = std::is_floating_point_v<T> ? static_cast<T>(1) : static_cast<T>(255);
			image.Data.normalize(0, maximum);
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static void Brightness(Image& image, const Float magnitude) NOEXCEPT
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			const auto delta = (magnitude - Float(1)) / 2;

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZ(srcImage, w, h, d) { srcImage(w, h, d, 2) = cimg_library::cimg::cut(srcImage(w, h, d, 2) + delta, 0, 1); }

			srcImage.HSLtoRGB();

			cimg_forXYZC(srcImage, w, h, d, c) { image(c, d, h, w) = Saturate<Float>(srcImage(w, h, d, c)); }
#else
			cimg_forXY(srcImage, w, h) { srcImage(w, h, 0, 2) = cimg_library::cimg::cut(srcImage(w, h, 0, 2) + delta, 0, 1); }

			srcImage.HSLtoRGB();

			cimg_forXYC(srcImage, w, h, c) { image(c, 0, h, w) = Saturate<Float>(srcImage(w, h, 0, c)); }
#endif
		}

		// magnitude = 0   // black-and-white image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static void Color(Image& image, const Float magnitude) NOEXCEPT
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZ(srcImage, w, h, d) { srcImage(w, h, d, 0) = cimg_library::cimg::cut(srcImage(w, h, d, 0) * magnitude, 0, 360); }

			srcImage.HSLtoRGB();

			cimg_forXYZC(srcImage, w, h, d, c) { image(c, d, h, w) = Saturate<Float>(srcImage(w, h, d, c)); }
#else

			cimg_forXY(srcImage, w, h) { srcImage(w, h, 0, 0) = cimg_library::cimg::cut(srcImage(w, h, 0, 0) * magnitude, 0, 360); }

			srcImage.HSLtoRGB();

			cimg_forXYC(srcImage, w, h, c) { image(c, 0, h, w) = Saturate<Float>(srcImage(w, h, 0, c)); }
#endif
		}

		static void ColorCast(Image& image, const UInt angle, std::mt19937& generator) NOEXCEPT
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

			const auto shift = Float(Bernoulli<bool>(generator, Float(0.5)) ? static_cast<int>(UniformInt<UInt>(generator, 0, 2 * angle)) - static_cast<int>(angle) : 0);

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZ(srcImage, w, h, d) { srcImage(w, h, d, 0) = cimg_library::cimg::cut(srcImage(w, h, d, 0) + shift, 0, 360); }

			srcImage.HSLtoRGB();

			cimg_forXYZC(srcImage, w, h, d, c) { image(c, d, h, w) = Saturate<Float>(srcImage(w, h, d, c)); }
#else
			cimg_forXY(srcImage, w, h) { srcImage(w, h, 0, 0) = cimg_library::cimg::cut(srcImage(w, h, 0, 0) + shift, 0, 360); }

			srcImage.HSLtoRGB();

			cimg_forXYC(srcImage, w, h, c) { image(c, 0, h, w) = Saturate<Float>(srcImage(w, h, 0, c)); }
#endif
		}
		
		// magnitude = 0   // gray image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static void Contrast(Image& image, const Float magnitude) NOEXCEPT
		{
			auto srcImage = ImageToCImgFloat(image);

			srcImage.RGBtoHSL();

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZ(srcImage, w, h, d) { srcImage(w, h, d, 1) = cimg_library::cimg::cut(srcImage(w, h, d, 1) * magnitude, 0, 1); }

			srcImage.HSLtoRGB();

			cimg_forXYZC(srcImage, w, h, d, c) { image(c, d, h, w) = Saturate<Float>(srcImage(w, h, d, c)); }
#else
			cimg_forXY(srcImage, w, h) { srcImage(w, h, 0, 1) = cimg_library::cimg::cut(srcImage(w, h, 0, 1) * magnitude, 0, 1); }

			srcImage.HSLtoRGB();

			cimg_forXYC(srcImage, w, h, c) { image(c, 0, h, w) = Saturate<Float>(srcImage(w, h, 0, c)); }
#endif
		}

		// magnitude = 0   // gray image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static Image Crop(const Image& image, const Positions position, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean) NOEXCEPT
		{
			Image img(image.C(), static_cast<unsigned>(depth), static_cast<unsigned>(height), static_cast<unsigned>(width));
			
			//cimg_forXYZC(img, w, h, d, c) { img(c, d, h, w) = std::is_floating_point_v<T> ? static_cast<T>(0) : static_cast<T>(mean[c]); }

			for (auto c = 0u; c < img.C(); c++)
			{
				const T channelMean = std::is_floating_point_v<T> ? static_cast<T>(0) : static_cast<T>(mean[c]);
				for (auto d = 0u; d < img.D(); d++)
					for (auto h = 0u; h < img.H(); h++)
						for (auto w = 0u; w < img.W(); w++)
							img(c, d, h, w) = channelMean;
			}

			const auto minDepth = std::min(img.D(), image.D());
			const auto minHeight = std::min(img.H(), image.H());
			const auto minWidth = std::min(img.W(), image.W());

			const auto srcDdelta = img.D() < image.D() ? (image.D() - img.D()) / 2 : 0u;
			const auto dstDdelta = img.D() > image.D() ? (img.D() - image.D()) / 2 : 0u;

			switch (position)
			{
			case Positions::Center:
			{
				const auto srcHdelta = img.H() < image.H() ? (image.H() - img.H()) / 2 : 0u;
				const auto dstHdelta = img.H() > image.H() ? (img.H() - image.H()) / 2 : 0u;
				const auto srcWdelta = img.W() < image.W() ? (image.W() - img.W()) / 2 : 0u;
				const auto dstWdelta = img.W() > image.W() ? (img.W() - image.W()) / 2 : 0u;

				for (auto c = 0u; c < img.C(); c++)
					for (auto d = 0u; d < minDepth; d++)
						for (auto h = 0u; h < minHeight; h++)
							for (auto w = 0u; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::TopLeft:
			{
				for (auto c = 0u; c < img.C(); c++)
					for (auto d = 0u; d < minDepth; d++)
						for (auto h = 0u; h < minHeight; h++)
							for (auto w = 0u; w < minWidth; w++)
								img(c, d + dstDdelta, h, w) = image(c, d + srcDdelta, h, w);
			}
			break;

			case Positions::TopRight:
			{
				const auto srcWdelta = img.W() < image.W() ? (image.W() - img.W()) : 0u;
				const auto dstWdelta = img.W() > image.W() ? (img.W() - image.W()) : 0u;

				for (auto c = 0u; c < img.C(); c++)
					for (auto d = 0u; d < minDepth; d++)
						for (auto h = 0u; h < minHeight; h++)
							for (auto w = 0u; w < minWidth; w++)
								img(c, d + dstDdelta, h, w + dstWdelta) = image(c, d + srcDdelta, h, w + srcWdelta);
			}
			break;

			case Positions::BottomRight:
			{
				const auto srcHdelta = img.H() < image.H() ? (image.H() - img.H()) : 0u;
				const auto dstHdelta = img.H() > image.H() ? (img.H() - image.H()) : 0u;
				const auto srcWdelta = img.W() < image.W() ? (image.W() - img.W()) : 0u;
				const auto dstWdelta = img.W() > image.W() ? (img.W() - image.W()) : 0u;

				for (auto c = 0u; c < img.C(); c++)
					for (auto d = 0u; d < minDepth; d++)
						for (auto h = 0u; h < minHeight; h++)
							for (auto w = 0u; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			}
			break;

			case Positions::BottomLeft:
			{
				const auto srcHdelta = img.H() < image.H() ? (image.H() - img.H()) : 0u;
				const auto dstHdelta = img.H() > image.H() ? (img.H() - image.H()) : 0u;

				for (auto c = 0u; c < img.C(); c++)
					for (auto d = 0u; d < minDepth; d++)
						for (auto h = 0u; h < minHeight; h++)
							for (auto w = 0u; w < minWidth; w++)
								img(c, d + dstDdelta, h + dstHdelta, w) = image(c, d + srcDdelta, h + srcHdelta, w);
			}
			break;
			}

			return img;
		}

		static Image Distorted(Image& image, const Float scale, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean, std::mt19937& generator) NOEXCEPT
		{
			const auto zoom = scale / Float(100) * UniformReal<Float>(generator, Float(-1), Float(1));
			const auto height = static_cast<unsigned>(static_cast<int>(image.H()) + static_cast<int>(std::round(static_cast<int>(image.H()) * zoom)));
			const auto width = static_cast<unsigned>(static_cast<int>(image.W()) + static_cast<int>(std::round(static_cast<int>(image.W()) * zoom)));
			
			Image::Resize(image, image.D(), height, width, interpolation);

			return Image::Rotate(image, angle * UniformReal<Float>(generator, Float(-1), Float(1)), interpolation, mean);
		}

		static void Dropout(Image& image, const Float dropout, const std::vector<Float>& mean, std::mt19937& generator) NOEXCEPT
		{
			for (auto d = 0u; d < image.D(); d++)
				for (auto h = 0u; h < image.H(); h++)
					for (auto w = 0u; w < image.W(); w++)
						if (Bernoulli<bool>(generator, dropout))
							for (auto c = 0u; c < image.C(); c++)
							{
								if constexpr (std::is_floating_point_v<T>)
									image(c, d, h, w) = static_cast<T>(0);
								else
									image(c, d, h, w) = static_cast<T>(mean[c]);
							}
		}
		
		static void Equalize(Image& image) NOEXCEPT
		{
			image.Data.equalize(256);
		}
		
		static void HorizontalMirror(Image& image) NOEXCEPT
		{
			T left;
#ifdef DNN_IMAGEDEPTH
			cimg_forXYZC(image.Data, w, h, d, c)
			{
				left = image(c, d, h, w);
				image(c, d, h, w) = image(c, d, h, image.W() - 1 - w);
				image(c, d, h, image.W() - 1 - w) = left;
			}
#else
			cimg_forXYC(image.Data, w, h, c)
			{
				left = image(c, 0, h, w);
				image(c, 0, h, w) = image(c, 0, h, image.W() - 1 - w);
				image(c, 0, h, image.W() - 1 - w) = left;
			}
#endif
		}

		static void Invert(Image& image) NOEXCEPT
		{
			constexpr T maximum = std::is_floating_point_v<T> ? static_cast<T>(1) : static_cast<T>(255);
			
#ifdef DNN_IMAGEDEPTH
			cimg_forXYZC(image.Data, w, h, d, c) { image(c, d, h, w) = maximum - image(c, d, h, w); }
#else

			cimg_forXYC(image.Data, w, h, c) { image(c, 0, h, w) = maximum - image(c, 0, h, w); }
#endif
		}

		static Image MirrorPad(const Image& image, const unsigned depth, const unsigned height, const unsigned width) NOEXCEPT
		{
			Image img(image.C(), image.D() + (depth * 2), image.H() + (height * 2), image.W() + (width * 2));

			for (auto c = 0u; c < image.C(); c++)
			{
				for (auto d = 0u; d < depth; d++)
				{
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d, h, w) = image(c, d, height - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d, h, w + width) = image(c, d, height - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d, h, w + width + image.W()) = image(c, d, height - (h + 1), image.W() - (w + 1));
					}
					for (auto h = 0u; h < image.H(); h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d, h + height, w) = image(c, d, h, width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth, h + height, w + width) = image(c, d, h, w);
						for (auto w = 0u; w < width; w++)
							img(c, d, h + height, w + width + image.W()) = image(c, d, h, image.W() - (w + 1));
					}
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d, h + height + image.H(), w) = image(c, d, image.H() - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d, h + height + image.H(), w + width) = image(c, d, image.H() - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d, h + height + image.H(), w + width + image.W()) = image(c, d, image.H() - (h + 1), image.W() - (w + 1));
					}
				}
				for (auto d = 0u; d < image.D(); d++)
				{
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h, w) = image(c, d + depth, height - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth, h, w + width) = image(c, d + depth, height - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h, w + width + image.W()) = image(c, d + depth, height - (h + 1), image.W() - (w + 1));
					}
					for (auto h = 0u; h < image.H(); h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h + height, w) = image(c, d + depth, h, width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth, h + height, w + width) = image(c, d + depth, h, w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h + height, w + width + image.W()) = image(c, d + depth, h, image.W() - (w + 1));
					}
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h + height + image.H(), w) = image(c, d + depth, image.H() - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth, h + height + image.H(), w + width) = image(c, d + depth, image.H() - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth, h + height + image.H(), w + width + image.W()) = image(c, d + depth, image.H() - (h + 1), image.W() - (w + 1));
					}
				}
				for (auto d = 0u; d < depth; d++)
				{
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h, w) = image(c, d + depth + image.D(), height - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth + image.D(), h, w + width) = image(c, d + depth + image.D(), height - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h, w + width + image.W()) = image(c, d + depth + image.D(), height - (h + 1), image.W() - (w + 1));
					}
					for (auto h = 0u; h < image.H(); h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h + height, w) = image(c, d + depth + image.D(), h, width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth + image.D(), h + height, w + width) = image(c, d + depth + image.D(), h, w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h + height, w + width + image.W()) = image(c, d + depth + image.D(), h, image.W() - (w + 1));
					}
					for (auto h = 0u; h < height; h++)
					{
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h + height + image.H(), w) = image(c, d + depth + image.D(), image.H() - (h + 1), width - (w + 1));
						for (auto w = 0u; w < image.W(); w++)
							img(c, d + depth + image.D(), h + height + image.H(), w + width) = image(c, d + depth + image.D(), image.H() - (h + 1), w);
						for (auto w = 0u; w < width; w++)
							img(c, d + depth + image.D(), h + height + image.H(), w + width + image.W()) = image(c, d + depth + image.D(), image.H() - (h + 1), image.W() - (w + 1));
					}
				}
			}

			return img;
		}

		static Image Padding(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean, const bool mirrorPad = false) NOEXCEPT
		{
			return mirrorPad ? Image::MirrorPad(image, static_cast<unsigned>(depth), static_cast<unsigned>(height), static_cast<unsigned>(width)) : Image::ZeroPad(image, static_cast<unsigned>(depth), static_cast<unsigned>(height), static_cast<unsigned>(width), mean);
		}

		static void Posterize(Image& image, const unsigned levels = 16) NOEXCEPT
		{
			auto palette = std::vector<Byte>(256);
			const auto q = 256u / levels;

			for (auto c = 0u; c < 255u; c++)
				palette[c] = Saturate<unsigned>((((c / q) * q) * levels) / (levels - 1));

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZC(image.Data, w, h, d, c) { image(c, d, h, w) = palette[image(c, d, h, w)]; }
#else
			cimg_forXYC(image.Data, w, h, c) { image(c, 0, h, w) = palette[image(c, 0, h, w)]; }
#endif
		}
		
		static Image RandomCrop(const Image& image, const UInt depth, const UInt height, const UInt width, const std::vector<Float>& mean, std::mt19937& generator) NOEXCEPT
		{
			Image img(image.C(), static_cast<unsigned>(depth), static_cast<unsigned>(height), static_cast<unsigned>(width));

			auto channelMean = static_cast<T>(0);
			for (auto c = 0u; c < img.C(); c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = static_cast<T>(mean[c]);
				
				for (auto d = 0u; d < img.D(); d++)
					for (auto h = 0u; h < img.H(); h++)
						for (auto w = 0u; w < img.W(); w++)
							img(c, d, h, w) = channelMean;
			}
			
			const auto minD = std::min(img.D(), image.D());
			const auto minH = std::min(img.H(), image.H());
			const auto minW = std::min(img.W(), image.W());
			
			const auto srcDdelta = img.D() < image.D() ? UniformInt<unsigned>(generator, 0, image.D() - img.D()) : 0u;
			const auto srcHdelta = img.H() < image.H() ? UniformInt<unsigned>(generator, 0, image.H() - img.H()) : 0u;
			const auto srcWdelta = img.W() < image.W() ? UniformInt<unsigned>(generator, 0, image.W() - img.W()) : 0u;
			
			const auto dstDdelta = img.D() > image.D() ? UniformInt<unsigned>(generator, 0, img.D() - image.D()) : 0u;
			const auto dstHdelta = img.H() > image.H() ? UniformInt<unsigned>(generator, 0, img.H() - image.H()) : 0u;
			const auto dstWdelta = img.W() > image.W() ? UniformInt<unsigned>(generator, 0, img.W() - image.W()) : 0u;

			for (auto c = 0u; c < img.C(); c++)
				for (auto d = 0u; d < minD; d++)
					for (auto h = 0u; h < minH; h++)
						for (auto w = 0u; w < minW; w++)
							img(c, d + dstDdelta, h + dstHdelta, w + dstWdelta) = image(c, d + srcDdelta, h + srcHdelta, w + srcWdelta);
			
			return img;
		}

		static void RandomCutout(Image& image, const std::vector<Float>& mean, std::mt19937& generator) NOEXCEPT
		{
			const auto centerH = UniformInt<unsigned>(generator, 0, image.H());
			const auto centerW = UniformInt<unsigned>(generator, 0, image.W());
			const auto rangeH = UniformInt<unsigned>(generator, image.H() / 8, image.H() / 4);
			const auto rangeW = UniformInt<unsigned>(generator, image.W() / 8, image.W() / 4);
			const auto startH = static_cast<long>(centerH) - static_cast<long>(rangeH) > 0l ? centerH - rangeH : 0u;
			const auto startW = static_cast<long>(centerW) - static_cast<long>(rangeW) > 0l ? centerW - rangeW : 0u;
			const auto enheight = centerH + rangeH < image.H() ? centerH + rangeH : image.H();
			const auto enwidth = centerW + rangeW < image.W() ? centerW + rangeW : image.W();

			auto channelMean =  static_cast<T>(0);
			for (auto c = 0u; c < image.C(); c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean =  static_cast<T>(mean[c]);
				for (auto d = 0u; d < image.D(); d++)
					for (auto h = startH; h < enheight; h++)
						for (auto w = startW; w < enwidth; w++)
							image(c, d, h, w) = channelMean;
			}
		}

		static void RandomCutMix(Image& image, const Image& imageMix, double* lambda, std::mt19937& generator) NOEXCEPT
		{
			const auto cutRate = std::sqrt(1.0 - *lambda);
			const auto cutH = static_cast<int>(static_cast<double>(image.H()) * cutRate);
			const auto cutW = static_cast<int>(static_cast<double>(image.W()) * cutRate);
			const auto cy = UniformInt<int>(generator, 0, static_cast<int>(image.H()));
			const auto cx = UniformInt<int>(generator, 0, static_cast<int>(image.W()));
			const auto bby1 = Clamp<int>(cy - cutH / 2, 0, static_cast<int>(image.H()));
			const auto bby2 = Clamp<int>(cy + cutH / 2, 0, static_cast<int>(image.H()));
			const auto bbx1 = Clamp<int>(cx - cutW / 2, 0, static_cast<int>(image.W()));
			const auto bbx2 = Clamp<int>(cx + cutW / 2, 0, static_cast<int>(image.W()));

			for (auto c = 0u; c < image.C(); c++)
				for (auto d = 0u; d < image.D(); d++)
					for (auto h = bby1; h < bby2; h++)
						for (auto w = bbx1; w < bbx2; w++)
							image(c, d, h, w) = imageMix(c, d, h, w);

			*lambda = 1.0 - (static_cast<double>((bbx2 - bbx1) * (bby2 - bby1)) / static_cast<double>(image.H() * image.W()));
		}

		static void Resize(Image& image, const UInt depth, const UInt height, const UInt width, const Interpolations interpolation) NOEXCEPT
		{
			switch (interpolation)
			{
			case Interpolations::Cubic:
				image.Data.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C()), 5, 0);
				break;
			case Interpolations::Linear:
				image.Data.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C()), 3, 0);
				break;
			case Interpolations::Nearest:
				image.Data.resize(static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth), static_cast<int>(image.C()), 1, 0);
				break;
			}
		}

		static Image Rotate(const Image& image, const Float angle, const Interpolations interpolation, const std::vector<Float>& mean) NOEXCEPT
		{
			auto img = ZeroPad(image, image.D() / 2, image.H() / 2, image.W() / 2, mean);

			switch (interpolation)
			{
			case Interpolations::Cubic:
				img.Data.rotate(angle, 2, 0);
				break;
			case Interpolations::Linear:
				img.Data.rotate(angle, 1, 0);
				break;
			case Interpolations::Nearest:
				img.Data.rotate(angle, 0, 0);
				break;
			}

			return Crop(img, Positions::Center, image.D(), image.H(), image.W(), mean);
		}
			
		// magnitude = 0   // blurred image
		// magnitude = 1   // original
		// range 0.1 --> 1.9
		static void Sharpness(Image& image, const Float magnitude) NOEXCEPT
		{
			image.Data.sharpen(magnitude, false);
		}

		static void Solarize(Image& image, const T treshold = 128) NOEXCEPT
		{
			constexpr T maximum = std::is_floating_point_v<T> ? static_cast<T>(1) : static_cast<T>(255);

#ifdef DNN_IMAGEDEPTH
			cimg_forXYZC(image.Data, w, h, d, c) { image(c, d, h, w) = image(c, d, h, w) < treshold ? image(c, d, h, w) : maximum - image(c, d, h, w); }
#else
			cimg_forXYC(image.Data, w, h, c) { image(c, 0, h, w) = image(c, 0, h, w) < treshold ? image(c, 0, h, w) : maximum - image(c, 0, h, w); }
#endif
		}
		
		static void Translate(Image& image, const int height, const int width, const std::vector<Float>& mean) NOEXCEPT
		{
			if (height == 0 && width == 0)
				return;

			if (width <= -static_cast<int>(image.W()) || width >= static_cast<int>(image.W()) || height <= -static_cast<int>(image.H()) || height >= static_cast<int>(image.H()))
			{
				T channelMean =  static_cast<T>(0);
				for (auto c = 0u; c < image.C(); c++)
				{
					if constexpr (!std::is_floating_point_v<T>)
						channelMean =  static_cast<T>(mean[c]);

					for (auto d = 0u; d < image.D(); d++)
						for (auto h = 0u; h < image.H(); h++)
							for (auto w = 0u; w < image.W(); w++)
								image(c, d, h, w) = channelMean;
				}
				return;
			}

			if (width != 0)
			{
				if (width < 0)
					cimg_forYZC(image.Data, y, z, c)
					{
						std::memmove(image.Data.data(0, y, z, c), image.Data.data(-width, y, z, c), static_cast<UInt>(image.W() + width) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(image.Data.data(image.W() + width, y, z, c), 0, -width * sizeof(T));
						else
							std::memset(image.Data.data(image.W() + width, y, z, c), (int)mean[c], -width * sizeof(T));
					}
				else
					cimg_forYZC(image.Data, y, z, c)
					{
						std::memmove(image.Data.data(width, y, z, c), image.Data.data(0, y, z, c), static_cast<UInt>(image.W() - width) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(image.Data.data(0, y, z, c), 0, width * sizeof(T));
						else
							std::memset(image.Data.data(0, y, z, c), (int)mean[c], width * sizeof(T));
					}
			}

			if (height != 0)
			{
				if (height < 0)
					cimg_forZC(image.Data, z, c)
					{
						std::memmove(image.Data.data(0, 0, z, c), image.Data.data(0, -height, z, c), static_cast<UInt>(image.W()) * static_cast<UInt>(image.H() + height) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(image.Data.data(0, image.H() + height, z, c), 0, -height * static_cast<UInt>(image.W()) * sizeof(T));
						else
							std::memset(image.Data.data(0, image.H() + height, z, c), (int)mean[c], -height * static_cast<UInt>(image.W()) * sizeof(T));
					}
				else
					cimg_forZC(image.Data, z, c)
					{
						std::memmove(image.Data.data(0, height, z, c), image.Data.data(0, 0, z, c), static_cast<UInt>(image.W()) * static_cast<UInt>(image.H() - height) * sizeof(T));
						if constexpr (std::is_floating_point_v<T>)
							std::memset(image.Data.data(0, 0, z, c), 0, height * static_cast<UInt>(image.W()) * sizeof(T));
						else
							std::memset(image.Data.data(0, 0, z, c), (int)mean[c], height * static_cast<UInt>(image.W()) * sizeof(T));
					}
			}
		}

		static void VerticalMirror(Image& image) NOEXCEPT
		{
			T top; const auto d = 0u;
			for (auto c = 0u; c < image.C(); c++)
				for (auto d = 0u; d < image.D(); d++)
					for (auto w = 0u; w < image.W(); w++)
						for (auto h = 0u; h < image.H(); h++)
						{
							top = image(c, d, h, w);
							image(c, d, h, w) = image(c, d, image.H() - 1 - h, w);
							image(c, d, image.H() - 1 - h, w) = top;
						}
		}
		
		static Image ZeroPad(const Image& image, const unsigned depth, const unsigned height, const unsigned width, const std::vector<Float>& mean) NOEXCEPT
		{
			Image img(image.C(), image.D() + (depth * 2), image.H() + (height * 2), image.W() + (width * 2));
			
			T channelMean = static_cast<T>(0);

#ifdef DNN_IMAGEDEPTH
			for (auto c = 0u; c < img.C(); c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = static_cast<T>(mean[c]);

				for (auto d = 0u; d < img.D(); d++)
					for (auto h = 0u; h < img.H(); h++)
						for (auto w = 0u; w < img.W(); w++)
							img(c, d, h, w) = channelMean;
			}

			cimg_forXYZC(image.Data, w, h, d, c) { img(c, d + depth, h + height, w + width) = image(c, d, h, w); }
#else
			for (auto c = 0u; c < img.C(); c++)
			{
				if constexpr (!std::is_floating_point_v<T>)
					channelMean = static_cast<T>(mean[c]);
				for (auto h = 0u; h < img.H(); h++)
					for (auto w = 0u; w < img.W(); w++)
						img(c, 0, h, w) = channelMean;
			}

			cimg_forXYC(image.Data, w, h, c) { img(c, depth, h + height, w + width) = image(c, 0, h, w); }
#endif
			
			return img;
		}
	};
	}
}