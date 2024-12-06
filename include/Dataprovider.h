#pragma once
#include "Image.h"


namespace dnn
{
	//using namespace image;

	typedef std::vector<Image<Byte>, AlignedAllocator<dnn::Image<Byte>, 64ull>> ImageByteVector;

	enum class Datasets
	{
		cifar10 = 0,
		cifar100 = 1,
		fashionmnist = 2,
		mnist = 3,
		tinyimagenet = 4
	};

	class Dataprovider final
	{
	public:
		std::filesystem::path StorageDirectory;
		std::filesystem::path DatasetsDirectory;
		Datasets Dataset;
		UInt C;
		UInt D;
		UInt H;
		UInt W;
		std::vector<Float> Mean;
		std::vector<Float> StdDev;
		UInt ShuffleCount;
		UInt TrainSamplesCount;
		UInt TestSamplesCount;
		UInt Hierarchies;
		std::vector<UInt> ClassCount;
		std::vector<std::string> ClassNames;
		ImageByteVector TrainSamples;
		ImageByteVector TestSamples;
		std::vector<std::vector<UInt>> TrainLabels;
		std::vector<std::vector<UInt>> TestLabels;

		Dataprovider(const std::string& directory) :
			StorageDirectory(std::filesystem::path(directory)),
			DatasetsDirectory(StorageDirectory / "datasets"),
			Dataset(Datasets::cifar10),
			C(3),
			D(1),
			H(32),
			W(32),
			Mean(std::vector<Float>({ Float(125.79831808), Float(123.43251712), Float(114.31199744) })),
			StdDev(std::vector<Float>({ Float(63.24027648), Float(62.3321728), Float(66.96644608) })),
			ShuffleCount(64ull),
			TrainSamplesCount(50000),
			TestSamplesCount(10000),
			Hierarchies(1),
			ClassCount(std::vector<UInt>({ 10 }))
		{
			std::filesystem::create_directories(DatasetsDirectory);

			std::locale::global(std::locale(""));
		}

		virtual ~Dataprovider() = default;

		bool DatasetAvailable(const Datasets dataset) const
		{
			std::filesystem::path path;

			auto available = false;

			switch (dataset)
			{
			case Datasets::cifar10:
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				if (std::filesystem::exists(path / "data_batch_1.bin") && std::filesystem::exists(path / "data_batch_2.bin") && std::filesystem::exists(path / "data_batch_3.bin") && std::filesystem::exists(path / "data_batch_4.bin") && std::filesystem::exists(path / "data_batch_5.bin") && std::filesystem::exists(path / "test_batch.bin") && std::filesystem::exists(path / "batches.meta.txt"))
					available = true;
				break;

			case Datasets::cifar100:
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				if (std::filesystem::exists(path / "train.bin") && std::filesystem::exists(path / "test.bin") && std::filesystem::exists(path / "fine_label_names.txt") && std::filesystem::exists(path / "coarse_label_names.txt"))
					available = true;
				break;

			case Datasets::fashionmnist:
			case Datasets::mnist:
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				if (std::filesystem::exists(path / "t10k-labels-idx1-ubyte") && std::filesystem::exists(path / "train-labels-idx1-ubyte") && std::filesystem::exists(path / "t10k-images-idx3-ubyte") && std::filesystem::exists(path / "train-images-idx3-ubyte"))
					available = true;
				break;

			case Datasets::tinyimagenet:
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				if (std::filesystem::exists(path / "wnids.txt") && std::filesystem::exists(path / "words.txt"))
					available = true;
				break;

			default:
				available = false;
			}

			return available;
		}

		bool GetDataset(const Datasets dataset)
		{
			std::filesystem::path path;

			switch (dataset)
			{
			case Datasets::fashionmnist:
			{
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				std::filesystem::create_directories(path);

				auto batchesmeta = std::ofstream((path / "batches.meta.txt").string(), std::ios::trunc);

				batchesmeta <<
					"tshirt_top" << std::endl <<
					"trouser" << std::endl <<
					"pullover" << std::endl <<
					"dress" << std::endl <<
					"coat" << std::endl <<
					"sandal" << std::endl <<
					"shirt" << std::endl <<
					"sneaker" << std::endl <<
					"bag" << std::endl <<
					"ankle_boot" << std::endl;

				batchesmeta.close();
			}
			break;

			case Datasets::mnist:
			{
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				std::filesystem::create_directories(path);
			}
			break;

			case Datasets::cifar10:
			case Datasets::cifar100:
			case Datasets::tinyimagenet:
			break;
			}

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
			switch (dataset)
			{
			case Datasets::fashionmnist:
			case Datasets::mnist:
			{
				auto unzipScript = std::ofstream((path / "UnZip-File.ps1").string(), std::ios::trunc);

				unzipScript <<
					"Function UnZip-File{" << std::endl <<
					"Param(" << std::endl <<
					"$infile," << std::endl <<
					"$outfile = ($infile -replace '\\.gz$','')" << std::endl <<
					")" << std::endl <<
					"$input = New-Object System.IO.FileStream $inFile, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), ([IO.FileShare]::Read)" << std::endl <<
					"$output = New-Object System.IO.FileStream $outFile, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None)" << std::endl <<
					"$gzipStream = New-Object System.IO.Compression.GzipStream $input, ([IO.Compression.CompressionMode]::Decompress)" << std::endl <<
					"$buffer = New-Object byte[](1024)" << std::endl <<
					"while ($true) {" << std::endl <<
					"$read = $gzipstream.Read($buffer, 0, 1024)" << std::endl <<
					"if ($read -le 0) { break }" << std::endl <<
					"$output.Write($buffer, 0, $read)" << std::endl <<
					"}" << std::endl <<
					"$gzipStream.Close()" << std::endl <<
					"$output.Close()" << std::endl <<
					"$input.Close()" << std::endl <<
					"}" << std::endl <<
					"UnZip-File \"./train-images-idx3-ubyte.gz\" \"./train-images-idx3-ubyte\"" << std::endl <<
					"UnZip-File \"./t10k-images-idx3-ubyte.gz\" \"./t10k-images-idx3-ubyte\"" << std::endl <<
					"UnZip-File \"./train-labels-idx1-ubyte.gz\" \"./train-labels-idx1-ubyte\"" << std::endl <<
					"UnZip-File \"./t10k-labels-idx1-ubyte.gz\" \"./t10k-labels-idx1-ubyte\"" << std::endl;

				unzipScript.close();
			}
			break;

			case Datasets::tinyimagenet:
			{
				auto unzipScript = std::ofstream((DatasetsDirectory / "UnZip-File.ps1").string(), std::ios::trunc);

				unzipScript <<
					"function UnZip-File($file, $destination)" << std::endl <<
					"{" << std::endl <<
					"$shell= new-object -com shell.application" << std::endl <<
					"$zip = $shell.NameSpace($file)" << std::endl <<
					"foreach($item in $zip.items())" << std::endl <<
					"{" << std::endl <<
					"$shell.Namespace($destination).copyhere($item)" << std::endl <<
					"}" << std::endl <<
					"}" << std::endl <<
					"UnZip-File -File \"" << DatasetsDirectory.string() << "\\tiny-imagenet-200.zip\" -Destination \"" << DatasetsDirectory.string() << "\"" << std::endl;

				unzipScript.close();
			}
			break;

			case Datasets::cifar10:
			case Datasets::cifar100:
			break;
			}

			const std::string fileName = "commands.cmd";
#else
			const std::string fileName = "commands.sh";
#endif

			auto batch = std::ofstream((DatasetsDirectory / fileName).string(), std::ios::trunc);

			switch (dataset)
			{
			case Datasets::cifar10:
			{
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				std::filesystem::create_directories(path);

				batch <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					std::string("@echo off") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("echo.") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
					std::string("curl -O http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && tar -xf cifar-10-binary.tar.gz --strip-components=1 && del /Q cifar-10-binary.tar.gz") << std::endl;
#else
					std::string("#!/bin/bash") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
					std::string("curl -O http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz") << std::endl << 
					std::string("tar -xf cifar-10-binary.tar.gz --strip-components=1") << std::endl << 
					std::string("rm ./cifar-10-binary.tar.gz") << std::endl;
#endif
			}
			break;

			case Datasets::cifar100:
			{
				path = DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset));
				std::filesystem::create_directories(path);

				batch <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					std::string("@echo off") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("echo.") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
					std::string("curl -O http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz && tar -xf cifar-100-binary.tar.gz --strip-components=1 && del /Q cifar-100-binary.tar.gz") << std::endl;
#else
					std::string("#!/bin/bash") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
					std::string("curl -O http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz") << std::endl << 
					std::string("tar -xf cifar-100-binary.tar.gz --strip-components=1") << std::endl << 
					std::string("rm ./cifar-100-binary.tar.gz") << std::endl;
#endif
			}
			break;

			case Datasets::fashionmnist:
			{
				batch <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					std::string("@echo off") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("echo.") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#else
					std::string("#!/bin/bash") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#endif
					std::string("curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz") << std::endl <<
					std::string("curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz") << std::endl <<
					std::string("curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz") << std::endl <<
					std::string("curl -O http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz") << std::endl <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					"powershell -ExecutionPolicy Bypass -command \"& {&./UnZip-File.ps1}\";" << std::endl <<
					"del *.gz" << std::endl <<
					"del UnZip-File.ps1" << std::endl;
#else
					std::string("gunzip -f ./train-images-idx3-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./t10k-images-idx3-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./train-labels-idx1-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./t10k-labels-idx1-ubyte.gz") << std::endl;
#endif
			}
			break;

			case Datasets::mnist:
			{
				batch <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					std::string("@echo off") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("echo.") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#else
					std::string("#!/bin/bash") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#endif
					std::string("curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz") << std::endl <<
					std::string("curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz") << std::endl <<
					std::string("curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz") << std::endl <<
					std::string("curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz") << std::endl <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					"powershell -ExecutionPolicy Bypass -command \"& {&./UnZip-File.ps1}\";" << std::endl <<
					"del *.gz" << std::endl <<
					"del UnZip-File.ps1" << std::endl;
#else
					std::string("gunzip -f ./train-images-idx3-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./t10k-images-idx3-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./train-labels-idx1-ubyte.gz") << std::endl <<
					std::string("gunzip -f ./t10k-labels-idx1-ubyte.gz") << std::endl;
#endif
			}
			break;

			case Datasets::tinyimagenet:
			{
				path = DatasetsDirectory;

				batch <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					std::string("@echo off") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("echo.") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#else
					std::string("#!/bin/bash") << std::endl <<
					std::string("echo loading ") + std::string(magic_enum::enum_name<Datasets>(dataset)) + std::string(" dataset...") << std::endl <<
					std::string("cd ") + path.string() << std::endl <<
#endif
					std::string("curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip") << std::endl <<
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
					"powershell -ExecutionPolicy Bypass -command \"& {&./UnZip-File.ps1}\";" << std::endl <<
					"ren tiny-imagenet-200 tinyimagenet && del /Q tiny-imagenet-200.zip" << std::endl <<
					"del UnZip-File.ps1" << std::endl;
#else
					std::string("unzip -o ./tiny-imagenet-200.zip") << std::endl <<
					std::string("mv tiny-imagenet-200 tinyimagenet") << std::endl <<
					std::string("rm tiny-imagenet-200.zip") << std::endl;
#endif
			}
			break;
			}
			
			batch.close();

			std::filesystem::permissions((DatasetsDirectory / fileName).string(), std::filesystem::perms::owner_all | std::filesystem::perms::group_all);

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)

#ifdef _UNICODE
			const std::wstring command = L"cmd /c " + (DatasetsDirectory / fileName).wstring();
			LPTSTR cmdLine = _wcsdup(command.c_str());
#else
			const std::string command = "cmd /c " + (DatasetsDirectory / fileName).string();
			LPSTR cmdLine = _strdup(command.c_str());
#endif
			STARTUPINFO info;
			PROCESS_INFORMATION processInfo;
			ZeroMemory(&info, sizeof(STARTUPINFO));
			info.cb = sizeof(STARTUPINFO);
			ZeroMemory(&processInfo, sizeof(PROCESS_INFORMATION));
			if (CreateProcess(NULL, cmdLine, NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
			{
				DWORD status = WaitForSingleObject(processInfo.hProcess, INFINITE);

				CloseHandle(processInfo.hProcess);
				CloseHandle(processInfo.hThread);
				std::free(cmdLine);

				if (status == 0ul && dataset == Datasets::tinyimagenet)
					GetTinyImageNetLabels(path / std::string(magic_enum::enum_name<Datasets>(dataset)));

				std::filesystem::remove((DatasetsDirectory / fileName));

				return status == 0ul;
			}
			else
			{
				std::free(cmdLine);

				return false;
			}
#else
			int status = std::system((DatasetsDirectory / fileName).c_str());

			if (status == 0 && dataset == Datasets::tinyimagenet)
				GetTinyImageNetLabels(path / std::string(magic_enum::enum_name<Datasets>(dataset)));

			std::filesystem::remove((DatasetsDirectory / fileName));

			return status == 0;
#endif
		}

		std::vector<Float> GetMean(const UInt N)
		{
			auto mean = std::vector<double>(C);
			
			for_i(C, C, [&](UInt c)
			{
				auto channelMean = double(0);
				auto correction = double(0);

#ifdef DNN_IMAGEDEPTH
				for (auto n = 0ull; n < N; n++)
					for (auto d = 0u; d < D; d++)
						for (auto h = 0u; h < H; h++)
							for (auto w = 0u; w < W; w++)
								KahanSum<double>(double(TrainingSamples[n](static_cast<unsigned int>(c), d, h, w)), meanC, correction);
#else
				for (auto n = 0ull; n < N; n++)
					for (auto h = 0u; h < H; h++)
						for (auto w = 0u; w < W; w++)
							KahanSum<double>(double(TrainSamples[n](static_cast<unsigned int>(c), 0, h, w)), channelMean, correction);
#endif

				mean[c] = channelMean / double(N * D * H * W);
			});

			auto result = std::vector<Float>();
			for (auto c = 0ull; c < C; c++)
				result.push_back(Float(mean[c]));

			return result;
		}
		
		std::vector<Float> GetStdDev(const std::vector<Float>& mean, const UInt N)
		{
			auto stddev = std::vector<double>(C);
			
			auto eps = double(1);
			while (double(1) + eps != double(1))
				eps /= double(2);
			
			for_i(C, C, [&](UInt c)
			{
				auto channelMean = double(mean[c]);
				auto channelStdDev = double(0);
				auto correction = double(0);

#ifdef DNN_IMAGEDEPTH
				for (auto n = 0ull; n < N; n++)
					for (auto d = 0u; d < D; d++)
						for (auto h = 0u; h < H; h++)
							for (auto w = 0u; w < W; w++)
								KahanSum<double>(Square<double>(double(TrainingSamples[n](static_cast<unsigned int>(c), d, h, w)) - channelMean), channelStdDev, correction);
#else
				for (auto n = 0ull; n < N; n++)
					for (auto h = 0u; h < H; h++)
						for (auto w = 0u; w < W; w++)
							KahanSum<double>(Square<double>(double(TrainSamples[n](static_cast<unsigned int>(c), 0, h, w)) - channelMean), channelStdDev, correction);
#endif
				
				channelStdDev /= double(N * D * H * W);
				channelStdDev = std::max(double(0), channelStdDev);
				channelStdDev = std::sqrt(channelStdDev + eps);
				stddev[c] = std::max(std::sqrt(channelStdDev), double(1) / std::sqrt(double(N * D * H * W)));
			});
			
			auto result = std::vector<Float>();
			for (auto c = 0ull; c < C; c++)
				result.push_back(Float(stddev[c]));

			return result;
		}

		bool LoadDataset(const Datasets dataset)
		{
			if (!DatasetAvailable(dataset))
			{
				GetDataset(dataset);

				if (!DatasetAvailable(dataset))
					return false;
			}

			switch (dataset)
			{
			case Datasets::cifar10:
				C = 3;
				D = 1;
				H = 32;
				W = 32;
				ClassCount = std::vector<UInt>({ 10 });
				Hierarchies = ClassCount.size();
				TrainSamplesCount = 50000;
				TestSamplesCount = 10000;
				TrainLabels = std::vector<std::vector<UInt>>(TrainSamplesCount, std::vector<UInt>(Hierarchies));
				TestLabels = std::vector<std::vector<UInt>>(TestSamplesCount, std::vector<UInt>(Hierarchies));
				Mean = std::vector<Float>({ Float(125.79831808), Float(123.43251712), Float(114.31199744) });
				StdDev = std::vector<Float>({ Float(63.24027648), Float(62.3321728), Float(66.96644608) });
				TrainSamples = ImageByteVector(TrainSamplesCount);
				TestSamples = ImageByteVector(TestSamplesCount);
				break;

			case Datasets::cifar100:
			    C = 3;
				D = 1;
				H = 32;
				W = 32;
				ClassCount = std::vector<UInt>({ 20, 100 });
				Hierarchies = ClassCount.size();
				TrainSamplesCount = 50000;
				TestSamplesCount = 10000;
				TrainLabels = std::vector<std::vector<UInt>>(TrainSamplesCount, std::vector<UInt>(Hierarchies));
				TestLabels = std::vector<std::vector<UInt>>(TestSamplesCount, std::vector<UInt>(Hierarchies));
				Mean = std::vector<Float>({ Float(129.3), Float(124.1), Float(112.4) });
				StdDev = std::vector<Float>({ Float(68.2),  Float(65.4),  Float(70.4) });
				TrainSamples = ImageByteVector(TrainSamplesCount);
				TestSamples = ImageByteVector(TestSamplesCount);
			    break;

			case Datasets::fashionmnist:
				C = 1;
				D = 1;
				H = 28;
				W = 28;
				ClassCount = std::vector<UInt>({ 10 });
				Hierarchies = ClassCount.size();
				TrainSamplesCount = 60000;
				TestSamplesCount = 10000;
				TrainLabels = std::vector<std::vector<UInt>>(TrainSamplesCount, std::vector<UInt>(Hierarchies));
				TestLabels = std::vector<std::vector<UInt>>(TestSamplesCount, std::vector<UInt>(Hierarchies));
				Mean = std::vector<Float>({ Float(72.940247) });
				StdDev = std::vector<Float>({ Float(90.021133) });
				TrainSamples = ImageByteVector();
				TestSamples = ImageByteVector();
				break;

			case Datasets::mnist:
				C = 1;
				D = 1;
				H = 28;
				W = 28;
				ClassCount = std::vector<UInt>({ 10 });
				Hierarchies = ClassCount.size();
				TrainSamplesCount = 60000;
				TestSamplesCount = 10000;
				TrainLabels = std::vector<std::vector<UInt>>(TrainSamplesCount, std::vector<UInt>(Hierarchies));
				TestLabels = std::vector<std::vector<UInt>>(TestSamplesCount, std::vector<UInt>(Hierarchies));
				Mean = std::vector<Float>({ Float(33.318443) });
				StdDev = std::vector<Float>({ Float(78.567261) });
				TrainSamples = ImageByteVector();
				TestSamples = ImageByteVector();
				break;

			case Datasets::tinyimagenet:
				C = 3;
				D = 1;
				H = 64;
				W = 64;
				ClassCount = std::vector<UInt>({ 200 });
				Hierarchies = ClassCount.size();
				TrainSamplesCount = 100000;
				TestSamplesCount = 10000;
				TrainLabels = std::vector<std::vector<UInt>>(TrainSamplesCount, std::vector<UInt>(Hierarchies));
				TestLabels = std::vector<std::vector<UInt>>(TestSamplesCount, std::vector<UInt>(Hierarchies));
				Mean = std::vector<Float>({ Float(117.56279), Float(109.588692), Float(96.981331) });
				StdDev = std::vector<Float>({ Float(69.272858), Float(67.387779), Float(70.635902) });
				TrainSamples = ImageByteVector(TrainSamplesCount);
				TestSamples = ImageByteVector(TestSamplesCount);
				break;
			}

			switch (dataset)
			{
			case Datasets::cifar10:
			{				
				auto pathTrainPatterns = std::vector<std::string>();
				pathTrainPatterns.push_back((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "data_batch_1.bin").string());
				pathTrainPatterns.push_back((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "data_batch_2.bin").string());
				pathTrainPatterns.push_back((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "data_batch_3.bin").string());
				pathTrainPatterns.push_back((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "data_batch_4.bin").string());
				pathTrainPatterns.push_back((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "data_batch_5.bin").string());
				const auto pathTestPatterns = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "test_batch.bin").string();

				auto ok = true;
				for_i(5, 5, [=, &ok](UInt batch)
				{
					auto infile = std::ifstream(pathTrainPatterns[batch], std::ios::binary | std::ios::in);

					if (ok && !infile.bad() && infile.is_open())
					{
						auto TrainPatterns = new Byte[30730000];
						infile.read(reinterpret_cast<char*>(TrainPatterns), 30730000);
						infile.close();
						const auto offset = batch * 10000;
						for (UInt index = 0; index < 10000; index++)
						{
							TrainSamples[index + offset] = Image<Byte>(3, 1, 32, 32, &TrainPatterns[3073 * index + 1]);
							TrainLabels[index + offset][0] = TrainPatterns[3073 * index];
							
						}
						delete[] TrainPatterns;
					}
					else
						ok = false;
				});

				if (!ok)
					return false;

				auto infile = std::ifstream(pathTestPatterns, std::ios::binary | std::ios::in);
				if (!infile.bad() && infile.is_open())
				{
					auto TestPatterns = new Byte[30730000];
					infile.read(reinterpret_cast<char*>(TestPatterns), 30730000);
					infile.close();
					for (UInt index = 0; index < TestSamplesCount; index++)
					{
						TestSamples[index] = Image<Byte>(3, 1, 32, 32, &TestPatterns[3073 * index + 1]);
						TestLabels[index][0] = TestPatterns[3073 * index];
					}
					delete[] TestPatterns;
				}
				else
					return false;
			}
			break;

			case Datasets::cifar100:
			{
				const auto pathTrainPatterns = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "train.bin").string();
				const auto pathTestPatterns = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "test.bin").string();

				auto infile = std::ifstream(pathTrainPatterns, std::ios::binary | std::ios::in);
				if (!infile.bad() && infile.is_open())
				{
					auto TrainPatterns = new Byte[153700000];
					infile.read(reinterpret_cast<char*>(TrainPatterns), 153700000);
					infile.close();
					for (UInt index = 0; index < 50000; index++)
					{
						TrainSamples[index] = Image<Byte>(3, 1, 32, 32, &TrainPatterns[3074 * index + 2]);
						TrainLabels[index][0] = TrainPatterns[3074 * index];
						TrainLabels[index][1] = TrainPatterns[3074 * index + 1];
					}
					delete[] TrainPatterns;
				}
				else
					return false;

				infile.open(pathTestPatterns, std::ios::binary | std::ios::in);
				if (!infile.bad() && infile.is_open())
				{
					auto TestPatterns = new Byte[30740000];
					infile.read(reinterpret_cast<char*>(TestPatterns), 30740000);
					infile.close();
					for (UInt index = 0; index < TestSamplesCount; index++)
					{
						TestSamples[index] = Image<Byte>(3, 1, 32, 32, &TestPatterns[3074 * index + 2]);
						TestLabels[index][0] = TestPatterns[3074 * index];
						TestLabels[index][1] = TestPatterns[3074 * index + 1];
					}
					delete[] TestPatterns;
				}
				else
					return false;
			}
			break;

			case Datasets::fashionmnist:
			case Datasets::mnist:
			{
				const auto pathTrainPatterns = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "train-images-idx3-ubyte").string();
				const auto pathTrainLabels = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset)) / "train-labels-idx1-ubyte").string();
				const auto pathTestPatterns = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "t10k-images-idx3-ubyte").string();
				const auto pathTestLabels = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "t10k-labels-idx1-ubyte").string();
				
				auto infile = std::ifstream(pathTrainLabels, std::ios::binary | std::ios::in);
				if (!infile.bad() && infile.is_open())
				{
					auto fileBufLabels = new Byte[60000];
					infile.seekg(8ll, std::ios::beg);
					infile.read(reinterpret_cast<char*>(fileBufLabels), 60000);
					infile.close();

					infile.open(pathTrainPatterns, std::ios::binary | std::ios::in);
					if (!infile.bad() && infile.is_open())
					{
						auto fileBuf = new Byte[47040000];
						infile.seekg(16ll, std::ios::beg);
						infile.read(reinterpret_cast<char*>(fileBuf), 47040000);
						infile.close();

						for (UInt i = 0; i < TrainSamplesCount; i++)
						{
							TrainSamples.push_back(Image<Byte>(1, 1, 28, 28, &fileBuf[i * 784]));
							TrainLabels[i][0] = static_cast<UInt>(fileBufLabels[i]);
						}

						delete[] fileBuf;
					}
					else
					{
						delete[] fileBufLabels;
						return false;
					}

					delete[] fileBufLabels;
				}
				else
					return false;

				infile.open(pathTestLabels, std::ios::binary | std::ios::in);
				if (!infile.bad() && infile.is_open())
				{
					auto fileBufLabels = new Byte[10000];
					infile.seekg(8ll, std::ios::beg);
					infile.read(reinterpret_cast<char*>(fileBufLabels), 10000);
					infile.close();

					infile.open(pathTestPatterns, std::ios::binary | std::ios::in);
					if (!infile.bad() && infile.is_open())
					{
						auto fileBuf = new Byte[7840000];
						infile.seekg(16ll, std::ios::beg);
						infile.read(reinterpret_cast<char*>(fileBuf), 7840000);
						infile.close();

						for (UInt i = 0; i < TestSamplesCount; i++)
						{
							TestSamples.push_back(Image<Byte>(1, 1, 28, 28, &fileBuf[i * 784]));
							TestLabels[i][0] = static_cast<UInt>(fileBufLabels[i]);
						}

						delete[] fileBuf;
					}
					else
					{
						delete[] fileBufLabels;
						return false;
					}

					delete[] fileBufLabels;
				}
				else
					return false;
			}
			break;

			case Datasets::tinyimagenet:
			{
				auto labels = std::vector<std::string>(TestSamplesCount);
				auto labels_idx = std::vector<UInt>(TestSamplesCount);
				ClassNames = std::vector<std::string>();

				auto infile = std::ifstream((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "wnids.txt").string());
				if (!infile.bad() && infile.is_open())
				{
					std::string line;
					while (std::getline(infile, line))
						ClassNames.push_back(line);
					infile.close();
				}
				else
					return false;

				for_i_dynamic(200ull, [=](UInt item)
				{
					const auto offset = item * 500;
					for (UInt i = 0; i < 500; i++)
					{
						const auto pos = i + offset;
						const auto fileName = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "train" / ClassNames[item] / "images" / (ClassNames[item] + "_" + std::to_string(i) + ".JPEG")).string();
#ifdef cimg_use_jpeg
						TrainSamples[pos] = Image<Byte>::LoadJPEG(fileName, true);
#endif
						TrainLabels[pos][0] = item;
					}
				});

				infile.open((DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "val" / "val_annotations.txt").string());
				if (!infile.bad() && infile.is_open())
				{
					std::string line;
					int index = 0;
					while (std::getline(infile, line, '\n'))
					{
						std::istringstream ss(line);
						std::string token;

						std::getline(ss, token, '\t');
						std::getline(ss, token, '\t');

						labels[index] = token;
						for (UInt x = 0; x < 200; x++)
						{
							if (ClassNames[x] == token)
							{
								labels_idx[index] = x;
								break;
							}
						}
						index++;
					}
					infile.close();
				}
				else
					return false;

				for_i_dynamic(TestSamplesCount, [=](UInt i)
				{
					const auto fileName = (DatasetsDirectory / std::string(magic_enum::enum_name<Datasets>(dataset))  / "val" / "images" / ("val_" + std::to_string(i) + ".JPEG")).string();
					//const auto fileName = (DatasetsDirectory() / std::string(magic_enum::enum_name<Datasets>(dataset))  / "test" / "images" / ("test_" + std::to_string(i) + ".JPEG")).string();
#ifdef cimg_use_jpeg
					TestSamples[i] = Image<Byte>::LoadJPEG(fileName, true);
#endif
					TestLabels[i][0] = labels_idx[i];
				});
			}
			break;
			}

			if constexpr (!DefaultDatasetMeanStdDev)
			{
				Mean = GetMean(TrainSamplesCount);
				StdDev = GetStdDev(Mean, TrainSamplesCount);
			}

			Dataset = dataset;

			return true;
		}

		void GetTinyImageNetLabels(const std::filesystem::path& path)
		{
			auto classnames = std::ofstream((path / "classnames.txt").string(), std::ios::trunc);

			classnames <<
				"Egyptian_cat" << std::endl <<
				"reel" << std::endl <<
				"volleyball" << std::endl <<
				"rocking_chair" << std::endl <<
				"lemon" << std::endl <<
				"bullfrog" << std::endl <<
				"basketball" << std::endl <<
				"cliff" << std::endl <<
				"espresso" << std::endl <<
				"plunger" << std::endl <<
				"parking_meter" << std::endl <<
				"German_shepherd" << std::endl <<
				"dining_table" << std::endl <<
				"monarch" << std::endl <<
				"brown_bear" << std::endl <<
				"school_bus" << std::endl <<
				"pizza" << std::endl <<
				"guinea_pig" << std::endl <<
				"umbrella" << std::endl <<
				"organ" << std::endl <<
				"oboe" << std::endl <<
				"maypole" << std::endl <<
				"goldfish" << std::endl <<
				"potpie" << std::endl <<
				"hourglass" << std::endl <<
				"seashore" << std::endl <<
				"computer_keyboard" << std::endl <<
				"Arabian_camel" << std::endl <<
				"ice_cream" << std::endl <<
				"nail" << std::endl <<
				"space_heater" << std::endl <<
				"cardigan" << std::endl <<
				"baboon" << std::endl <<
				"snail" << std::endl <<
				"coral_reef" << std::endl <<
				"albatross" << std::endl <<
				"spider_web" << std::endl <<
				"sea_cucumber" << std::endl <<
				"backpack" << std::endl <<
				"Labrador_retriever" << std::endl <<
				"pretzel" << std::endl <<
				"king_penguin" << std::endl <<
				"sulphur_butterfly" << std::endl <<
				"tarantula" << std::endl <<
				"lesser_panda" << std::endl <<
				"pop_bottle" << std::endl <<
				"banana" << std::endl <<
				"sock" << std::endl <<
				"cockroach" << std::endl <<
				"projectile" << std::endl <<
				"beer_bottle" << std::endl <<
				"mantis" << std::endl <<
				"freight_car" << std::endl <<
				"guacamole" << std::endl <<
				"remote_control" << std::endl <<
				"European_fire_salamander" << std::endl <<
				"lakeside" << std::endl <<
				"chimpanzee" << std::endl <<
				"pay-phone" << std::endl <<
				"fur_coat" << std::endl <<
				"alp" << std::endl <<
				"lampshade" << std::endl <<
				"torch" << std::endl <<
				"abacus" << std::endl <<
				"moving_van" << std::endl <<
				"barrel" << std::endl <<
				"tabby" << std::endl <<
				"goose" << std::endl <<
				"koala" << std::endl <<
				"bullet_train" << std::endl <<
				"CD_player" << std::endl <<
				"teapot" << std::endl <<
				"birdhouse" << std::endl <<
				"gazelle" << std::endl <<
				"academic_gown" << std::endl <<
				"tractor" << std::endl <<
				"ladybug" << std::endl <<
				"miniskirt" << std::endl <<
				"golden_retriever" << std::endl <<
				"triumphal_arch" << std::endl <<
				"cannon" << std::endl <<
				"neck_brace" << std::endl <<
				"sombrero" << std::endl <<
				"gasmask" << std::endl <<
				"candle" << std::endl <<
				"desk" << std::endl <<
				"frying_pan" << std::endl <<
				"bee" << std::endl <<
				"dam" << std::endl <<
				"spiny_lobster" << std::endl <<
				"police_van" << std::endl <<
				"iPod" << std::endl <<
				"punching_bag" << std::endl <<
				"beacon" << std::endl <<
				"jellyfish" << std::endl <<
				"wok" << std::endl <<
				"potters_wheel" << std::endl <<
				"sandal" << std::endl <<
				"pill_bottle" << std::endl <<
				"butcher_shop" << std::endl <<
				"slug" << std::endl <<
				"hog" << std::endl <<
				"cougar" << std::endl <<
				"crane" << std::endl <<
				"vestment" << std::endl <<
				"dragonfly" << std::endl <<
				"cash_machine" << std::endl <<
				"mushroom" << std::endl <<
				"jinrikisha" << std::endl <<
				"water_tower" << std::endl <<
				"chest" << std::endl <<
				"snorkel" << std::endl <<
				"sunglasses" << std::endl <<
				"fly" << std::endl <<
				"limousine" << std::endl <<
				"black_stork" << std::endl <<
				"dugong" << std::endl <<
				"sports_car" << std::endl <<
				"water_jug" << std::endl <<
				"suspension_bridge" << std::endl <<
				"ox" << std::endl <<
				"ice_lolly" << std::endl <<
				"turnstile" << std::endl <<
				"Christmas_stocking" << std::endl <<
				"broom" << std::endl <<
				"scorpion" << std::endl <<
				"wooden_spoon" << std::endl <<
				"picket_fence" << std::endl <<
				"rugby_ball" << std::endl <<
				"sewing_machine" << std::endl <<
				"steel_arch_bridge" << std::endl <<
				"Persian_cat" << std::endl <<
				"refrigerator" << std::endl <<
				"barn" << std::endl <<
				"apron" << std::endl <<
				"Yorkshire_terrier" << std::endl <<
				"swimming_trunks" << std::endl <<
				"stopwatch" << std::endl <<
				"lawn_mower" << std::endl <<
				"thatch" << std::endl <<
				"fountain" << std::endl <<
				"black_widow" << std::endl <<
				"bikini" << std::endl <<
				"plate" << std::endl <<
				"teddy" << std::endl <<
				"barbershop" << std::endl <<
				"confectionery" << std::endl <<
				"beach_wagon" << std::endl <<
				"scoreboard" << std::endl <<
				"orange" << std::endl <<
				"flagpole" << std::endl <<
				"American_lobster" << std::endl <<
				"trolleybus" << std::endl <<
				"drumstick" << std::endl <<
				"dumbbell" << std::endl <<
				"brass" << std::endl <<
				"bow_tie" << std::endl <<
				"convertible" << std::endl <<
				"bighorn" << std::endl <<
				"orangutan" << std::endl <<
				"American_alligator" << std::endl <<
				"centipede" << std::endl <<
				"syringe" << std::endl <<
				"go-kart" << std::endl <<
				"brain_coral" << std::endl <<
				"sea_slug" << std::endl <<
				"cliff_dwelling" << std::endl <<
				"mashed_potato" << std::endl <<
				"viaduct" << std::endl <<
				"military_uniform" << std::endl <<
				"pomegranate" << std::endl <<
				"chain" << std::endl <<
				"kimono" << std::endl <<
				"comic_book" << std::endl <<
				"trilobite" << std::endl <<
				"bison" << std::endl <<
				"pole" << std::endl <<
				"boa_constrictor" << std::endl <<
				"poncho" << std::endl <<
				"bathtub" << std::endl <<
				"grasshopper" << std::endl <<
				"walking_stick" << std::endl <<
				"Chihuahua" << std::endl <<
				"tailed_frog" << std::endl <<
				"lion" << std::endl <<
				"altar" << std::endl <<
				"obelisk" << std::endl <<
				"beaker" << std::endl <<
				"bell_pepper" << std::endl <<
				"bannister" << std::endl <<
				"bucket" << std::endl <<
				"magnetic_compass" << std::endl <<
				"meat_loaf" << std::endl <<
				"gondola" << std::endl <<
				"standard_poodle" << std::endl <<
				"acorn" << std::endl <<
				"lifeboat" << std::endl <<
				"binoculars" << std::endl <<
				"cauliflower" << std::endl <<
				"African_elephant" << std::endl;

			classnames.close();
		}
		
#ifdef cimg_use_jpeg
		static cimg_library::CImg<Byte> LoadJPEG(const std::string& fileName, const bool forceColorFormat = false) NOEXCEPT
		{
			auto img = cimg_library::CImg<Byte>().get_load_jpeg(fileName.c_str());

			if (forceColorFormat && img._spectrum == 1)
			{
				auto imgColor = cimg_library::CImg<Byte>(img._width, img._height, img._depth, 3);
#ifdef DNN_IMAGEDEPTH
				cimg_forXYZC(imgColor, w, h, d, c) { imgColor(w, h, d, c) = img(w, h, d, 0); }
#else
				cimg_forXYC(imgColor, w, h, c) { imgColor(w, h, 0, c) = img(w, h, 0, 0); }
#endif		
				return imgColor;
			}
			else
				return img;
		}
#endif

#ifdef cimg_use_png
		static cimg_library::CImg<Byte> LoadPNG(const std::string& fileName, const bool forceColorFormat = false) NOEXCEPT
		{
			auto bitsPerPixel = 0u;

			auto img = cimg_library::CImg<Byte>().get_load_png(fileName.c_str(), &bitsPerPixel);

			if (forceColorFormat && img._spectrum == 1)
			{
				auto imgColor = cimg_library::CImg<Byte>(img._width, img._height, img._depth, 3);
#ifdef DNN_IMAGEDEPTH
				cimg_forXYZC(imgColor, w, h, d, c) { imgColor(w, h, d, c) = img(w, h, d, 0); }
#else
				cimg_forXYC(imgColor, w, h, c) { imgColor(w, h, 0, c) = img(w, h, 0, 0); }
#endif
				return imgColor;
			}
			else
				return img;
		}
#endif
	};
}