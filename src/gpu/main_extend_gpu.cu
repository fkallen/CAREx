#include <version.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <options.hpp>
#include <gpu/dispatch_care_extend_gpu.cuh>

#include <threadpool.hpp>

#include <readlibraryio.hpp>

#include <fstream>
#include <iostream>
#include <ios>
#include <string>
#include <omp.h>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

using namespace care;

void printCommandlineArguments(std::ostream& out, const cxxopts::ParseResult& parseresults){

	const auto args = parseresults.arguments();
	for(const auto& opt : args){
		out << opt.key() << '=' << opt.value() << '\n';
	}
}

bool checkMandatoryArguments(const cxxopts::ParseResult& parseresults){

	const std::vector<std::string> mandatory = {
		"inputfiles", "outdir", "coverage",
		"minFragmentSize", "maxFragmentSize"
	};

	bool success = true;
	for(const auto& opt : mandatory){
		if(parseresults.count(opt) == 0){
			success = false;
			std::cerr << "Mandatory argument " << opt << " is missing.\n";
		}
	}

	return success;
}

void listAvailableGpus(std::ostream& os){
	int numGpus = 0;
    cudaError_t status = cudaGetDeviceCount(&numGpus);
	if(status == cudaSuccess){
		os << "Found " << numGpus << " CUDA-capable GPUs in your system\n";
		for(int deviceId = 0; deviceId < numGpus; deviceId++){
			os << "Device ID " << deviceId << ": ";

			cudaDeviceProp prop;
			status = cudaGetDeviceProperties(&prop, deviceId);
			if(status == cudaSuccess){
				os << prop.name << ", compute capability " << prop.major << "." << prop.minor << "\n";
			}else{
				os << "Error: Could not retrieve device properties for device ID " << deviceId << "\n";
			}
		}
	}else{
		os << "Error: Could not determine number of GPUs\n";
	}
}


template<class T>
std::string tostring(const T& t){
	return std::to_string(t);
}

template<>
std::string tostring(const bool& b){
	return b ? "true" : "false";
}

int main(int argc, char** argv){

	bool help = false;
	bool showVersion = false;
	bool printAvailableGpus = false;

	cxxopts::Options commandLineOptions(argv[0], "CAREx GPU");

	addMandatoryOptions(commandLineOptions);
	addMandatoryOptionsExtend(commandLineOptions);
	addMandatoryOptionsExtendGpu(commandLineOptions);

	addAdditionalOptions(commandLineOptions);
	addAdditionalOptionsExtend(commandLineOptions);
	addAdditionalOptionsExtendGpu(commandLineOptions);

	commandLineOptions.add_options("Additional")
		("help", "Show this help message", cxxopts::value<bool>(help))
		("version", "Print version", cxxopts::value<bool>(showVersion))
		("listGpus", "Print a list of available GPUs in the system and their corresponding device ID", cxxopts::value<bool>(printAvailableGpus));



	auto parseresults = commandLineOptions.parse(argc, argv);

	if(showVersion){
		std::cout << "CAREx version " << CAREX_VERSION_STRING << std::endl;
		std::exit(0);
	}

	if(help) {
		std::cout << commandLineOptions.help({"", "Mandatory", "Additional"}) << std::endl;
		std::exit(0);
	}

	if(printAvailableGpus){
		listAvailableGpus(std::cout);
		std::exit(0);
	}


	const bool mandatoryPresent = checkMandatoryArguments(parseresults);
	if(!mandatoryPresent){
		std::cout << commandLineOptions.help({"Mandatory"}) << std::endl;
		std::exit(0);
	}

	ProgramOptions programOptions(parseresults);

	if(!programOptions.isValid()) throw std::runtime_error("Invalid program options!");

    programOptions.deviceIds = extension::getUsableDeviceIds(programOptions.deviceIds);
    programOptions.canUseGpu = programOptions.deviceIds.size() > 0;

	if(programOptions.useQualityScores){
		const bool hasQ = std::all_of(
			programOptions.inputfiles.begin(),
			programOptions.inputfiles.end(),
			[](const auto& s){
				return hasQualityScores(s);
			}
		);

		if(!hasQ){
			std::cerr << "Quality scores have been disabled because there exist reads in an input file without quality scores.\n";
			
			programOptions.useQualityScores = false;
		}
	}

	std::cout << std::boolalpha;
	std::cout << "CARE EXTEND GPU will be started with the following parameters:\n";
	std::cout << "----------------------------------------\n";

	programOptions.printMandatoryOptions(std::cout);
	programOptions.printMandatoryOptionsExtend(std::cout);
	programOptions.printMandatoryOptionsExtendGpu(std::cout);

	programOptions.printAdditionalOptions(std::cout);
	programOptions.printAdditionalOptionsExtend(std::cout);
	programOptions.printAdditionalOptionsExtendGpu(std::cout);

	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;

	// if(programOptions.pairType == SequencePairType::SingleEnd || programOptions.pairType == SequencePairType::Invalid){
	// 	std::cout << "Only paired-end extension is supported. Abort.\n";
	// 	std::exit(0);
	// }

	if(!programOptions.canUseGpu){
		std::cout << "No valid GPUs selected. Abort\n";
		std::exit(0);
	}

    const int numThreads = programOptions.threads;

	omp_set_num_threads(numThreads);

	care::performExtension(
		programOptions
	);

	return 0;
}
