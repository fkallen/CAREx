#include <options.hpp>
#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <filehelpers.hpp>

#include <iostream>
#include <thread>
#include <string>
#include <stdexcept>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{


    std::string to_string(SequencePairType s){
        switch (s)
        {
        case SequencePairType::Invalid:
            return "Invalid";
        case SequencePairType::SingleEnd:
            return "SingleEnd";
        case SequencePairType::PairedEnd:
            return "PairedEnd";
        default:
            return "Error";
        }
    }

    std::string to_string(GpuDataLayout t)
    {
        switch (t)
        {
        case GpuDataLayout::FirstFit:
            return "FirstFit";
            break;
        case GpuDataLayout::EvenShare:
            return "EvenShare";
            break;
        default:
            return "Error";
            break;
        }
    }

    ProgramOptions::ProgramOptions(const cxxopts::ParseResult& pr){
        ProgramOptions& result = *this;

        if(pr.count("minalignmentoverlap")){
            result.min_overlap = pr["minalignmentoverlap"].as<int>();
        }
        if(pr.count("maxmismatchratio")){
            result.maxErrorRate = pr["maxmismatchratio"].as<float>();
        }
        if(pr.count("minalignmentoverlapratio")){
            result.min_overlap_ratio = pr["minalignmentoverlapratio"].as<float>();
        }

        if(pr.count("excludeAmbiguous")){
            result.excludeAmbiguousReads = pr["excludeAmbiguous"].as<bool>();
        }

        if(pr.count("useQualityScores")){
            result.useQualityScores = pr["useQualityScores"].as<bool>();
        }

        if(pr.count("enforceHashmapCount")){
            result.mustUseAllHashfunctions = pr["enforceHashmapCount"].as<bool>();
        }

        if(pr.count("coverage")){
            result.estimatedCoverage = pr["coverage"].as<float>();
        }

        if(pr.count("kmerlength")){
            result.kmerlength = pr["kmerlength"].as<int>();
            if(result.kmerlength == 0){
                result.autodetectKmerlength = true;
            }else{
                result.autodetectKmerlength = false;
            }
        }else{
            result.autodetectKmerlength = true;
        }

        if(pr.count("hashmaps")){
            result.numHashFunctions = pr["hashmaps"].as<int>();
        }        

        if(pr.count("batchsize")){
            result.batchsize = pr["batchsize"].as<int>();
        }

        if(pr.count("pairedFilterThreshold")){
            result.pairedFilterThreshold = pr["pairedFilterThreshold"].as<float>();
        }

        if(pr.count("minFragmentSize")){
            result.minFragmentSize = pr["minFragmentSize"].as<int>();
        }

        if(pr.count("maxFragmentSize")){
            result.maxFragmentSize = pr["maxFragmentSize"].as<int>();
        }

        if(pr.count("fixedStepsize")){
            result.fixedStepsize = pr["fixedStepsize"].as<int>();
        }

        if(pr.count("allowOutwardExtension")){
            result.allowOutwardExtension = pr["allowOutwardExtension"].as<bool>();
        }

        if(pr.count("sortedOutput")){
            result.sortedOutput = pr["sortedOutput"].as<bool>();
        }

        if(pr.count("outputRemaining")){
            result.outputRemainingReads = pr["outputRemaining"].as<bool>();
        }

        if(pr.count("threads")){
            result.threads = pr["threads"].as<int>();
        }
        result.threads = std::min(result.threads, (int)std::thread::hardware_concurrency());
      
        if(pr.count("showProgress")){
            result.showProgress = pr["showProgress"].as<bool>();
        }

        if(pr.count("gpu")){
            result.deviceIds = pr["gpu"].as<std::vector<int>>();
        }

        result.canUseGpu = result.deviceIds.size() > 0;

        if(pr.count("warpcore")){
            result.warpcore = pr["warpcore"].as<int>();
        }

        if(pr.count("gpuExtenderThreadConfig")){
            std::string configString = pr["gpuExtenderThreadConfig"].as<std::string>();

            auto split = [](const std::string& str, char c) -> std::vector<std::string>{
                std::vector<std::string> result;

                std::stringstream ss(str);
                std::string s;

                while (std::getline(ss, s, c)) {
                        result.emplace_back(s);
                }

                return result;
            };
        
            auto tokens = split(configString, ':');
            if(tokens.size() != 2){
                throw std::runtime_error("gpuExtenderThreadConfig wrong format. Expected: numExtenders:numHashers");
            }

            GpuExtenderThreadConfig x;
            x.numExtenders = std::stoi(tokens[0]);
            x.numHashers = std::stoi(tokens[1]);

            result.gpuExtenderThreadConfig = x;
        }

        

        if(pr.count("replicateGpuReadData")){
            result.replicateGpuReadData = pr["replicateGpuReadData"].as<bool>();
        }

        if(pr.count("replicateGpuHashtables")){
            result.replicateGpuHashtables = pr["replicateGpuHashtables"].as<bool>();
        }

        if(pr.count("strictExtensionMode")){
            result.strictExtensionMode = pr["strictExtensionMode"].as<int>();
        }

        if(pr.count("gpuReadDataLayout")){
            int val = pr["gpuReadDataLayout"].as<int>();
            GpuDataLayout opt;
            switch(val){
                case 0: opt = GpuDataLayout::FirstFit; break;
                case 1: opt = GpuDataLayout::EvenShare; break;
                default: opt = GpuDataLayout::FirstFit; break;
            }
            result.gpuReadDataLayout = opt;
        }

        if(pr.count("gpuHashtableLayout")){
            int val = pr["gpuHashtableLayout"].as<int>();
            GpuDataLayout opt;
            switch(val){
                case 0: opt = GpuDataLayout::FirstFit; break;
                case 1: opt = GpuDataLayout::EvenShare; break;
                default: opt = GpuDataLayout::FirstFit; break;
            }
            result.gpuHashtableLayout = opt;
        }    

        if(pr.count("fixedNumberOfReads")){
            result.fixedNumberOfReads = pr["fixedNumberOfReads"].as<std::size_t>();
        }

        auto parseMemoryString = [](const auto& string) -> std::size_t{
            if(string.length() > 0){
                std::size_t factor = 1;
                bool foundSuffix = false;
                switch(string.back()){
                    case 'K':{
                        factor = std::size_t(1) << 10; 
                        foundSuffix = true;
                    }break;
                    case 'M':{
                        factor = std::size_t(1) << 20;
                        foundSuffix = true;
                    }break;
                    case 'G':{
                        factor = std::size_t(1) << 30;
                        foundSuffix = true;
                    }break;
                }
                if(foundSuffix){
                    const auto numberString = string.substr(0, string.size()-1);
                    return factor * std::stoull(numberString);
                }else{
                    return std::stoull(string);
                }
            }else{
                return 0;
            }
        };

        if(pr.count("memTotal")){
            const auto memoryTotalLimitString = pr["memTotal"].as<std::string>();
            const std::size_t parsedMemory = parseMemoryString(memoryTotalLimitString);
            const std::size_t availableMemory = getAvailableMemoryInKB() * 1024;

            // user-provided memory limit could be greater than currently available memory.
            result.memoryTotalLimit = std::min(parsedMemory, availableMemory);
        }else{
            std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
            if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
                availableMemoryInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
            }

            result.memoryTotalLimit = availableMemoryInBytes;
        }

        if(pr.count("memHashtables")){
            const auto memoryForHashtablesString = pr["memHashtables"].as<std::string>();
            result.memoryForHashtables = parseMemoryString(memoryForHashtablesString);
        }else{
            std::size_t availableMemoryInBytes = result.memoryTotalLimit;
            if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
                availableMemoryInBytes = availableMemoryInBytes - 1*(std::size_t(1) << 30);
            }

            result.memoryForHashtables = availableMemoryInBytes;
        }

        result.memoryForHashtables = std::min(result.memoryForHashtables, result.memoryTotalLimit);

        if(pr.count("hashloadfactor")){
            result.hashtableLoadfactor = pr["hashloadfactor"].as<float>();
        }

        if(pr.count("qualityScoreBits")){
            result.qualityScoreBits = pr["qualityScoreBits"].as<int>();
        }

        if(pr.count("outdir")){
		    result.outputdirectory = pr["outdir"].as<std::string>();
        }

        // if(pr.count("pairmode")){
        //     const std::string arg = pr["pairmode"].as<std::string>();

        //     if(arg == "se" || arg == "SE"){
        //         result.pairType = SequencePairType::SingleEnd;
        //     }else if(arg == "pe" || arg == "PE"){
        //         result.pairType = SequencePairType::PairedEnd;
        //     }else{
        //         result.pairType = SequencePairType::Invalid;
        //     }
        // }  


        if(pr.count("save-preprocessedreads-to")){
            result.save_binary_reads_to = pr["save-preprocessedreads-to"].as<std::string>();
        }

        if(pr.count("load-preprocessedreads-from")){
            result.load_binary_reads_from = pr["load-preprocessedreads-from"].as<std::string>();
        }

        if(pr.count("save-hashtables-to")){
            result.save_hashtables_to = pr["save-hashtables-to"].as<std::string>();
        }

        if(pr.count("load-hashtables-from")){
            result.load_hashtables_from = pr["load-hashtables-from"].as<std::string>();
        }

        if(pr.count("tempdir")){
            result.tempdirectory = pr["tempdir"].as<std::string>();
        }else{
            result.tempdirectory = result.outputdirectory;
        }


        if(pr.count("inputfiles")){
            result.inputfiles = pr["inputfiles"].as<std::vector<std::string>>();
        }

        if(pr.count("eo")){
            result.extendedReadsOutputfilename = pr["eo"].as<std::string>();
        }

        if(pr.count("ro")){
            result.remainingReadsOutputfilename = pr["ro"].as<std::string>();
        }

    }

    bool ProgramOptions::isValid() const noexcept{
        const ProgramOptions& opt = *this;
        bool valid = true;

        if(opt.maxErrorRate < 0.0f || opt.maxErrorRate > 1.0f){
            valid = false;
            std::cout << "Error: maxmismatchratio must be in range [0.0, 1.0], is " + std::to_string(opt.maxErrorRate) << std::endl;
        }

        if(opt.min_overlap < 1){
            valid = false;
            std::cout << "Error: min_overlap must be > 0, is " + std::to_string(opt.min_overlap) << std::endl;
        }

        if(opt.min_overlap_ratio < 0.0f || opt.min_overlap_ratio > 1.0f){
            valid = false;
            std::cout << "Error: min_overlap_ratio must be in range [0.0, 1.0], is "
                        + std::to_string(opt.min_overlap_ratio) << std::endl;
        }

        if(opt.estimatedCoverage <= 0.0f){
            valid = false;
            std::cout << "Error: estimatedCoverage must be > 0.0, is " + std::to_string(opt.estimatedCoverage) << std::endl;
        }

        if(opt.batchsize < 1 /*|| corOpts.batchsize > 16*/){
            valid = false;
            std::cout << "Error: batchsize must be in range [1, ], is " + std::to_string(opt.batchsize) << std::endl;
        }

        if(opt.numHashFunctions < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(opt.numHashFunctions) << std::endl;
        }

        if(opt.kmerlength < 0 || opt.kmerlength > max_k<kmer_type>::value){
            valid = false;
            std::cout << "Error: kmer length must be in range [0, " << max_k<kmer_type>::value 
                << "], is " + std::to_string(opt.kmerlength) << std::endl;
        }

        if(opt.minFragmentSize < 0){
            valid = false;
            std::cout << "Error: expected minFragmentSize > 0, is " 
                << opt.minFragmentSize << std::endl;
        }

        if(opt.maxFragmentSize < 0){
            valid = false;
            std::cout << "Error: expected maxFragmentSize > 0, is " 
                << opt.maxFragmentSize << std::endl;
        }

        if(opt.minFragmentSize > opt.maxFragmentSize){
            valid = false;
            std::cout << "Error: expected minFragmentSize <= maxFragmentSize, is " 
                << opt.minFragmentSize << std::endl;
        }

        if(opt.fixedStepsize < 0){
            valid = false;
            std::cout << "Error: fixedStepsize must be >= 0, is " 
                << opt.fixedStepsize << std::endl;
        }

        if(opt.threads < 1){
            valid = false;
            std::cout << "Error: threads must be > 0, is " + std::to_string(opt.threads) << std::endl;
        }

        if(opt.qualityScoreBits != 1 && opt.qualityScoreBits != 2 && opt.qualityScoreBits != 8){
            valid = false;
            std::cout << "Error: qualityScoreBits must be 1,2,or 8, is " + std::to_string(opt.qualityScoreBits) << std::endl;
        }

        if(!filesys::exists(opt.tempdirectory)){
            bool created = filesys::create_directories(opt.tempdirectory);
            if(!created){
                valid = false;
                std::cout << "Error: Could not create temp directory" << opt.tempdirectory << std::endl;
            }
        }

        if(!filesys::exists(opt.outputdirectory)){
            bool created = filesys::create_directories(opt.outputdirectory);
            if(!created){
                valid = false;
                std::cout << "Error: Could not create output directory" << opt.outputdirectory << std::endl;
            }
        }

        {
            for(const auto& inputfile : opt.inputfiles){
                std::ifstream is(inputfile);
                if(!(bool)is){
                    valid = false;
                    std::cout << "Error: cannot find input file " << inputfile << std::endl;
                }
            }            
        }

        {
            std::vector<FileFormat> formats;
            for(const auto& inputfile : opt.inputfiles){
                FileFormat f = getFileFormat(inputfile);
                if(f == FileFormat::FASTQGZ)
                    f = FileFormat::FASTQ;
                if(f == FileFormat::FASTAGZ)
                    f = FileFormat::FASTA;
                formats.emplace_back(f);
            }
            bool sameFormats = std::all_of(
                formats.begin()+1, 
                formats.end(), [&](const auto f){
                    return f == formats[0];
                }
            );
            if(!sameFormats){
                valid = false;
                std::cout << "Error: Must not specify both fasta and fastq files!" << std::endl;
            }
        }

        {
            assert(opt.pairType == SequencePairType::PairedEnd); //single end extension not implemented

            //Disallow invalid type
            if(opt.pairType == SequencePairType::Invalid){
                valid = false;
                std::cout << "Error: pairmode is invalid." << std::endl;
            }

            //In paired end mode, there must be a single input file with interleaved reads, or exactly two input files, one per direction.
            if(opt.pairType == SequencePairType::PairedEnd){
                const int countOk = opt.inputfiles.size() == 1 || opt.inputfiles.size() == 2;
                if(!countOk){
                    valid = false;
                    std::cout << "Error: Invalid number of input files for selected pairmode 'PairedEnd'." << std::endl;
                }
            }

            //In single end mode, a single file allowed
            if(opt.pairType == SequencePairType::SingleEnd){
                const int countOk = opt.inputfiles.size() == 1;
                if(!countOk){
                    valid = false;
                    std::cout << "Error: Invalid number of input files for selected pairmode 'SingleEnd'." << std::endl;
                }
            }
        }

        return valid;
    }


    void ProgramOptions::printMandatoryOptions(std::ostream& stream) const{
        stream << "Output directory: " << outputdirectory << "\n";
        stream << "Estimated dataset coverage: " << estimatedCoverage << "\n";
        stream << "Input files: ";
        for(auto& s : inputfiles){
            stream << s << ' ';
        }
        stream << "\n";
    }

    void ProgramOptions::printMandatoryOptionsExtend(std::ostream& stream) const{
        stream << "Minimum fragment size: " << minFragmentSize << "\n";
	    stream << "Maximum fragment size " << maxFragmentSize << "\n";
    }

    void ProgramOptions::printMandatoryOptionsExtendCpu(std::ostream&) const{
        //nothing
    }

    void ProgramOptions::printMandatoryOptionsExtendGpu(std::ostream& stream) const{
        stream << "Can use GPU(s): " << canUseGpu << "\n";
        if(canUseGpu){
            stream << "GPU device ids: [";
            for(int id : deviceIds){
                stream << " " << id;
            }
            stream << " ]\n";
        }
    }

    void ProgramOptions::printAdditionalOptions(std::ostream& stream) const{
        stream << "Number of hash tables / hash functions: " << numHashFunctions << "\n";
        if(autodetectKmerlength){
            stream << "K-mer size for hashing: auto\n";
        }else{
            stream << "K-mer size for hashing: " << kmerlength << "\n";
        }
        stream << "Enforce number of hash tables: " << mustUseAllHashfunctions << "\n";
        stream << "Threads: " << threads << "\n";
        stream << "Use quality scores: " << useQualityScores << "\n";
        stream << "Bits per quality score: " << qualityScoreBits << "\n";
        stream << "Exclude ambigious reads: " << excludeAmbiguousReads << "\n";
        stream << "Alignment absolute required overlap: " << min_overlap << "\n";
        stream << "Alignment relative required overlap: " << min_overlap_ratio << "\n";
        stream << "Alignment max relative number of mismatches in overlap: " << maxErrorRate << "\n";
        stream << "Show progress bar: " << showProgress << "\n";
        stream << "Output directory: " << outputdirectory << "\n";
        stream << "Temporary directory: " << tempdirectory << "\n";
        stream << "Extended reads output file: " << extendedReadsOutputfilename << "\n";
        stream << "Remaining reads output file: " << remainingReadsOutputfilename << "\n";
        stream << "Save preprocessed reads to file: " << save_binary_reads_to << "\n";
        stream << "Load preprocessed reads from file: " << load_binary_reads_from << "\n";
        stream << "Save hash tables to file: " << save_hashtables_to << "\n";
        stream << "Load hash tables from file: " << load_hashtables_from << "\n";
        stream << "Maximum memory for hash tables: " << memoryForHashtables << "\n";
        stream << "Maximum memory total: " << memoryTotalLimit << "\n";
        stream << "Hashtable load factor: " << hashtableLoadfactor << "\n";
        stream << "Fixed number of reads: " << fixedNumberOfReads << "\n";    
    }

    void ProgramOptions::printAdditionalOptionsExtend(std::ostream& stream) const{
        stream << "Allow extension outside of gap: " << allowOutwardExtension << "\n";
        stream << "Sort extended reads: " << sortedOutput << "\n";
        stream << "Output remaining reads: " << outputRemainingReads << "\n";
	    stream << "fixedStepsize: " << fixedStepsize << "\n";
        stream << "Replicate GPU hashtables: " << replicateGpuHashtables << "\n";
    }

    void ProgramOptions::printAdditionalOptionsExtendCpu(std::ostream&) const{
        //nothing
    }

    void ProgramOptions::printAdditionalOptionsExtendGpu(std::ostream& stream) const{
        stream << "Batch size: " << batchsize << "\n";
        stream << "Warpcore: " << warpcore << "\n";
	    stream << "Replicate GPU reads: " << replicateGpuReadData << "\n";
        stream << "Strict extension mode: " << strictExtensionMode << "\n";
        stream << "GPU read layout " << to_string(gpuReadDataLayout) << "\n";
        stream << "GPU hashtable layout " << to_string(gpuHashtableLayout) << "\n";

        if(gpuExtenderThreadConfig.isAutomatic()){
            stream << "GPU extender thread config: auto\n";
        }else{
            stream << "GPU extender thread config: " << gpuExtenderThreadConfig.numExtenders << ":" << gpuExtenderThreadConfig.numHashers << "\n";
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


    void addMandatoryOptions(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")
            ("d,outdir", "The output directory. Will be created if it does not exist yet.", 
            cxxopts::value<std::string>())
            ("c,coverage", "Estimated coverage of input file. (i.e. number_of_reads * read_length / genome_size)", 
            cxxopts::value<float>())
            ("i,inputfiles", 
                "The reads to extend. "
                "Fasta or Fastq format. May be gzip'ed. "
                "Split format: two input files \"-i reads_1.fastq -i reads_2.fastq\", interleaved format: one input file\"reads_interleaved.fastq\" "
                "Must not mix fasta and fastq files. ",
                cxxopts::value<std::vector<std::string>>())
            // ("pairmode", 
            //     "Type of input reads."
            //     "SE / se : Single-end reads"
            //     "PE / pe : Paired-end reads",
            //     cxxopts::value<std::string>())
            
            ; // end options
    }

    void addMandatoryOptionsExtend(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")
            ("minFragmentSize", 
                "Minimum fragment size to consider. Must be > 2*readlength", 
                cxxopts::value<int>())
            ("maxFragmentSize", 
                "Maximum fragment size to consider. Must be > minFragmentSize.", 
                cxxopts::value<int>());
    }

    void addMandatoryOptionsExtendCpu(cxxopts::Options&){
        //nothing
    }

    void addMandatoryOptionsExtendGpu(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")        
		    ("g,gpu", "Comma-separated list of GPU device ids to be used. (Example: --gpu 0,1 to use GPU 0 and GPU 1)", cxxopts::value<std::vector<int>>());
    }

    void addAdditionalOptions(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Additional")
            ("eo", 
                "The name of the output file containing extended reads",
                cxxopts::value<std::string>())
            ("ro", 
                "The name of the output file containing remaining reads",
                cxxopts::value<std::string>())
            ("h,hashmaps", "The requested number of hash maps. Must be greater than 0. "
                "The actual number of used hash maps may be lower to respect the set memory limit. "
                "Default: " + tostring(ProgramOptions{}.numHashFunctions), 
                cxxopts::value<int>())
            ("k,kmerlength", "The kmer length for minhashing. If 0 or missing, it is automatically determined.", cxxopts::value<int>())
            ("enforceHashmapCount",
                "If the requested number of hash maps cannot be fullfilled, the program terminates without extension. "
                "Default: " + tostring(ProgramOptions{}.mustUseAllHashfunctions),
                cxxopts::value<bool>()->implicit_value("true")
            )
            ("t,threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>())            
            ("q,useQualityScores", "If set, quality scores (if any) are considered during read extension. "
                "Default: " + tostring(ProgramOptions{}.useQualityScores),
            cxxopts::value<bool>()->implicit_value("true"))
            ("qualityScoreBits", "How many bits should be used to store a single quality score. Allowed values: 1,2,8. If not 8, a lossy compression via binning is used."
                "Default: " + tostring(ProgramOptions{}.qualityScoreBits), cxxopts::value<int>())
            ("excludeAmbiguous", 
                "If set, reads which contain at least one ambiguous nucleotide will not be processed. "
                "Default: " + tostring(ProgramOptions{}.excludeAmbiguousReads),
            cxxopts::value<bool>()->implicit_value("true"))
            ("maxmismatchratio", "Overlap between anchor and candidate must contain at "
                "most (maxmismatchratio * overlapsize) mismatches. "
                "Default: " + tostring(ProgramOptions{}.maxErrorRate),
            cxxopts::value<float>())
            ("minalignmentoverlap", "Overlap between anchor and candidate must be at least this long. "
                "Default: " + tostring(ProgramOptions{}.min_overlap),
            cxxopts::value<int>())
            ("minalignmentoverlapratio", "Overlap between anchor and candidate must be at least as "
                "long as (minalignmentoverlapratio * candidatelength). "
                "Default: " + tostring(ProgramOptions{}.min_overlap_ratio),
            cxxopts::value<float>())
            ("p,showProgress", "If set, progress information is shown during execution",
            cxxopts::value<bool>()->implicit_value("true"))
            ("tempdir", "Directory to store temporary files. Default: output directory", cxxopts::value<std::string>())
            ("save-preprocessedreads-to", "Save binary dump of data structure which stores input reads to disk",
            cxxopts::value<std::string>())
            ("load-preprocessedreads-from", "Load binary dump of read data structure from disk",
            cxxopts::value<std::string>())
            ("save-hashtables-to", "Save binary dump of hash tables to disk. Ignored for GPU hashtables.",
            cxxopts::value<std::string>())
            ("load-hashtables-from", "Load binary dump of hash tables from disk. Ignored for GPU hashtables.",
            cxxopts::value<std::string>())
            ("memHashtables", "Memory limit in bytes for hash tables and hash table construction. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: A bit less than memTotal.",
            cxxopts::value<std::string>())
            ("m,memTotal", "Total memory limit in bytes. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: All free memory.",
            cxxopts::value<std::string>())
            ("hashloadfactor", "Load factor of hashtables. 0.0 < hashloadfactor < 1.0. Smaller values can improve the runtime at the expense of greater memory usage."
                "Default: " + std::to_string(ProgramOptions{}.hashtableLoadfactor), cxxopts::value<float>())
            ("fixedNumberOfReads", "Process only the first n reads. Default: " + tostring(ProgramOptions{}.fixedNumberOfReads), cxxopts::value<std::size_t>()); 
    }

    void addAdditionalOptionsExtend(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Additional")
            ("allowOutwardExtension", "Will try to fill the gap and extend to the outside"
                "Default: " + tostring(ProgramOptions{}.allowOutwardExtension), cxxopts::value<bool>()->implicit_value("true"))	
            ("sortedOutput", "Extended reads in output file will be sorted by read id."
                "Default: " + tostring(ProgramOptions{}.sortedOutput), cxxopts::value<bool>()->implicit_value("true"))	
            ("outputRemaining", "Output remaining reads which could not be extended. Will be sorted by read id."
                "Default: " + tostring(ProgramOptions{}.outputRemainingReads), cxxopts::value<bool>()->implicit_value("true"))
            ("fixedStepsize", "fixedStepsize "
                "Default: " + tostring(ProgramOptions{}.fixedStepsize), cxxopts::value<int>())
            ("strictExtensionMode", "Use strict mode to reject more pseudo-reads in favor of more accurate pseudo-read lengths. "
                "0: No strict mode. "
                "1: Only consider pseudo-reads where at least one strand reached the mate."
                "2: Only consider pseudo-reads where both strands reached the mate."
                "Default: " + std::to_string(ProgramOptions{}.strictExtensionMode), cxxopts::value<int>());
    }

    void addAdditionalOptionsExtendCpu(cxxopts::Options&){
        //nothing	
    }

    void addAdditionalOptionsExtendGpu(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Additional")
            ("batchsize", "Number of reads in a single batch. Must be greater than 0. "
                "Default: " + tostring(ProgramOptions{}.batchsize),
                cxxopts::value<int>())
            ("warpcore", "Enable warpcore hash tables. 0: Disabled, 1: Enabled. "
                "Default: " + tostring(ProgramOptions{}.warpcore),
                cxxopts::value<int>())
            ("replicateGpuReadData", "If reads fit into the memory of a single GPU, allow its replication to other GPUs. This can improve the runtime when multiple GPUs are used."
                "Default: " + std::to_string(ProgramOptions{}.replicateGpuReadData), cxxopts::value<bool>())
            ("replicateGpuHashtables", "Construct warpcore hashtables on a single GPU, then replicate them on each GPU."
                "Default: " + std::to_string(ProgramOptions{}.replicateGpuHashtables), cxxopts::value<bool>())            
            ("gpuReadDataLayout", "GPU read layout. 0: first fit, 1: even share", cxxopts::value<int>())
            ("gpuHashtableLayout", "GPU hash table layout. 0: first fit, 1: even share", cxxopts::value<int>())
            ("gpuExtenderThreadConfig", "Per-GPU thread configuration for extension. Format numExtenders(int):numHashers(int)."
                "Default: automatic configuration (0:0). "
                "Example: 2:8 . When numExtenders:0 is used, each extender performs hashing itself. This is recommended with gpu hashtables.", 
                cxxopts::value<std::string>());
    }

} //namespace care