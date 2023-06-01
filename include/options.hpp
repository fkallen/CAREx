#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include <config.hpp>
#include <readlibraryio.hpp>

#include "cxxopts/cxxopts.hpp"

#include <string>
#include <vector>
#include <ostream>

namespace care
{

    enum class SequencePairType
    {
        Invalid,
        SingleEnd,
        PairedEnd,
    };

    enum class GpuDataLayout{
        FirstFit,
        EvenShare
    };

    std::string to_string(SequencePairType s);
    std::string to_string(GpuDataLayout t);

    struct GpuExtenderThreadConfig{
        int numExtenders = 0;
        int numHashers = 0;

        bool isAutomatic() const noexcept{
            return numExtenders == 0 && numHashers == 0;
        }
    };


    //Options which can be parsed from command-line arguments

    struct ProgramOptions{
        bool excludeAmbiguousReads = false;
        bool useQualityScores = false;
        bool autodetectKmerlength = false;
        bool mustUseAllHashfunctions = false;
        bool allowOutwardExtension = false;
        bool sortedOutput = false;
        bool outputRemainingReads = false;
        bool showProgress = false;
        bool canUseGpu = false;
        bool replicateGpuReadData = false;
        bool replicateGpuHashtables = false;
        int strictExtensionMode = 0;
        int warpcore = 0;
        int threads = 1;
        int batchsize = 2048;
        int kmerlength = 20;
        int numHashFunctions = 48;
        int qualityScoreBits = 8;
        int minFragmentSize{};
        int maxFragmentSize{};
        int fixedStepsize = 20;
        int min_overlap = 50;
        float maxErrorRate = 0.05f;
        float min_overlap_ratio = 0.50f;
        float estimatedCoverage = 1.0f;
        float pairedFilterThreshold = 0.06f;
        float hashtableLoadfactor = 0.8f;
        GpuDataLayout gpuReadDataLayout = GpuDataLayout::EvenShare;
        GpuDataLayout gpuHashtableLayout = GpuDataLayout::EvenShare;
        SequencePairType pairType = SequencePairType::PairedEnd;
        std::size_t fixedNumberOfReads = 0;
        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;

        GpuExtenderThreadConfig gpuExtenderThreadConfig;


        std::vector<int> deviceIds;
        std::string outputdirectory = "";
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory = "";
        std::string extendedReadsOutputfilename = "carex_extended.fastq";
        std::string remainingReadsOutputfilename = "carex_remaining.fastq";
        std::vector<std::string> inputfiles;

        ProgramOptions() = default;
        ProgramOptions(const ProgramOptions&) = default;
        ProgramOptions(ProgramOptions&&) = default;

        ProgramOptions(const cxxopts::ParseResult& pr);

        bool isValid() const noexcept;

        void printMandatoryOptions(std::ostream& stream) const;
        void printMandatoryOptionsExtend(std::ostream& stream) const;
        void printMandatoryOptionsExtendCpu(std::ostream& stream) const;
        void printMandatoryOptionsExtendGpu(std::ostream& stream) const;

        void printAdditionalOptions(std::ostream& stream) const;
        void printAdditionalOptionsExtend(std::ostream& stream) const;
        void printAdditionalOptionsExtendCpu(std::ostream& stream) const;
        void printAdditionalOptionsExtendGpu(std::ostream& stream) const;
    };


    template<class ReadStorage>
    std::size_t getNumReadsToProcess(const ReadStorage* readStorage, const ProgramOptions& options){
        if(options.fixedNumberOfReads == 0){ 
            return readStorage->getNumberOfReads();
        }else{
            return options.fixedNumberOfReads;
        }
    }


    void addMandatoryOptions(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsCorrect(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsCorrectCpu(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsCorrectGpu(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsExtend(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsExtendCpu(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsExtendGpu(cxxopts::Options& commandLineOptions);

    void addAdditionalOptions(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsCorrect(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsExtend(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsCorrectCpu(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsCorrectGpu(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsExtendCpu(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsExtendGpu(cxxopts::Options& commandLineOptions);


}

#endif
