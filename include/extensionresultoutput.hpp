#ifndef CARE_EXTENSION_RESULT_OUTPUT_HPP
#define CARE_EXTENSION_RESULT_OUTPUT_HPP

#include <options.hpp>

#include <serializedobjectstorage.hpp>
#include <readlibraryio.hpp>
#include <extendedread.hpp>

#include <string>
#include <vector>


namespace care{


void constructOutputFileFromExtensionResults(
    SerializedObjectStorage& partialResults,
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const ProgramOptions& programOptions
);

void constructOutputFileFromExtensionResults(
    const std::vector<std::string>& originalReadFiles,
    SerializedObjectStorage& partialResults,
    const std::vector<read_number>& idsOfNotExtended,
    FileFormat outputFormat,
    const std::string& extendedOutputfile,
    const std::string& remainingOutputfile,
    SequencePairType pairmode,
    const ProgramOptions& programOptions
);


Read makeOutputReadFromExtendedRead(const ExtendedRead& extendedRead, FileFormat outputFormat, const ProgramOptions& programOptions);

void outputUnchangedReadPairs(
    const std::vector<std::string>& originalReadFiles,
    const std::vector<read_number>& idsToOutput,
    FileFormat outputFormat,
    const std::string& outputfile
);

}



#endif