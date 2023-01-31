#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <cassert>
#include <cctype>
#include <map>
#include <limits>
#include <array>
#include <numeric>
#include <cmath>
#include <stdexcept>


#include "kseqpp/kseqpp.hpp"

std::uint64_t linecount(const std::string& filename){
	std::uint64_t count = 0;
	std::ifstream is(filename);
	if(is){
		std::string s;
		while(std::getline(is, s)){
			++count;
		}
	}
	return count;
}

std::vector<std::string> split(const std::string& str, char c){
	std::vector<std::string> result;

	std::stringstream ss(str);
	std::string s;

	while (std::getline(ss, s, c)) {
		result.emplace_back(s);
	}

	return result;
}

std::string str_toupper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
        [](unsigned char c){ return std::toupper(c); }
    );
    return s;
}

void reverseComplementString(char* reverseComplement, const char* sequence, int sequencelength){
    for(int i = 0; i < sequencelength; ++i){
        switch(sequence[i]){
            case 'A': reverseComplement[sequencelength-1-i] = 'T'; break;
            case 'C': reverseComplement[sequencelength-1-i] = 'G'; break;
            case 'G': reverseComplement[sequencelength-1-i] = 'C'; break;
            case 'T': reverseComplement[sequencelength-1-i] = 'A'; break;
            default : reverseComplement[sequencelength-1-i] = sequence[i]; // don't change N
        }
    }
}


struct Read{
    std::string name{};
    std::string comment{};
    std::string sequence{};
    std::string quality{};
};



int main(int argc, char** argv){


    if(argc < 4){
        std::cout << "Reorder reads by genome region names given in .fai. Does not change the ordering of reads from the same genoome region.\n";
        std::cout << "Usage: " << argv[0] << " .fai-file inputreads outputreads [lowmem]\n";
        std::cout << "If lowmem = 1 program tries to reduce memory consumption.\n";
        return 1;
    }

    const std::string faifilename = argv[1];
    const std::string inputfilename = argv[2];
    const std::string outputfilename = argv[3];

    bool lowmem = false;
    if(argc > 4){
        lowmem = std::atoi(argv[4]);
    }

    std::ofstream outfile(outputfilename);
    assert(bool(outfile));

    kseqpp::KseqPP originalreader(inputfilename);

    Read curRead{};

    auto getNextRead = [&](){
        const int status1 = originalreader.next();

        if(status1 >= 0){
            std::swap(curRead.name, originalreader.getCurrentName());
            std::swap(curRead.comment, originalreader.getCurrentComment());
            std::swap(curRead.sequence, originalreader.getCurrentSequence());
            std::swap(curRead.quality, originalreader.getCurrentQuality());
        }else if(status1 < -1){
            std::cerr << "parser error status " << status1 << " in file " << inputfilename << '\n';
        }

        bool success = (status1 >= 0);

        return success;
    };

    //load fai region names
    std::vector<std::string> regionnames;
    {
        std::string line;
        std::ifstream namestream(faifilename);

        while(std::getline(namestream, line)){
            auto tokens = split(line, '\t');
            assert(tokens.size() > 0);
            regionnames.emplace_back(tokens[0]);
        }
    }

    assert(regionnames.size() > 0);

    if(!lowmem){
        std::vector<Read> allreads;

        bool success = getNextRead();

        while(success){

            allreads.emplace_back(std::move(curRead));

            success = getNextRead();
        }

        auto nextregionbegin = allreads.begin();

        for(size_t i = 0; i < regionnames.size(); i++){
            nextregionbegin = std::stable_partition(
                nextregionbegin,
                allreads.end(),
                //check if read header begins with name of region i
                [&](const Read& read){
                    return 0 == read.name.find(regionnames[i]);
                }
            );
        }

        for(const auto& read : allreads){
            const bool fastq = (read.sequence.size() == read.quality.size());

            outfile << (fastq ? '@' : '>') << read.name <<"\n";
            outfile << read.sequence << "\n";
            if(fastq){
                outfile << '+' << "\n";
                outfile << read.quality << "\n";
            }
        }
    }else{
        throw std::runtime_error("lowmem not implemented");
        // for(size_t i = 0; i < regionnames.size(); i++){
        //     const auto& regionname = regionnames[i];

        //     originalreader.kseq_rewind();

        // }
        //     while(success){

        //     const bool fastq = (curRead.sequence.size() == curRead.quality.size());

        //     outfile << (fastq ? '@' : '>') << curRead.name <<"\n";
        //     outfile << curRead.sequence << "\n";
        //     if(fastq){
        //         outfile << '+' << "\n";
        //         outfile << curRead.quality << "\n";
        //     }

        //     success = getNextRead();
        // }
    }




    return 0;
}
