// g++ -std=c++17 -O3 -g markRepeatsInReadRegions.cpp -lpthread -lz -o markRepeatsInReadRegions

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
#include <string_view>
#include <optional>

#include <omp.h>

#include "kseqpp/kseqpp.hpp"


struct Result{
    int leftRepeats = 0;
    int rightRepeats = 0;
};



std::string str_toupper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
        [](unsigned char c){ return std::toupper(c); }
    );
    return s;
}

void str_toupper_inplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), 
        [](unsigned char c){ return std::toupper(c); }
    );
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

std::string reverseComplementString(const char* sequence, int sequencelength){
    std::string rev;
    rev.resize(sequencelength);

    reverseComplementString(&rev[0], sequence, sequencelength);

    return rev;
}

template<class Iter1, class Iter2>
int hammingDistanceOverlap(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
    auto isEqual = [](const auto& l, const auto& r){
        return l == r;
    };

    return hammingDistanceOverlap(first1, last1, first2, last2, isEqual);
}

template<class Iter1, class Iter2, class Equal>
int hammingDistanceOverlap(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
    int result = 0;

    while(first1 != last1 && first2 != last2){
        result += isEqual(*first1, *first2) ? 0 : 1;

        ++first1;
        ++first2;
    }

    return result;
}

template<class Iter1, class Iter2>
int hammingDistanceFull(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
    auto isEqual = [](const auto& l, const auto& r){
        return l == r;
    };

    return hammingDistanceFull(first1, last1, first2, last2, isEqual);
}

template<class Iter1, class Iter2, class Equal>
int hammingDistanceFull(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
    int result = 0;

    while(first1 != last1 && first2 != last2){
        result += isEqual(*first1, *first2) ? 0 : 1;

        ++first1;
        ++first2;
    }

    //positions which do not overlap count as mismatch.
    //at least one of the remaining ranges is empty
    result += std::distance(first1, last1);
    result += std::distance(first2, last2);

    return result;
}


Result processSequence(int maxHammingDistance, std::string& sequence, int origReadLength, int minF, int maxF){
    Result result;

    const int seqLength = sequence.size();
    const int minMateStart = std::max(origReadLength, minF - origReadLength);
    const int maxMateStart = std::min(seqLength - origReadLength, maxF - origReadLength);
    
    std::string_view mate(sequence.c_str() + seqLength - origReadLength, origReadLength);

    //exclusive loop to avoid self check
    for(int i = minMateStart; i < maxMateStart; i++){
        const int hamming = hammingDistanceFull(sequence.begin() + i, sequence.begin() + i + origReadLength, mate.begin(), mate.end());
        if(hamming <= maxHammingDistance){
            result.rightRepeats++;
        }
    }
    std::reverse(sequence.begin(), sequence.end());

    //now mate is left hand read

    //exclusive loop to avoid self check
    for(int i = minMateStart; i < maxMateStart; i++){
        const int hamming = hammingDistanceFull(sequence.begin() + i, sequence.begin() + i + origReadLength, mate.begin(), mate.end());
        if(hamming <= maxHammingDistance){
            result.leftRepeats++;
        }
    }

    return result;
}

int main(int argc, char** argv){

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " readlength minF maxF genomeregions.fasta maxHammingDistance\n";
        return 1;
    }

    const int readLengthBeforeExtension = std::atoi(argv[1]);

    const int minF = std::atoi(argv[2]);
    const int maxF = std::atoi(argv[3]);

    std::string genomeregionsFilename = argv[4]; // fasta file with one genome region per read pair in paired-end read file

    const int maxHammingDistance = std::atoi(argv[5]);

    std::int64_t genomeRegionId = 0;
    kseqpp::KseqPP reader(genomeregionsFilename);


    int atLeastOneLeft = 0;
    int atLeastOneRight = 0;
    int atLeastOne = 0;

    while(reader.next() >= 0){
        Result result = processSequence(maxHammingDistance, reader.getCurrentSequence(), readLengthBeforeExtension, minF, maxF);
        std::cout << result.leftRepeats << " " << result.rightRepeats << "\n";

        if(result.leftRepeats > 0){
            atLeastOneLeft++;
        }
        if(result.rightRepeats > 0){
            atLeastOneRight++;
        }
        if(result.leftRepeats > 0 || result.rightRepeats > 0){
            atLeastOne++;
        }

        genomeRegionId++;
    }


    std::cerr << "atLeastOneLeft = "<< atLeastOneLeft << ", atLeastOneRight = " << atLeastOneRight << ", atLeastOne = " << atLeastOne << "\n";


}