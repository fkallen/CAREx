// g++ -std=c++17 -O3 -g filterBySimpleRepeat.cpp -lpthread -lz -o filterBySimpleRepeat

#include <string>
#include <vector>
#include <iostream>
#include <fstream>


#include "kseqpp/kseqpp.hpp"


struct Read{
    std::string name{};
    std::string comment{};
    std::string sequence{};
    std::string quality{};
};



int main(int argc, char** argv){

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " genomeregionrepeatinfo.txt extendedreads.fast* norepeatout.fasta repeatout.fasta\n";
        return 1;
    }

    std::string repeatinfofilename = argv[1]; // fasta file with one genome region per read pair in paired-end read file
    std::string extendedReadsFilename = argv[2]; 
    std::string norepeatoutFilename = argv[3]; 
    std::string repeatoutFilename = argv[4]; 

    std::ifstream repeatinfofile(repeatinfofilename);
    assert(bool(repeatinfofile));

    std::ofstream norepeatoutfile(norepeatoutFilename);
    assert(bool(norepeatoutfile));

    std::ofstream repeatoutfile(repeatoutFilename);
    assert(bool(repeatoutfile));

    std::int64_t genomeRegionId = 0;

    kseqpp::KseqPP reader(extendedReadsFilename);
    Read read{};

    std::int64_t readNumberInFile = 0;
    std::string line;
    bool success = false;

    auto getNextRead = [&](){
        const int status = reader.next();
        //std::cerr << "parser status = 0 in file " << filenames[i] << '\n';
        if(status >= 0){
            std::swap(read.name, reader.getCurrentName());
            std::swap(read.comment, reader.getCurrentComment());
            std::swap(read.sequence, reader.getCurrentSequence());
            std::swap(read.quality, reader.getCurrentQuality());
        }else if(status < -1){
            std::cerr << "parser error status " << status << " in file " << extendedReadsFilename << '\n';
        }

        bool success = (status > 0);

        return success;
    };

    success = getNextRead();

    while(success){

        std::int64_t readId = readNumberInFile;
        try{
            readId = std::stoll(read.name);
        }catch(std::invalid_argument&){
            std::cerr << "error getting read id\n";
        }

        while(genomeRegionId < readId / 2){
            std::getline(repeatinfofile, line);
            assert(bool(repeatinfofile));
            genomeRegionId++;
        }

        std::getline(repeatinfofile, line); //repeatinfo of current read
        assert(bool(repeatinfofile));

        genomeRegionId++;
        readNumberInFile++;

        if(line == "0 0"){
            norepeatoutfile << ">" << read.name << " " << read.comment << "\n";
            norepeatoutfile << read.sequence << "\n";
        }else{
            repeatoutfile << ">" << read.name << " " << read.comment << "\n";
            repeatoutfile << read.sequence << "\n";
        }

        success = getNextRead(); //prepare next loop iteration
    }   

    return 0;
}



