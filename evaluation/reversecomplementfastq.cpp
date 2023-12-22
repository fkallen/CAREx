#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

void reverseComplementInplace(std::string& seq){
    std::reverse(seq.begin(), seq.end());
    for(auto& c : seq){
        if(c == 'A') c = 'T';
        else if(c == 'C') c = 'G';
        else if(c == 'G') c = 'C';
        else if(c == 'T') c = 'A';
        else if(c == 'a') c = 't';
        else if(c == 'c') c = 'g';
        else if(c == 'g') c = 'c';
        else if(c == 't') c = 'a';
    }
}

int main(int argc, char** argv){
    if(argc < 2) return -1;

    std::ifstream inputfile(argv[1]);
    std::string line;

    size_t numReads = 0;

    while(std::getline(inputfile, line)){
        //header
        std::cout << line << '\n';


        //sequence
        std::getline(inputfile, line);
        reverseComplementInplace(line);
        std::cout << line << '\n';

        //separator
        std::getline(inputfile, line);
        std::cout << line << '\n';

        //quality
        std::getline(inputfile, line);
        std::reverse(line.begin(), line.end());
        std::cout << line << '\n';

        numReads++;
    }

    std::cerr << "Processed " << numReads << " reads\n";

}