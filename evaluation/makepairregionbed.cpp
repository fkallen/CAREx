#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cctype>
#include <algorithm>
#include <map>


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

int main(int argc, char** argv){
	/*{
		std::cout << "Usage: " << argv[0] << " samfile" << std::endl;
        std::cout << "Given SAM file of paired reads, all properly aligned, create bed file which for each read pair contains the genome region of the read pair" << std::endl;
        std::cout << "If file is missing, stdin is used" << std::endl;
		return 0;
	}*/

	std::string alignmentFilename = "/dev/stdin";
    if(argc > 1){
        alignmentFilename = argv[1];
    }

	std::ifstream alignmentFile(alignmentFilename);

	if(!alignmentFile){
		std::cout << "cannot open file " << alignmentFilename << std::endl;
		return 0;
	}

    //0-based SAM columns
    constexpr int column_RNAME = 2;
    constexpr int column_POS = 3;
    constexpr int column_PNEXT = 7;
    constexpr int column_TLEN = 8;

	std::string line;


    while(std::getline(alignmentFile, line)){
        if(line[0] != '@'){
            auto tokens = split(line, '\t');

            const std::string& rname = tokens[column_RNAME];
            const std::int64_t pos = std::stoll(tokens[column_POS]);
            const std::int64_t pnext = std::stoll(tokens[column_PNEXT]);
            const std::int64_t tlen = std::stoll(tokens[column_TLEN]);

            assert(tlen >= 0);
            assert(pos < pnext);

            std::cout << rname << '\t' << (pos - 1) << '\t' << (pos -1 + tlen) << '\n';
		}
	}

	return 0;
}
