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
#include <iterator>



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


struct GapFillerRead{
    bool reachedMate;
    std::int64_t readId;
    std::string sequence;

    void parseHeader(const std::string& header){
        auto cend = header.find("contig") + 6;
        auto underscore = header.find("_", cend);
        std::string numbersub = header.substr(cend, underscore - cend);
        readId = (std::stoll(numbersub)-1) * 2;

        if(std::string::npos != header.find("pairFound", underscore)){
            reachedMate = true;
        }else{
            reachedMate = false;
        }
    }

    friend std::istream& operator>>(std::istream& is, GapFillerRead& gr){

        std::string line;
        std::getline(is, line);

        if(is){
            gr.parseHeader(line);
            std::getline(is, gr.sequence);
        }

        

        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, const GapFillerRead& gr){
        os << '>' << gr.readId << " reached:" << (gr.reachedMate ? '1' : '0') << '\n';
        os << gr.sequence << '\n';

        return os;
    }
};



int main(int argc, char** argv){

    if(argc < 4){
        std::cout << "Usage: " << argv[0] << " gapfiller.fasta gapfiller_trash.fasta output.fasta\n";
        return 1;
    }

    std::string filename1 = argv[1];
    std::string filename2 = argv[2];
    std::string outputname = argv[3];

    std::ifstream in1(filename1);
    std::ifstream in2(filename2);
    std::ofstream out(outputname);

    assert(in1);
    assert(in2);
    assert(out);

    auto comp = [](const auto&l, const auto& r){
        return l.readId < r.readId;
    };

    std::merge(std::istream_iterator<GapFillerRead>(in1),
                std::istream_iterator<GapFillerRead>(),
                std::istream_iterator<GapFillerRead>(in2),
                std::istream_iterator<GapFillerRead>(),
                std::ostream_iterator<GapFillerRead>(out, ""),
                comp);

    return 0;
}