#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <array>
#include <cassert>

#include "kseqpp/kseqpp.hpp"

std::string str_toupper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), 
        [](unsigned char c){ return std::toupper(c); }
    );
    return s;
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

struct Genome{
    std::map<std::string, std::string> data;
};


Genome parseGenome(const std::string& filename){
    kseqpp::KseqPP reader(filename);

    auto getNextSequence = [&](){
        const int status = reader.next();
        if(status >= 0){
        }else if(status < -1){
            std::cerr << "parser error status " << status << " in file " << filename << '\n';
        }

        bool success = (status >= 0);

        return success;
    };

    Genome genome;

    bool success = getNextSequence();

    while(success){
        std::string s = str_toupper(reader.getCurrentSequence());
        genome.data[str_toupper(reader.getCurrentName())] = "";
        genome.data[str_toupper(reader.getCurrentName())].swap(s);

        success = getNextSequence();
    }

    return genome;
}



struct CigarOp{
	int count;
	char op;
};

std::vector<CigarOp> parseCigarstring(const std::string& cigar){

	constexpr std::array<char, 8> allowedCigarOpCharacters{'M','D','I','S','H','=','X','N'};
	
	if(cigar.size() != 0){
		if(cigar.size() == 1 && cigar[0] == '*'){
			return {};
		}

		std::vector<CigarOp> result;

		CigarOp cigarop{1, 'F'};

		int l = 0;
		while(l < int(cigar.size())){
			if(std::isdigit(cigar[l])){
				//search for last consecutive digit, the convert digits to number
				int r = l+1;
				while(r < int(cigar.size()) && std::isdigit(cigar[r])){
					r++;
				}
				std::string countstr = cigar.substr(l, r-l);
				cigarop.count = std::stoi(countstr);
				
				l = r;
			}else{
				cigarop.op = cigar[l];

				assert(std::find(allowedCigarOpCharacters.begin(), allowedCigarOpCharacters.end(), cigarop.op) != allowedCigarOpCharacters.end());

				//op character ends the current cigar operation
				result.emplace_back(cigarop);
				cigarop = CigarOp{1, 'F'};

				l++;
			}

		}

		return result;
	}else{
		return {};
	}
}



int main(int argc, char** argv){

    if(argc < 4){
        std::cout << "Usage: " << argv[0] << " samfile genomefile readlength\n";
        return 1;
    }

    std::string samFilename = argv[1];
    std::string genomeFilename = argv[2];
    const int readLengthBeforeExtension = std::atoi(argv[3]);


    std::ifstream samfile(samFilename);
    assert(samfile);

    Genome genome = parseGenome(genomeFilename);

    std::uint64_t numseqs = 0;
    std::uint64_t numedits = 0;
    std::uint64_t numbases = 0;
    std::uint64_t clipped = 0;

    std::map<int, std::uint64_t> editsmap;

    std::string samline;
    while(std::getline(samfile, samline)){
        if(samline.size() > 0 && samline[0] != '@'){
            auto tokens = split(samline, '\t');
            assert(tokens.size() >= 10);

            const auto& chromosomename = tokens[2];
            const auto& startpos = std::stoi(tokens[3]) - 1; //sam position is 1-based
            const auto& cigarstring = tokens[5];
            const auto& sequence = tokens[9];

            const auto& chromosome = genome.data[str_toupper(chromosomename)];
            const int chromosomelength = chromosome.size();

            std::vector<CigarOp> cigarops = parseCigarstring(cigarstring);
            constexpr std::array<char, 6> clipOps = {'S', 'H'};
            const bool hasClipOps = std::any_of(cigarops.begin(), cigarops.end(), 
                [&](const auto& o){
                    return clipOps.end() != std::find(clipOps.begin(), clipOps.end(), o.op);
                }
            );
            if(hasClipOps){
                clipped++;
                continue;
            }

            int checkbegin = readLengthBeforeExtension;
            int checkend = std::max(0, int(sequence.size()) - readLengthBeforeExtension);
            numbases += (checkend - checkbegin);
            int edits = 0;
            
            int readpos = 0;
            int genomepos = startpos;
            for(const auto& cigarop : cigarops){
                constexpr std::array<char, 6> consumeReadOps = {'M', '=', 'X', 'I', 'S', 'H'};
                constexpr std::array<char, 5> consumeGenomeOps = {'M', '=', 'X', 'D', 'N'};

                const bool consumeRead = std::find(consumeReadOps.begin(), consumeReadOps.end(), cigarop.op) != consumeReadOps.end();
                const bool consumeGenome = std::find(consumeGenomeOps.begin(), consumeGenomeOps.end(), cigarop.op) != consumeGenomeOps.end();

                for(int i = 0; i < cigarop.count; i++){
                    if(readpos >= checkbegin && readpos < checkend){
                        if(cigarop.op == 'I'){
                            edits++;
                        }else if(cigarop.op == 'D'){
                            edits++;
                        }else if(cigarop.op == 'M'){
                            assert(genomepos < chromosomelength);

                            char readbase = std::toupper(sequence[readpos]);
                            char genomebase = std::toupper(chromosome[genomepos]);
                            if(readbase != genomebase){
                                edits++;
                            }
                        }else{
                            
                        }
                    }

                    if(consumeRead){
                        readpos++;
                    }
                    if(consumeGenome){
                        genomepos++;
                    }
                }
            }
            numseqs++;
            numedits += edits;
            editsmap[edits]++;
        }
    }

    std::cout << "numseqs: " << numseqs << ", clipped: " << clipped << ", numedits: " << numedits << ", numbases: " << numbases << "\n";
    for(auto pair : editsmap){
        std::cout << "( " << pair.first << " " << pair.second << " )\n";
    }
}