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

std::string reverseComplementString(const char* sequence, int sequencelength){
    std::string rev;
    rev.resize(sequencelength);

    reverseComplementString(&rev[0], sequence, sequencelength);

    return rev;
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


int editDistance(const std::string_view& s1, const std::string_view& s2){
    const int length1 = s1.size();
    const int length2 = s2.size();

    if(length1 == 0){
        return length2;
    }

    if(length2 == 0){
        return length1;
    }

    const int height = length1+1;
    const int width = length2+1;

    constexpr int matchDistance = 0;
    constexpr int mismatchDistance = 1;
    constexpr int gapDistance = 1;

    std::vector<int> M(width, 0);
    for(int i = 1; i < width; i++){
        M[i] = gapDistance * i;
    }

    for(int r = 1; r < height; r++){
        const char b1 = s1[r-1];
        int old = M[0];
        M[0] = r * gapDistance;
        //loop over columns
        for(int c = 1; c < width; c++){
            const char b2 = s2[c-1];
            const int temp = M[c];

            const int diag = old + (b1 == b2 ? matchDistance : mismatchDistance);
            const int left = M[c-1] + gapDistance;
            const int up = M[c] + gapDistance;

            int best = diag;
            if(left < best){
                best = left;
            }
            if(up < best){
                best = up;
            }
            M[c] = best;
            old = temp;
        }
    }

    return M[length2];
}





struct Read{
    std::string name{};
    std::string comment{};
    std::string sequence{};
    std::string quality{};
};



int main(int argc, char** argv){


    if(argc < 6){
        std::cout << "Modify header of konnector output of merging to include information about read id and whether the mate was reached or not\n";
        std::cout << "Usage: " << argv[0] << " insertsize insertsizestddev konnector_total.fa originalpairedfile_interleaved output.fa output_remaining.fa output_extended.fa\n";
        return 1;
    }

    const int insertsize = std::atoi(argv[1]);
    const int insertsizestd = std::atoi(argv[2]);

    const std::string konnectorfilename = argv[3];
    const std::string originalfilename = argv[4];
    const std::string outputfilename = argv[5];
    const std::string outputremainingfilename = argv[6];
    const std::string outputextendedfilename = argv[7];

    std::ofstream outfile(outputfilename);
    assert(bool(outfile));
    std::ofstream outremainingfile(outputremainingfilename);
    assert(bool(outremainingfile));
    std::ofstream outextendedfile(outputextendedfilename);
    assert(bool(outextendedfile));

    kseqpp::KseqPP konnectorreader(konnectorfilename);
    kseqpp::KseqPP originalreader(originalfilename);

    std::int64_t originalPairId = 0;

    std::array<Read, 2> readPair{};

    Read konnectorread{};

    auto getNextOriginalReadPair = [&](){
        const int status1 = originalreader.next();

        if(status1 >= 0){
            std::swap(readPair[0].name, originalreader.getCurrentName());
            std::swap(readPair[0].comment, originalreader.getCurrentComment());
            std::swap(readPair[0].sequence, originalreader.getCurrentSequence());
            std::swap(readPair[0].quality, originalreader.getCurrentQuality());
        }else if(status1 < -1){
            std::cerr << "parser error status " << status1 << " in file " << originalfilename << '\n';
        }

        const int status2 = originalreader.next();

        if(status2 >= 0){
            std::swap(readPair[1].name, originalreader.getCurrentName());
            std::swap(readPair[1].comment, originalreader.getCurrentComment());
            std::swap(readPair[1].sequence, originalreader.getCurrentSequence());
            std::swap(readPair[1].quality, originalreader.getCurrentQuality());
        }else if(status2 < -1){
            std::cerr << "parser error status " << status2 << " in file " << originalfilename << '\n';
        }

        bool success = (status1 >= 0) && (status2 >= 0);

        return success;
    };

    auto getNextKonnectorRead = [&](){
        const int status1 = konnectorreader.next();

        if(status1 >= 0){
            std::swap(konnectorread.name, konnectorreader.getCurrentName());
            std::swap(konnectorread.comment, konnectorreader.getCurrentComment());
            std::swap(konnectorread.sequence, konnectorreader.getCurrentSequence());
            std::swap(konnectorread.quality, konnectorreader.getCurrentQuality());
        }else if(status1 < -1){
            std::cerr << "parser error status " << status1 << " in file " << konnectorfilename << '\n';
        }

        bool success = (status1 >= 0);

        return success;
    };

    int numinvalidchars = 0;
    int numinvalidreads = 0;

    std::string line;

    bool success = getNextKonnectorRead();

    while(success){
        bool pairsuccess = getNextOriginalReadPair();
        assert(pairsuccess);

        

        int numinvalid = std::count_if(konnectorread.sequence.begin(), konnectorread.sequence.end(), 
            [](char c){
                constexpr std::array<char, 10> validchars{'A','C','G','T','N', 'a','c','g','t','n'};

                return validchars.end() == std::find(validchars.begin(), validchars.end(), c);
            }
        );

        numinvalidchars += numinvalid;
        numinvalidreads += (numinvalid > 0 ? 1 : 0);        

        if(konnectorread.sequence.size() > readPair[0].sequence.size()){
            bool foundMate = false;

            //We assume that konnector read begins with read1

            //check if read 2 appears within given insert size

            // const auto revc2 = reverseComplementString(readPair[1].sequence.data(), readPair[1].sequence.size());
            // const int revcLength = revc2.size();

            // const int rangebegin = insertsize - 5 * insertsizestd;
            // const int rangeend = insertsize + 5 * insertsizestd + 1;

            // const int shiftbegin = std::max(0, rangebegin - int(revc2.size()));
            // const int shiftend = std::max(0, std::min(int(konnectorread.sequence.size()), rangeend) - int(revc2.size()) );

            // constexpr int hammingthreshold = 15;

            // int curpos = 0;
            // int curbest = 9999;

            // for(int shift = shiftbegin; shift < shiftend; shift++){
            //     const int ham = hammingDistanceFull(
            //         konnectorread.sequence.begin() + shiftbegin, 
            //         konnectorread.sequence.begin() + shiftbegin + revc2.size(), 
            //         revc2.begin(),
            //         revc2.end()
            //     );

            //     if(ham < curbest){
            //         curpos = shift;
            //         curbest = ham;
            //     }

            //     if(ham <= hammingthreshold){
            //         foundMate = true;
            //         break;
            //     }
            // }

            bool foundMate2 = false;
            int foodist2 = 0;

            //if(int(konnectorread.sequence.size()) <= insertsize + 5 * insertsizestd){
                //check that konnector read begins with read1
                //int hammingdistance1 = hammingDistanceFull(konnectorread.sequence.begin(), konnectorread.sequence.begin() + 20, readPair[0].sequence.begin(), readPair[0].sequence.begin() + 20);
                //assert(hammingdistance1 <= 4);

                //check if konnector read ends with revc(read2)
                // const auto revc = reverseComplementString(readPair[1].sequence.data(), readPair[1].sequence.size());
                // const int revcLength = revc.size();

                // int hammingdistance2 = hammingDistanceFull(konnectorread.sequence.end() - revcLength, konnectorread.sequence.end(), revc.begin(), revc.end());
                // foodist2 = hammingdistance2;
                // foundMate2 = hammingdistance2 <= 25;

                // if(!foundMate2){
                //     std::string_view s1(konnectorread.sequence.data() + konnectorread.sequence.size() - revcLength, revcLength);
                //     std::string_view s2(revc);

                //     int editdistance = editDistance(s1, s2);

                //     foundMate2 = editdistance <= 25;
                // }

                // if(!foundMate2){
                //     std::cerr << "not found\n";
                //     std::cerr << konnectorread.sequence << "\n";
                //     std::cerr << readPair[0].sequence << "\n";
                //     std::cerr << revc << "\n";
                // }
            //}

            // if(foundMate2 != foundMate){
            //     std::cerr << curpos << " " << curbest << " " << foundMate << " " << foodist2 << " "  << foundMate2 << "\n";
            //     std::cerr << konnectorread.sequence << "\n";
            //     std::cerr << readPair[0].sequence << "\n";
            //     std::cerr << readPair[1].sequence << "\n";
            //     std::cerr << revc2 << "\n";

            // }

            foundMate2 = true;

            if(foundMate2){
                outfile << '>' << (originalPairId * 2) << " reached:1 lens:0," << readPair[0].sequence.size() << "," << (konnectorread.sequence.size() - readPair[1].sequence.size()) << "," << konnectorread.sequence.size() << '\n';
                outfile << konnectorread.sequence << '\n';

                outextendedfile << '>' << (originalPairId * 2) << " reached:1 lens:0," << readPair[0].sequence.size() << "," << (konnectorread.sequence.size() - readPair[1].sequence.size()) << "," << konnectorread.sequence.size() << '\n';
                outextendedfile << konnectorread.sequence << '\n';
            }else{
                outfile << '>' << (originalPairId * 2) << " reached:0 lens:0," << readPair[0].sequence.size() << ",-1,-1" << '\n';
                outfile << konnectorread.sequence << '\n';

                outextendedfile << '>' << (originalPairId * 2) << " reached:0 lens:0," << readPair[0].sequence.size() << ",-1,-1" << '\n';
                outextendedfile << konnectorread.sequence << '\n';
            }
        }else{
            outfile << '>' << (originalPairId * 2) << " reached:0 lens:0," << readPair[0].sequence.size() << ",-1,-1" << '\n';
            outfile << konnectorread.sequence << '\n';

            outremainingfile << '>' << (originalPairId * 2) << " reached:0 lens:0," << readPair[0].sequence.size() << ",-1,-1" << '\n';
            outremainingfile << konnectorread.sequence << '\n';
        }

        originalPairId++;

        success = getNextKonnectorRead();
    }

    std::cout << "numinvalidreads = " << numinvalidreads << "\n";
    std::cout << "numinvalidchars = " << numinvalidchars << "\n";


    return 0;
}
