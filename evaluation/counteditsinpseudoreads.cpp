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

#ifdef __CUDACC__

#define CUDA_HELPERS_DONT_INCLUDE_V11_GROUP_HEADERS

#include "cudaerrorcheck.cuh"
#include "hpc_helpers/include/hpc_helpers.h"
#include "hpc_helpers/include/utility_kernels.cuh"
#include "hpc_helpers/include/nvtx_markers.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#endif 

struct Read{
    std::string name{};
    std::string comment{};
    std::string sequence{};
    std::string quality{};
};

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


int semiglobalmatches(const std::string_view& s1, const std::string_view& s2){
    const int length1 = s1.size();
    const int length2 = s2.size();

    if(length1 == 0){
        return 0;
    }

    if(length2 == 0){
        return 0;
    }

    const int height = length1+1;
    const int width = length2+1;

    constexpr int matchScore = 1;
    constexpr int mismatchScore = 0;
    constexpr int gapScore = 0;

    std::vector<int> M(width, 0);
    for(int i = 1; i < width; i++){
        M[i] = 0;
    }

    int bestInLastCol = 0;
    int bestInLastRow = 0;
    // int bestCol = 0;
    // int bestRow = 0;

    for(int r = 1; r < height; r++){
        const char b1 = s1[r-1];
        int old = M[0];
        M[0] = 0;
        //loop over columns
        for(int c = 1; c < width; c++){
            const char b2 = s2[c-1];
            const int temp = M[c];

            const int diag = old + (b1 == b2 ? matchScore : mismatchScore);
            const int left = M[c-1] + gapScore;
            const int up = M[c] + gapScore;

            int best = diag;
            if(left > best){
                best = left;
            }
            if(up > best){
                best = up;
            }
            M[c] = best;
            old = temp;

            if(r == height-1){
                if(best > bestInLastRow){
                    bestInLastRow = best;
                    //bestCol = c;
                }
            }
        }

        if(M[width-1] > bestInLastCol){
            bestInLastCol = M[width-1];
            //bestRow = r;
        }
    }

    const int bestScore = std::max(bestInLastCol, bestInLastRow);

    return bestScore;
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

#ifdef __CUDACC__

//sequences are stored transposed
__global__
void editdistance_kernel(
    int* __restrict__ result,
    const char* __restrict__ sequences1,
    const int* __restrict__ lengths1,
    const char* __restrict__ sequences2,
    const int* __restrict__ lengths2,
    const std::size_t numSequences,
    const std::size_t maxLength,
    int* __restrict__ tempstorage // (maxLength + 1) * number of threads in grid
){
    const std::size_t tid = std::size_t(threadIdx.x) + std::size_t(blockIdx.x) * std::size_t(blockDim.x);
    const std::size_t stride = std::size_t(blockDim.x) * std::size_t(gridDim.x);

    //compute result[s]
    for(std::size_t s = tid; s < numSequences; s += stride){
        const int length1 = lengths1[s];
        const int length2 = lengths2[s];

        if(length1 == 0){
            result[s] = length2;
        }else if(length2 == 0){
            result[s] = length1;
        }else{

            const char* const sequence1 = sequences1 + s;
            const char* const sequence2 = sequences2 + s;
            int* const M = tempstorage + tid;

            const int height = length1+1;
            const int width = length2+1;

            constexpr int matchDistance = 0;
            constexpr int mismatchDistance = 1;
            constexpr int gapDistance = 1;

            M[0] = 0;
            for(int i = 1; i < width; i++){
                M[(numSequences) * i] = gapDistance * i;
            }

            for(int r = 1; r < height; r++){
                const char b1 = sequence1[numSequences * (r-1)];
                int old = M[0];
                M[0] = r * gapDistance;
                //loop over columns
                for(int c = 1; c < width; c++){
                    const char b2 = sequence2[numSequences * (c-1)];
                    const int temp = M[(numSequences) * c];

                    const int diag = old + (b1 == b2 ? matchDistance : mismatchDistance);
                    const int left = M[(numSequences) * (c-1)] + gapDistance;
                    const int up = M[(numSequences) * c] + gapDistance;

                    int best = diag;
                    if(left < best){
                        best = left;
                    }
                    if(up < best){
                        best = up;
                    }
                    M[(numSequences) * c] = best;
                    old = temp;
                }
            }

            result[s] = M[(numSequences) * length2];
        }
    }
}

// template<int blocksize>
// __global__
// void editdistance_kernel_smem(
//     int* __restrict__ result,
//     const char* __restrict__ sequences1,
//     const int* __restrict__ lengths1,
//     const char* __restrict__ sequences2,
//     const int* __restrict__ lengths2,
//     const std::size_t numSequences,
//     const std::size_t maxLength,
//     int* __restrict__ tempstorage // (maxLength + 1) * number of threads in grid
// ){
//     assert(blocksize == blockDim.x);

//     const std::size_t tid = std::size_t(threadIdx.x) + std::size_t(blockIdx.x) * std::size_t(blockDim.x);
//     const std::size_t stride = std::size_t(blockDim.x) * std::size_t(gridDim.x);

//     //compute result[s]
//     for(std::size_t s = tid; s < numSequences; s += stride){
//         const int length1 = lengths1[s];
//         const int length2 = lengths2[s];

//         if(length1 == 0){
//             result[s] = length2;
//         }else if(length2 == 0){
//             result[s] = length1;
//         }else{

//             const char* const sequence1 = sequences1 + s;
//             const char* const sequence2 = sequences2 + s;
//             int* const M = tempstorage + tid;

//             const int height = length1+1;
//             const int width = length2+1;

//             constexpr int matchDistance = 0;
//             constexpr int mismatchDistance = 1;
//             constexpr int gapDistance = 1;

//             M[0] = 0;
//             for(int i = 1; i < width; i++){
//                 M[(numSequences) * i] = gapDistance * i;
//             }

//             __shared__ char sharedsequences[48*1024];
//             constexpr int maxlenPerThread = (48*1024) / blocksize;

//             char* const mySharedSequence2 = &sharedsequences[threadIdx.x];

//             const int maxLengthInShared = min(maxlenPerThread, length2);
//             for(int c = 0; c < maxLengthInShared; c++){
//                 mySharedSequence2[blocksize * c] = sequence2[numSequences * c];
//             }

//             for(int r = 1; r < height; r++){
//                 const char b1 = sequence1[numSequences * (r-1)];
//                 int old = M[0];
//                 M[0] = r * gapDistance;
//                 //loop over columns
//                 for(int c = 1; c < width; c++){
//                     const char b2 = [&](){
//                         if(c < maxLengthInShared)
//                             return mySharedSequence2[blocksize * c];
//                         else
//                             return sequence2[numSequences * (c-1)];
//                     }();

//                     const int temp = M[(numSequences) * c];

//                     const int diag = old + (b1 == b2 ? matchDistance : mismatchDistance);
//                     const int left = M[(numSequences) * (c-1)] + gapDistance;
//                     const int up = M[(numSequences) * c] + gapDistance;

//                     int best = diag;
//                     if(left < best){
//                         best = left;
//                     }
//                     if(up < best){
//                         best = up;
//                     }
//                     M[(numSequences) * c] = best;
//                     old = temp;
//                 }
//             }

//             result[s] = M[(numSequences) * length2];
//         }
//     }
// }

template<int blocksize>
__global__
void editdistance_kernel_smem(
    int* __restrict__ result,
    const char* __restrict__ sequences1,
    const int* __restrict__ lengths1,
    const char* __restrict__ sequences2,
    const int* __restrict__ lengths2,
    const std::size_t numSequences,
    const std::size_t maxLength,
    int* __restrict__ tempstorage // (maxLength + 1) * number of threads in grid
){
    const std::size_t tid = std::size_t(threadIdx.x) + std::size_t(blockIdx.x) * std::size_t(blockDim.x);
    const std::size_t stride = std::size_t(blockDim.x) * std::size_t(gridDim.x);

    //compute result[s]
    for(std::size_t s = tid; s < numSequences; s += stride){
        const int length1 = lengths1[s];
        const int length2 = lengths2[s];

        if(length1 == 0){
            result[s] = length2;
        }else if(length2 == 0){
            result[s] = length1;
        }else{

            const char* const sequence1 = sequences1 + s;
            const char* const sequence2 = sequences2 + s;
            int* const myTempStorage = tempstorage + tid;

            const int height = length1+1;
            const int width = length2+1;

            for(int i = 0; i < maxLength; i++){
                myTempStorage[blocksize * i] = 0;
            }

            constexpr int matchDistance = 0;
            constexpr int mismatchDistance = 1;
            constexpr int gapDistance = 1;

            constexpr int batchsize_width = 32;
            __shared__ int sharedM[blocksize * batchsize_width];

            int* const myM = &sharedM[threadIdx.x];

            const int numBatchesWidth = SDIV(width, batchsize_width);

            for(int bw = 0; bw < numBatchesWidth; bw++){
                int firstColumn = bw * batchsize_width;
                const int endColumn = min((bw+1) * batchsize_width, width);

                for(int i = firstColumn; i < endColumn; i++){
                    myM[blocksize * (i - firstColumn)] = gapDistance * i;
                }

                if(bw == 0) firstColumn = 1;

                for(int r = 1; r < height; r++){
                    const char b1 = sequence1[numSequences * (r-1)];

                    int old = myTempStorage[blocksize * r];
                    myM[blocksize * 0] = old + r * gapDistance;

                    //loop over columns
                    for(int c = firstColumn; c < endColumn; c++){
                        const char b2 = sequence2[numSequences * (c-1)];
                        const int temp = myM[(blocksize) * (c - firstColumn)];

                        const int diag = old + (b1 == b2 ? matchDistance : mismatchDistance);
                        const int left = myM[(blocksize) * ((c - firstColumn)-1)] + gapDistance;
                        const int up = myM[(blocksize) * (c - firstColumn)] + gapDistance;

                        int best = diag;
                        if(left < best){
                            best = left;
                        }
                        if(up < best){
                            best = up;
                        }
                        myM[(blocksize) * (c - firstColumn)] = best;
                        old = temp;
                    }

                    myTempStorage[blocksize * r] = myM[(blocksize) * ((endColumn - 1) - firstColumn)];
                }

            }

            result[s] =  myTempStorage[blocksize * height];
        }
    }
}


std::vector<int> 
editDistance_cuda(const std::vector<std::string_view>& s1, const std::vector<std::string_view>& s2){
    assert(s1.size() == s2.size());

    if(s1.size() == 0) return {};

    const std::size_t numComputations = s1.size();

    const std::size_t maxLength1 = std::max_element(s1.begin(), s1.end(), [](const auto& l, const auto& r){
        return l.length() < r.length();
    })->length();

    const std::size_t maxLength2 = std::max_element(s2.begin(), s2.end(), [](const auto& l, const auto& r){
        return l.length() < r.length();
    })->length();

    const std::size_t maxLength = SDIV(std::max(maxLength1, maxLength2), 4) * 4; //pad to 4

    // auto getTotalLength = [](const auto& vec){
    //     return std::accumulate(vec.begin(), vec.end(), std::size_t(0),
    //         [](auto current, const auto& sv){ return current + sv.length(); }
    //     );
    // }

    // const std::size_t totalLengthS1 = getTotalLength(s1);
    // const std::size_t totalLengthS2 = getTotalLength(s2);

    cudaStream_t stream = 0;
    //auto thrustpolicy = thrust::cuda::par.on(stream);

    std::vector<char> sequences1(numComputations * maxLength); 
    std::vector<int> lengths1(numComputations);

    for(std::size_t i = 0; i < numComputations; i++){
        std::copy(s1[i].begin(), s1[i].end(), sequences1.begin() + i * maxLength);
        lengths1[i] = s1[i].length();
    }

    thrust::device_vector<char> d_sequences1(numComputations * maxLength);
    thrust::device_vector<int> d_lengths1(numComputations);
    //thrust::device_vector<int> d_offsets1(numComputations);

    thrust::copy(sequences1.begin(), sequences1.end(), d_sequences1.begin());
    thrust::copy(lengths1.begin(), lengths1.end(), d_lengths1.begin());
    //thrust::exclusive_scan(thrustpolicy, d_lengths1.begin(), d_lengths1.end(), d_offsets1.begin());
    thrust::device_vector<char> d_sequences1_transposed(numComputations * maxLength);

    CUDACHECK(cudaStreamSynchronize(stream));

    helpers::call_transpose_kernel(
        thrust::raw_pointer_cast(d_sequences1_transposed.data()), 
        thrust::raw_pointer_cast(d_sequences1.data()), 
        numComputations, 
        maxLength, 
        maxLength, 
        stream
    );

    d_sequences1 = thrust::device_vector<char>(); //destroy 


    std::vector<char> sequences2(numComputations * maxLength); 
    std::vector<int> lengths2(numComputations);

    for(std::size_t i = 0; i < numComputations; i++){
        std::copy(s2[i].begin(), s2[i].end(), sequences2.begin() + i * maxLength);
        lengths2[i] = s2[i].length();
    }

    thrust::device_vector<char> d_sequences2(numComputations * maxLength);
    thrust::device_vector<int> d_lengths2(numComputations);
    //thrust::device_vector<int> d_offsets2(numComputations);

    thrust::copy(sequences2.begin(), sequences2.end(), d_sequences2.begin());
    thrust::copy(lengths2.begin(), lengths2.end(), d_lengths2.begin());
    //thrust::exclusive_scan(thrustpolicy, d_lengths2.begin(), d_lengths2.end(), d_offsets2.begin());

    thrust::device_vector<char> d_sequences2_transposed(numComputations * maxLength);    

    CUDACHECK(cudaStreamSynchronize(stream));

    helpers::call_transpose_kernel(
        thrust::raw_pointer_cast(d_sequences2_transposed.data()), 
        thrust::raw_pointer_cast(d_sequences2.data()), 
        numComputations, 
        maxLength, 
        maxLength, 
        stream
    );

    d_sequences2 = thrust::device_vector<char>(); //destroy 

    thrust::device_vector<int> d_results(numComputations);

    dim3 block(128);
    dim3 grid((numComputations + block.x - 1) / block.x);

    thrust::device_vector<int> d_temp((maxLength + 1) * grid.x * block.x);

    editdistance_kernel<<<grid, block, 0, stream>>>(
        thrust::raw_pointer_cast(d_results.data()),
        thrust::raw_pointer_cast(d_sequences1_transposed.data()),
        thrust::raw_pointer_cast(d_lengths1.data()),
        thrust::raw_pointer_cast(d_sequences2_transposed.data()),
        thrust::raw_pointer_cast(d_lengths2.data()),
        numComputations,
        maxLength,
        thrust::raw_pointer_cast(d_temp.data())
    );

    CUDACHECKASYNC;
    CUDACHECK(cudaStreamSynchronize(stream));


    std::vector<int> results(numComputations);
    thrust::copy(d_results.begin(), d_results.end(), results.begin());


    CUDACHECK(cudaStreamSynchronize(stream));

    return results;

}

std::vector<int> 
editDistance_cuda(const std::vector<std::string>& s1, const std::vector<std::string>& s2){
    std::vector<std::string_view> s1sv(s1.size());
    std::vector<std::string_view> s2sv(s2.size());
    std::copy(s1.begin(), s1.end(), s1sv.begin());
    std::copy(s2.begin(), s2.end(), s2sv.begin());

    return editDistance_cuda(s1sv, s2sv);
}

#endif



struct EditDistanceTraceback{
    int score;
    std::string s1_aligned;
    std::string s2_aligned;
};

EditDistanceTraceback editDistanceWithTraceback(const std::string_view& s1, const std::string_view& s2){
    const int length1 = s1.size();
    const int length2 = s2.size();

    const int height = length1+1;
    const int width = length2+1;

    constexpr int matchDistance = 0;
    constexpr int mismatchDistance = 1;
    constexpr int gapDistance = 1;

    std::vector<int> M(width, 0);
    for(int i = 1; i < width; i++){
        M[i] = gapDistance * i;
    }

    constexpr int traceleft = 1;
    constexpr int tracediag = 2;
    constexpr int traceup = 3;

    std::vector<int> traceback(height * width, 0);
    for(int c = 1; c < width; c++){
        traceback[0 * width + c] = traceleft;
    }
    for(int r = 1; r < height; r++){
        traceback[r * width + 0] = traceup;
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

            int trace = tracediag;
            int best = diag;
            if(left < best){
                best = left;
                trace = traceleft;
            }
            if(up < best){
                best = up;
                trace = traceup;
            }
            M[c] = best;
            old = temp;

            traceback[r * width + c] = trace;
        }
    }

    //follow traceback pointers

    EditDistanceTraceback result;
    result.score = M[length2];

    int r = height-1;
    int c = width-1;

    while(!((r == 0) && (c == 0))){
        switch(traceback[r * width + c]){
            case traceleft:
                assert(c > 0);
                result.s1_aligned.push_back('-');
                result.s2_aligned.push_back(s2[c-1]);
                --c;
                break;
            case tracediag:
                assert(c > 0);
                assert(r > 0);
                result.s1_aligned.push_back(s1[r-1]);
                result.s2_aligned.push_back(s2[c-1]);
                --c;
                --r;
                break;
            case traceup:
                assert(r > 0);
                result.s1_aligned.push_back(s1[r-1]);
                result.s2_aligned.push_back('-');
                --r;
                break;
        }
    }

    std::reverse(result.s1_aligned.begin(), result.s1_aligned.end());
    std::reverse(result.s2_aligned.begin(), result.s2_aligned.end());

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
            // std::swap(read.name, reader.getCurrentName());
            // std::swap(read.comment, reader.getCurrentComment());
            // std::swap(read.sequence, reader.getCurrentSequence());
            // std::swap(read.quality, reader.getCurrentQuality());
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

struct GenomeHits{
    std::map<std::string, std::vector<int>> data{}; //map sequence name to counts per position in sequence

    GenomeHits() = default;

    GenomeHits(const std::string& filename){
        init(filename);
    }

    void init(const std::string& filename){
        destroy();

        kseqpp::KseqPP reader(filename);

        auto getNextSequence = [&](){
            const int status = reader.next();
            if(status >= 0){
                // std::swap(read.name, reader.getCurrentName());
                // std::swap(read.comment, reader.getCurrentComment());
                // std::swap(read.sequence, reader.getCurrentSequence());
                // std::swap(read.quality, reader.getCurrentQuality());
            }else if(status < -1){
                std::cerr << "parser error status " << status << " in file " << filename << '\n';
            }

            bool success = (status >= 0);

            return success;
        };

        bool success = getNextSequence();

        while(success){

            data[reader.getCurrentName()]
                .resize(reader.getCurrentSequence().length(), 0);

            success = getNextSequence();
        }
    }

    void destroy(){
        std::map<std::string, std::vector<int>> tmp{};
        data.swap(tmp);
    }

    void addHit(const std::string& sequencename, std::int64_t firstPos, std::int64_t lastPos, int numReadPairs){
        auto it = data.find(sequencename);

        assert(it != data.end());
        const std::int64_t size = it->second.size();
        
        const std::int64_t firstPosIncl = firstPos;
        const std::int64_t lastPosExcl = lastPos;

        if(firstPosIncl >= size || firstPosIncl < 0 || lastPosExcl > size || lastPosExcl < 0){
            std::cerr << "numReadPairs = " << numReadPairs << " " 
                << "firstPosIncl = " << firstPosIncl << " lastPosExcl = " << lastPosExcl << "\n";
        }
        
        assert(firstPosIncl < size);
        assert(lastPosExcl <= size);
        assert(firstPosIncl >= 0);
        assert(lastPosExcl >= 0);

        //#pragma omp parallel for
        for(auto p = firstPosIncl; p < lastPosExcl; p++){
            it->second[p]++;
        }
    }


    void printStatistics(std::ostream& os = std::cout){

        for(const auto& pair : data){
            std::int64_t numPos = pair.second.size();
            std::int64_t numPosHit = getNumPositionsWithHits(pair.first, 1);
            double numPosHitPercent = double(numPosHit) / double(numPos) * 100.0;
            std::int64_t numHits = getNumHits(pair.first);
            double mean = double(numHits) / double(numPos);
            double variance = getNumHitsVariance(pair.first, mean);
            double stddev = std::sqrt(variance);

            std::cout << pair.first << " length: " << numPos << ", positions hit: " << numPosHit 
                << " (" << numPosHitPercent << "%)"
                << ", total hits: " << numHits << ", avg hits per pos: " << mean
                << ", std deviation: " << stddev << "\n";
        }
    }
   
    std::int64_t getNumPositionsWithHits(const std::string& sequencename, int minHits) const{
        auto it = data.find(sequencename);
        assert(it != data.end());

        std::int64_t result = std::count_if(
            it->second.begin(),
            it->second.end(),
            [&](auto x){return x >= minHits;}
        );

        return result;
    }

    std::int64_t getNumHits(const std::string& sequencename) const{
        auto it = data.find(sequencename);
        assert(it != data.end());

        std::int64_t result = std::accumulate(
            it->second.begin(),
            it->second.end(),
            std::int64_t{0}
        );

        return result;
    }

    double getNumHitsVariance(const std::string& sequencename, double mean) const{
        auto it = data.find(sequencename);
        assert(it != data.end());

        double result = std::transform_reduce(
            it->second.begin(),
            it->second.end(),
            double{0},
            std::plus<double>{},
            [&](auto x){return (x-mean) * (x-mean);}
        );

        return result / std::distance(it->second.begin(), it->second.end());
    }
};



struct GenomeCover{
    std::map<std::string, std::vector<bool>> coverdata{}; //map sequence name to bool flag per position in sequence indicating a match

    GenomeCover() = default;

    GenomeCover(const std::string& filename){
        init(filename);
    }

    void init(const std::string& filename){
        destroy();

        kseqpp::KseqPP reader(filename);

        auto getNextSequence = [&](){
            const int status = reader.next();
            if(status >= 0){
                // std::swap(read.name, reader.getCurrentName());
                // std::swap(read.comment, reader.getCurrentComment());
                // std::swap(read.sequence, reader.getCurrentSequence());
                // std::swap(read.quality, reader.getCurrentQuality());
            }else if(status < -1){
                std::cerr << "parser error status " << status << " in file " << filename << '\n';
            }

            bool success = (status >= 0);

            return success;
        };

        bool success = getNextSequence();

        while(success){

            coverdata[reader.getCurrentName()]
                .resize(reader.getCurrentSequence().length(), false);

            success = getNextSequence();
        }
    }

    void destroy(){
        std::map<std::string, std::vector<bool>> tmp{};
        coverdata.swap(tmp);
    }

    void addHit(const std::string& sequencename, std::int64_t firstPos, std::int64_t lastPos, int numReadPairs){
        auto it = coverdata.find(sequencename);

        assert(it != coverdata.end());
        const std::int64_t size = it->second.size();
        
        const std::int64_t firstPosIncl = firstPos;
        const std::int64_t lastPosExcl = lastPos;

        if(firstPosIncl >= size || firstPosIncl < 0 || lastPosExcl > size || lastPosExcl < 0){
            std::cerr << "numReadPairs = " << numReadPairs << " " 
                << "firstPosIncl = " << firstPosIncl << " lastPosExcl = " << lastPosExcl << "\n";
        }
        
        assert(firstPosIncl < size);
        assert(lastPosExcl <= size);
        assert(firstPosIncl >= 0);
        assert(lastPosExcl >= 0);

        for(auto p = firstPosIncl; p < lastPosExcl; p++){
            it->second[p] = true;
        }
    }


    void printStatistics(std::ostream& os = std::cout){

        for(const auto& pair : coverdata){
            std::int64_t numPos = pair.second.size();
            std::int64_t numPosHit = getNumPositionsWithHits(pair.first);
            double numPosHitPercent = double(numPosHit) / double(numPos) * 100.0;

            std::cout << pair.first << " length: " << numPos << ", positions hit: " << numPosHit 
                << " (" << numPosHitPercent << "%)\n";
        }
    }
   
    std::int64_t getNumPositionsWithHits(const std::string& sequencename) const{
        auto it = coverdata.find(sequencename);
        assert(it != coverdata.end());

        std::int64_t result = std::count_if(
            it->second.begin(),
            it->second.end(),
            [&](bool x){return x;}
        );

        return result;
    }

};





struct LengthStatistics{
    int minlen = std::numeric_limits<int>::max();
    int maxlen = 0;
    float averagelen = 0.0f;
    std::uint64_t num = 1;

    void observedLength(int length){
        updateMin(length);
        updateMax(length);
        updateAverage(length);

        num++;
    }

    void reset(){
        minlen = std::numeric_limits<int>::max();
        maxlen = 0;
        averagelen = 0.0f;
        num = 1;
    }

    void updateMin(int length){
        minlen = std::min(minlen, length);
    }

    void updateMax(int length){
        maxlen = std::max(maxlen, length);
    }

    void updateAverage(int length){
        averagelen = averagelen + (float(length) - averagelen)/num;
    }
};

std::ostream& operator<<(std::ostream& os, const LengthStatistics& stat){
    os << "{";
    os << "min: " << stat.minlen << ", ";
    os << "max: " << stat.maxlen << ", ";
    os << "avg: " << stat.averagelen;
    os << "}";
    return os;
}

LengthStatistics operator+(const LengthStatistics& l, const LengthStatistics& r){
    LengthStatistics result;

    result.minlen = std::min(l.minlen, r.minlen);
    result.maxlen = std::max(l.maxlen, r.maxlen);
    result.num = l.num + r.num;
    result.averagelen = ((double(l.averagelen) * l.num) + (double(r.averagelen) * r.num)) / result.num;
    
    return result;
}

struct EditDistanceStatistics{
    std::uint64_t numPairs = 0;
    std::uint64_t positionsInGap = 0;
    std::uint64_t positionsOutside = 0;
    std::uint64_t editDistancesInGap = 0;
    std::uint64_t editDistancesOutside = 0;
    std::uint64_t positionsInReads = 0;
    std::uint64_t editsInReads = 0;

    void addFullRead(int numPositions, int edits){
        positionsInReads += numPositions;
        editsInReads += edits;
    }

    void addGap(int numPositions, int editDistance){
        positionsInGap += numPositions;
        editDistancesInGap += editDistance;
        numPairs++;
    }

    void addOutside(int numPositions, int editDistance){
        positionsOutside += numPositions;
        editDistancesOutside += editDistance;
    }

    std::uint64_t totalPositions() const{
        return positionsInGap + positionsOutside;
    }

    std::uint64_t totalEdits() const{
        return editDistancesInGap + editDistancesOutside;
    }

    float ratioInGap() const{
        return double(editDistancesInGap) / double(positionsInGap);
    }

    float ratioOutside() const{
        return double(editDistancesOutside) / double(positionsOutside);
    }

    float ratioTotal() const{
        return double(totalEdits()) / double(totalPositions());
    }

    float ratioFullReads() const{
        return double(editsInReads) / double(positionsInReads);
    }
};

std::ostream& operator<<(std::ostream& os, const EditDistanceStatistics& stat){
    constexpr char tab = '\t';
    constexpr char nl = '\n';

    os << "{" << nl;
    os << tab << "numPairs: " << stat.numPairs << ", " << nl;
    os << tab << "positionsInGap: " << stat.positionsInGap << ", " << nl;
    os << tab << "editDistancesInGap: " << stat.editDistancesInGap << ", "<< nl;
    os << tab << "ratioInGap: " << stat.ratioInGap() << ", "<< nl;
    os << tab << "positionsOutside: " << stat.positionsOutside << ", "<< nl;
    os << tab << "editDistancesOutside: " << stat.editDistancesOutside << ", "<< nl;
    os << tab << "ratioOutside: " << stat.ratioOutside() << ", "<< nl;
    os << tab << "positionsTotal: " << stat.totalPositions() << ", "<< nl;
    os << tab << "editDistancesTotal: " << stat.totalEdits() << ", "<< nl;
    os << tab << "ratioTotal: " << stat.ratioTotal() << nl;
    os << tab << "positionsInReads: " << stat.positionsInReads << ", "<< nl;
    os << tab << "editsInReads: " << stat.editsInReads << ", "<< nl;
    os << tab << "ratioFullReads: " << stat.ratioFullReads() << nl;
    os << "}";
    return os;
}

EditDistanceStatistics operator+(const EditDistanceStatistics& l, const EditDistanceStatistics& r){
    EditDistanceStatistics result;
    result.positionsInGap = l.positionsInGap + r.positionsInGap;
    result.positionsOutside = l.positionsOutside + r.positionsOutside;
    result.editDistancesInGap = l.editDistancesInGap + r.editDistancesInGap;
    result.editDistancesOutside = l.editDistancesOutside + r.editDistancesOutside;
    result.positionsInReads = l.positionsInReads + r.positionsInReads;
    result.editsInReads = l.editsInReads + r.editsInReads;

    return result;
}





struct Batch{
    int numElements = 0;
    int maxNumElements = 0;
    std::vector<std::int64_t> readIds;
    std::vector<Read> reads;
    std::vector<std::string> genomeregions;

    int numReadPairs = 0;
    int numNotExtended = 0;
    int numClipped = 0;
    int numReachedMate = 0;
    int numShrinkedPairs = 0;
    std::map<int, std::pair<int,int>> hammingMapGap;
    std::map<int, std::pair<int,int>> hammingMapAfter;
    std::map<int, std::pair<int,int>> hammingMapTotal;

    std::map<int, std::pair<int,int>> gapsizeDifferenceMap;

    std::map<int, std::pair<int,int>> editsMapFull;

    static constexpr int indexReachedWithoutRepeat = 0;
    static constexpr int indexReachedWithRepeat = 1;
    static constexpr int indexNotReached = 2;
    static constexpr int indexReachedWithoutRepeatNotMerged = 3;
    static constexpr int indexReachedWithoutRepeatMerged = 4;
    static constexpr int indexReachedWithRepeatNotMerged = 5;
    static constexpr int indexReachedWithRepeatMerged = 6;
    static constexpr int numBins = 7;

    std::array<EditDistanceStatistics, numBins> editDistanceStatistics{};
    std::array<LengthStatistics, numBins> lengthStatistics{};


    Batch(int size){
        resize(size);
    }

    void resize(int num){
        readIds.resize(num);
        reads.resize(num);
        genomeregions.resize(num);

        maxNumElements = num;
    }

    void resetElements(){
        numElements = 0;
    }

    void resetBatch(){

    }

    void addElement(std::int64_t readId, Read&& read, std::string&& genomeregion){
        assert(numElements < maxNumElements);

        readIds[numElements] = readId;
        reads[numElements] = std::move(read);
        genomeregions[numElements] = std::move(genomeregion);
        numElements++;

        numReadPairs++;
    }

    bool isFull() const{
        return numElements == maxNumElements;
    }

    void process(int readLengthBeforeExtension, Genome& genome){

        int newNumElements = 0;
        for(int i = 0; i < numElements; i++){
            const int length = reads[i].sequence.size();
            if(length == readLengthBeforeExtension){
                lengthStatistics[indexNotReached].observedLength(length);
                numNotExtended++;
            }else{
                //avoid self-move
                if(i != newNumElements){
                    readIds[newNumElements] = readIds[i];
                    reads[newNumElements] = std::move(reads[i]);
                    genomeregions[newNumElements] = std::move(genomeregions[i]);
                }
                newNumElements++;
            }
        }

        numElements = newNumElements;

        #pragma omp parallel for reduction(+: numReachedMate) reduction(+: numClipped) reduction(+: numShrinkedPairs)
        for(int i = 0; i < numElements; i++){
            auto& read = reads[i];
            auto& genomeregionHeader = genomeregions[i];

            try{

            auto tokens = split(read.comment, ' ');

            std::map<std::string, std::string> infomap;

            for(const auto& s : tokens){
                auto semitokens = split(s, ':');
                if(semitokens.size() == 2){
                    infomap.emplace(std::make_pair(std::move(semitokens[0]), std::move(semitokens[1])));
                }
            }

            bool reachedMate = false;
            auto reachediter = infomap.find("reached");
            if(reachediter != infomap.end()){
                try{
                    reachedMate = std::stoi(reachediter->second);
                }catch(std::invalid_argument&){
                    
                }
            }

            if(reachedMate){
                numReachedMate += 1;
            }

            int foundAfterRepetition = 0;
            auto foundAfterRepetitioniter = infomap.find("a");
            if(foundAfterRepetitioniter != infomap.end()){
                try{
                    foundAfterRepetition = std::stoi(foundAfterRepetitioniter->second);
                }catch(std::invalid_argument&){
                    
                }
            }

            int mergedTwoStrands = 0;
            auto mergedTwoStrandsiter = infomap.find("m");
            if(mergedTwoStrandsiter != infomap.end()){
                try{
                    mergedTwoStrands = std::stoi(mergedTwoStrandsiter->second);
                }catch(std::invalid_argument&){
                    
                }
            }

            const auto& extendedRead = read.sequence;
            const int extendedReadLength = extendedRead.length();
            std::string_view extendedReadView(extendedRead);

            assert(extendedReadLength >= readLengthBeforeExtension);

            #pragma omp critical
            {
                const int statisticsIndex = (reachedMate ? (foundAfterRepetition ? indexReachedWithRepeat : indexReachedWithoutRepeat) : indexNotReached);
                lengthStatistics[statisticsIndex].observedLength(extendedReadLength);
            }

            //in extended read, the first read of the read pair spans positions [initialBoundaries[0], initialBoundaries[1]),
            //the second read spans positions [initialBoundaries[2], initialBoundaries[3]),

            auto boundaryiter = infomap.find("lens");

            std::array<int, 4> initialBoundaries;

            if(boundaryiter != infomap.end()){

                auto tok = split(boundaryiter->second, ',');
                assert(tok.size() == 4);
                
                try{
                    initialBoundaries[0] = std::stoi(tok[0]);
                    initialBoundaries[1] = std::stoi(tok[1]);
                    initialBoundaries[2] = std::stoi(tok[2]);
                    initialBoundaries[3] = std::stoi(tok[3]);
                }catch(std::invalid_argument&){
                    std::cerr << "invalid tok " << boundaryiter->second << "\n";
                    assert(false);
                }

            }else{
                initialBoundaries[0] = 0;
                initialBoundaries[1] = readLengthBeforeExtension;
                initialBoundaries[2] = extendedReadLength - readLengthBeforeExtension;
                initialBoundaries[3] = extendedReadLength;
            }

            auto boundaries = initialBoundaries;

            assert(std::all_of(boundaries.begin(), boundaries.begin() + 2, [](int b){return b >= 0;}));
            assert(std::all_of(boundaries.begin() + 2, boundaries.end(), [](int b){return b == -1 || b >= 0;}));

            for(int i = 2; i < 4; i++){
                if(boundaries[i] == -1){
                    boundaries[i] = (i < 2) ? 0 : extendedReadLength;
                }
            }
            assert(boundaries[0] <= boundaries[1]);
            assert(boundaries[2] <= boundaries[3]);

            if(extendedReadView.size() < (boundaries[1] + (extendedReadView.size() - boundaries[2]))){
                numShrinkedPairs += 1;
                continue;
            }

            const int read1length = initialBoundaries[1] - initialBoundaries[0];
            const int read2length = initialBoundaries[3] - initialBoundaries[2];

            const std::string genomeRegionName = genomeregionHeader.substr(
                1,
                genomeregionHeader.find_first_of(':')-1
            );

            const auto regionRangeStrings = split(split(genomeregionHeader, ':')[1], '-');
            std::int64_t origregionRangeBegin = std::stoll(regionRangeStrings[0]);
            std::int64_t origregionRangeEnd_excl = std::stoll(regionRangeStrings[1]);
            const int origRegionLength = origregionRangeEnd_excl - origregionRangeBegin;
            auto mapiter = genome.data.find(genomeRegionName);
            if(mapiter == genome.data.end()){
                #pragma omp critical
                {
                    std::cerr << "Read header: " << read.name << " " << read.comment << "\n";
                    std::cerr << "could not find genomeRegionName " << genomeRegionName << "\n";
                    std::cerr << "valid names:\n";
                    for(const auto& p : genome.data){
                        std::cerr << p.first << "\n";
                    }
                }
            }
            assert(mapiter != genome.data.end());
            std::string_view origGenomeRegion(mapiter->second.data() + origregionRangeBegin, origRegionLength);
            const int origGapSize = (origregionRangeEnd_excl - origregionRangeBegin - readLengthBeforeExtension - readLengthBeforeExtension);

            //find out orientation of extension with respect to genome region

            int hammingfwd = std::numeric_limits<int>::max();    
            int hammingrevc = std::numeric_limits<int>::max();  

            #if 1

            {
                assert(read1length >= 42);
                std::string_view firstReadView(extendedRead.data() + initialBoundaries[0], 42);

                hammingfwd = editDistance(
                    firstReadView,
                    std::string_view{origGenomeRegion.data(), 42}
                );

                if(hammingfwd > 0){ 
                    //align reverse complement of first read to end of genome range

                    std::string firstReadRevc 
                        = reverseComplementString(extendedRead.data() + initialBoundaries[0], 
                            initialBoundaries[1] - initialBoundaries[0]);

                    std::string_view dfgdff(firstReadRevc.data() + firstReadRevc.size() - 42, 42);

                    hammingrevc = editDistance(
                        dfgdff,
                        std::string_view{origGenomeRegion.data() + origRegionLength - 42, 42}
                    );
                }
            }  

            
            #else

            {

                std::string_view firstReadView(extendedRead.data() + initialBoundaries[0], 42);     

                hammingfwd = editDistance(
                    firstReadView,
                    std::string_view{origGenomeRegion.data(), 42}
                );

                hammingrevc = std::numeric_limits<int>::max();        
                
                if(hammingfwd > 0){          
                    std::string secondReadRevc 
                        = reverseComplementString(extendedRead.data() + initialBoundaries[2], 
                            initialBoundaries[3] - initialBoundaries[2]);

                    std::string_view dfgdff(secondReadRevc.data(), 42);

                    hammingrevc = editDistance(
                        dfgdff,
                        std::string_view{origGenomeRegion.data(), 42}
                    );
                }                

            }

            #endif

            std::optional<std::string> genomeRegionString;
            
            std::string_view genomeRegion;
            std::int64_t regionRangeBegin = -42;
            std::int64_t regionRangeEnd_excl = -42;
            bool clipped = false;
            std::array<int, 4> readBoundariesInGenomeRegion;

            if(hammingfwd <= hammingrevc){
                //read position initialBoundaries[0] maps to position 0 in orig genome range

                const int extensionLengthLeftOfFirstRead = initialBoundaries[0];
                const int extensionLengthRightOfSecondRead = extendedReadLength - initialBoundaries[3];

                //extend orig region to the left
                regionRangeBegin = origregionRangeBegin - extensionLengthLeftOfFirstRead;
                //extend orig region to the right
                regionRangeEnd_excl = origregionRangeEnd_excl + extensionLengthRightOfSecondRead;

                readBoundariesInGenomeRegion[0] = extensionLengthLeftOfFirstRead;
                readBoundariesInGenomeRegion[1] = readBoundariesInGenomeRegion[0] + read1length;
                readBoundariesInGenomeRegion[2] = readBoundariesInGenomeRegion[1] + origGapSize;
                readBoundariesInGenomeRegion[3] = readBoundariesInGenomeRegion[2] + read2length;

                //check out of bounds, and clip

                if(regionRangeBegin < 0){
                    const int difference = -regionRangeBegin;
                    extendedReadView.remove_prefix(difference);
                    regionRangeBegin = 0;
                    for(auto& b : boundaries){
                        b -= difference;
                    }
                    clipped = true;
                }

                if(std::size_t(regionRangeEnd_excl) > genome.data[genomeRegionName].size()){
                    const int difference = std::size_t(regionRangeEnd_excl) - genome.data[genomeRegionName].size();
                    extendedReadView.remove_suffix(difference);
                    regionRangeEnd_excl = genome.data[genomeRegionName].size();
                    for(auto& b : boundaries){
                        if(int(extendedReadView.size()) < b){
                            b -= difference;
                        }
                    }
                    clipped = true;
                }

                if(!clipped){

                    const int genomeRegionLength = regionRangeEnd_excl - regionRangeBegin;

                    genomeRegion = std::string_view(genome.data[genomeRegionName].data() + regionRangeBegin, genomeRegionLength);
                }

            }else{
                //read position initialBoundaries[3]-1 maps to position 0 in orig genome range, and decreasing positions in extended read correspond to increasing positions in genome range

                // //extend orig region to the left. this corresponds to the part between end of second read, and end of extendedread
                // regionRangeBegin -= extendedReadView.size() - initialBoundaries[3];
                // //extend orig region to the right
                // regionRangeEnd_excl = regionRangeBegin + extendedReadView.size();

                const int extensionLengthLeftOfFirstRead = initialBoundaries[0];
                const int extensionLengthRightOfSecondRead = extendedReadLength - initialBoundaries[3];

                regionRangeBegin = origregionRangeBegin - extensionLengthRightOfSecondRead;
                regionRangeEnd_excl = origregionRangeEnd_excl + extensionLengthLeftOfFirstRead;

                readBoundariesInGenomeRegion[0] = extensionLengthLeftOfFirstRead;
                readBoundariesInGenomeRegion[1] = readBoundariesInGenomeRegion[0] + read1length;
                readBoundariesInGenomeRegion[2] = readBoundariesInGenomeRegion[1] + origGapSize;
                readBoundariesInGenomeRegion[3] = readBoundariesInGenomeRegion[2] + read2length;

                if(regionRangeBegin < 0){
                    const int difference = -regionRangeBegin;
                    //remove from the right end of extended read
                    extendedReadView.remove_suffix(difference);
                    regionRangeBegin = 0;
                    for(auto& b : boundaries){
                        if(int(extendedReadView.size()) < b){
                            b -= difference;
                        }
                    }
                    clipped = true;
                }

                if(std::size_t(regionRangeEnd_excl) > genome.data[genomeRegionName].size()){
                    const int difference = std::size_t(regionRangeEnd_excl) - genome.data[genomeRegionName].size();
                    //remove from the left end of extended read
                    extendedReadView.remove_prefix(difference);
                    regionRangeEnd_excl = genome.data[genomeRegionName].size();
                    for(auto& b : boundaries){
                        b -= difference;
                    }
                    clipped = true;
                }

                if(!clipped){
                    const int genomeRegionLength = regionRangeEnd_excl - regionRangeBegin;

                    std::string regionReverseComplement = reverseComplementString(genome.data[genomeRegionName].data() + regionRangeBegin, genomeRegionLength);
                    genomeRegionString = std::move(regionReverseComplement);

                    genomeRegion = std::string_view(genomeRegionString->data(), genomeRegionLength);
                }
            }

            if(clipped){
                numClipped++;
            }else{
                const auto& region = genomeRegion;

                if(!reachedMate){
                    //check parts of filled gap, and extension left of first read. (second read does not exist)
                    
                    std::string_view filledgapOfExtendedRead(extendedRead);
                    filledgapOfExtendedRead.remove_prefix(boundaries[1]);

                    std::string_view filledGapOfGenomeRegion(region);
                    filledGapOfGenomeRegion.remove_prefix(readBoundariesInGenomeRegion[1]);       
                    filledGapOfGenomeRegion.remove_suffix(filledGapOfGenomeRegion.size() - filledgapOfExtendedRead.size());                

                    const int edFilled = editDistance(filledgapOfExtendedRead, filledGapOfGenomeRegion);                    

                    std::string_view extensionleftRead(extendedReadView.data(), boundaries[0]);
                    std::string_view extensionleftGenome(region.data(), readBoundariesInGenomeRegion[0]);

                    const int edLeft = editDistance(extensionleftRead, extensionleftGenome);

                    const int ed = edFilled + edLeft;

                    // const int matchesFull = semiglobalmatches(extendedRead, region);
                    // const int editsInFullRead = extendedRead.size() - matchesFull;

                    #pragma omp critical
                    {

                        //only extended reads with extension outside of original region contribute.
                        if(extensionleftRead.size() > 0){
                            const int edBorder = edLeft;
                            hammingMapAfter[edBorder].first++;                      
                        }

                        hammingMapTotal[ed].first++;
                        hammingMapGap[edFilled].first++;

                        //editsMapFull[editsInFullRead].first++;

                        //editDistanceStatistics[indexNotReached].addFullRead(extendedRead.size(), editsInFullRead);
                        editDistanceStatistics[indexNotReached].addGap(filledgapOfExtendedRead.size(), edFilled);
                        editDistanceStatistics[indexNotReached].addOutside(extensionleftRead.size(), edLeft);
                    }
                }else{

                    std::string_view filledgapOfExtendedRead(extendedReadView);
                    filledgapOfExtendedRead.remove_prefix(boundaries[1]);
                    filledgapOfExtendedRead.remove_suffix(extendedReadView.size() - boundaries[2]);

                    std::string_view filledGapOfGenomeRegion(region);
                    filledGapOfGenomeRegion.remove_prefix(readBoundariesInGenomeRegion[1]);
                    filledGapOfGenomeRegion.remove_suffix(region.size() - readBoundariesInGenomeRegion[2]);

                    const int edFilled = editDistance(filledgapOfExtendedRead, filledGapOfGenomeRegion);

                    //check extension outside of original genome region

                    std::string_view extensionleftRead(extendedReadView.data(), boundaries[0]);
                    std::string_view extensionleftGenome(region.data(), readBoundariesInGenomeRegion[0]);

                    std::string_view extensionrightRead(extendedReadView.data() + boundaries[3], extendedReadView.size() - boundaries[3]);
                    std::string_view extensionrightGenome(region.data() + readBoundariesInGenomeRegion[3], region.size() - readBoundariesInGenomeRegion[3]);

                    const int edLeft = editDistance(extensionleftRead, extensionleftGenome);
                    const int edRight = editDistance(extensionrightRead, extensionrightGenome);

                    const int ed = edFilled + edLeft + edRight;

                    #pragma omp critical
                    {

                        //only extended reads with extension outside of original region contribute.
                        if(extensionleftRead.size() > 0 || extensionrightRead.size() > 0){

                            const int edBorder = edLeft + edRight;
                            hammingMapAfter[edBorder].second++;                              
                        }                    
                        
                        hammingMapTotal[ed].second++;
                        hammingMapGap[edFilled].second++;

                        //editsMapFull[editsInFullRead].second++;

                        const int statisticsIndex = foundAfterRepetition ? indexReachedWithRepeat : indexReachedWithoutRepeat;

                        //editDistanceStatistics[statisticsIndex].addFullRead(extendedRead.size(), editsInFullRead);
                        editDistanceStatistics[statisticsIndex].addGap(filledgapOfExtendedRead.size(), edFilled);
                        editDistanceStatistics[statisticsIndex].addOutside(extensionleftRead.size(), edLeft);
                        editDistanceStatistics[statisticsIndex].addOutside(extensionrightRead.size(), edRight);

                        int statisticsIndexFineGrained = 0;
                        if(!foundAfterRepetition){
                            if(!mergedTwoStrands){
                                statisticsIndexFineGrained = indexReachedWithoutRepeatNotMerged;
                            }else{
                                statisticsIndexFineGrained = indexReachedWithoutRepeatMerged;
                            }
                        }else{
                            if(!mergedTwoStrands){
                                statisticsIndexFineGrained = indexReachedWithRepeatNotMerged;
                            }else{
                                statisticsIndexFineGrained = indexReachedWithRepeatMerged;
                            }
                        }

                        editDistanceStatistics[statisticsIndexFineGrained].addGap(filledgapOfExtendedRead.size(), edFilled);
                        editDistanceStatistics[statisticsIndexFineGrained].addOutside(extensionleftRead.size(), edLeft);
                        editDistanceStatistics[statisticsIndexFineGrained].addOutside(extensionrightRead.size(), edRight);

                        const int gapsizeDifference = filledGapOfGenomeRegion.size() - filledgapOfExtendedRead.size();
                        gapsizeDifferenceMap[gapsizeDifference].second++;

                    }
                }
            }
            } catch (...){
                 #pragma omp critical
                 {
                     
                 std::cerr << "Caught exception. Current read:\n";
                 std::cerr << i << " " << readIds[i] << "\n";
                 std::cerr << read.name << " " << read.comment << "\n" << read.sequence << "\n";
                    auto eptr = std::current_exception();
                    try {
                        if (eptr) {
                            std::rethrow_exception(eptr);
                        }
                    } catch(const std::exception& e) {
                        std::cerr << "Caught exception \"" << e.what() << "\"\n";
                    }
                 }
            }
        }
    }

    #ifdef __CUDACC__

    void process_gpu(int readLengthBeforeExtension, Genome& genome){

        int newNumElements = 0;
        for(int i = 0; i < numElements; i++){
            const int length = reads[i].sequence.size();
            if(length == readLengthBeforeExtension){
                lengthStatistics[indexNotReached].observedLength(length);
                numNotExtended++;
            }else{
                //avoid self-move
                if(i != newNumElements){
                    readIds[newNumElements] = readIds[i];
                    reads[newNumElements] = std::move(reads[i]);
                    genomeregions[newNumElements] = std::move(genomeregions[i]);
                }
                newNumElements++;
            }
        }

        numElements = newNumElements;

        const int numThreads = omp_get_max_threads();

        std::vector<std::vector<bool>> vecvecReached(numThreads);
        std::vector<std::vector<bool>> vecvecFoundAfterRepetition(numThreads);
        std::vector<std::vector<std::string>> vecvecS1(numThreads);
        std::vector<std::vector<std::string>> vecvecS2(numThreads);

        nvtx::push_range("make inputs", 0);

        #pragma omp parallel for reduction(+: numReachedMate) reduction(+: numClipped) reduction(+: numShrinkedPairs)
        for(int i = 0; i < numElements; i++){
            auto& read = reads[i];
            auto& genomeregionHeader = genomeregions[i];

            try{

            auto tokens = split(read.comment, ' ');

            std::map<std::string, std::string> infomap;

            for(const auto& s : tokens){
                auto semitokens = split(s, ':');
                if(semitokens.size() == 2){
                    infomap.emplace(std::make_pair(std::move(semitokens[0]), std::move(semitokens[1])));
                }
            }

            bool reachedMate = false;
            auto reachediter = infomap.find("reached");
            if(reachediter != infomap.end()){
                try{
                    reachedMate = std::stoi(reachediter->second);
                }catch(std::invalid_argument&){
                    
                }
            }

            if(reachedMate){
                numReachedMate += 1;
            }

            int foundAfterRepetition = 0;
            auto foundAfterRepetitioniter = infomap.find("a");
            if(foundAfterRepetitioniter != infomap.end()){
                try{
                    foundAfterRepetition = std::stoi(foundAfterRepetitioniter->second);
                }catch(std::invalid_argument&){
                    
                }
            }

            const auto& extendedRead = read.sequence;
            const int extendedReadLength = extendedRead.length();
            std::string_view extendedReadView(extendedRead);

            assert(extendedReadLength >= readLengthBeforeExtension);

            #pragma omp critical
            {
                const int statisticsIndex = (reachedMate ? (foundAfterRepetition ? indexReachedWithRepeat : indexReachedWithoutRepeat) : indexNotReached);
                lengthStatistics[statisticsIndex].observedLength(extendedReadLength);
            }

            //in extended read, the first read of the read pair spans positions [initialBoundaries[0], initialBoundaries[1]),
            //the second read spans positions [initialBoundaries[2], initialBoundaries[3]),

            auto boundaryiter = infomap.find("lens");

            std::array<int, 4> initialBoundaries;

            if(boundaryiter != infomap.end()){

                auto tok = split(boundaryiter->second, ',');
                assert(tok.size() == 4);
                
                try{
                    initialBoundaries[0] = std::stoi(tok[0]);
                    initialBoundaries[1] = std::stoi(tok[1]);
                    initialBoundaries[2] = std::stoi(tok[2]);
                    initialBoundaries[3] = std::stoi(tok[3]);
                }catch(std::invalid_argument&){
                    std::cerr << "invalid tok " << boundaryiter->second << "\n";
                    assert(false);
                }

            }else{
                initialBoundaries[0] = 0;
                initialBoundaries[1] = readLengthBeforeExtension;
                initialBoundaries[2] = extendedReadLength - readLengthBeforeExtension;
                initialBoundaries[3] = extendedReadLength;
            }

            auto boundaries = initialBoundaries;

            assert(std::all_of(boundaries.begin(), boundaries.begin() + 2, [](int b){return b >= 0;}));
            assert(std::all_of(boundaries.begin() + 2, boundaries.end(), [](int b){return b == -1 || b >= 0;}));

            for(int i = 2; i < 4; i++){
                if(boundaries[i] == -1){
                    boundaries[i] = (i < 2) ? 0 : extendedReadLength;
                }
            }
            assert(boundaries[0] <= boundaries[1]);
            assert(boundaries[2] <= boundaries[3]);

            const int read1length = initialBoundaries[1] - initialBoundaries[0];
            const int read2length = initialBoundaries[3] - initialBoundaries[2];

            const std::string genomeRegionName = genomeregionHeader.substr(
                1,
                genomeregionHeader.find_first_of(':')-1
            );

            const auto regionRangeStrings = split(split(genomeregionHeader, ':')[1], '-');
            std::int64_t origregionRangeBegin = std::stoll(regionRangeStrings[0]);
            std::int64_t origregionRangeEnd_excl = std::stoll(regionRangeStrings[1]);
            const int origRegionLength = origregionRangeEnd_excl - origregionRangeBegin;
            auto mapiter = genome.data.find(genomeRegionName);
            if(mapiter == genome.data.end()){
                #pragma omp critical
                {
                    std::cerr << "Read header: " << read.name << " " << read.comment << "\n";
                    std::cerr << "could not find genomeRegionName " << genomeRegionName << "\n";
                    std::cerr << "valid names:\n";
                    for(const auto& p : genome.data){
                        std::cerr << p.first << "\n";
                    }
                }
            }
            assert(mapiter != genome.data.end());
            std::string_view origGenomeRegion(mapiter->second.data() + origregionRangeBegin, origRegionLength);
            const int origGapSize = (origregionRangeEnd_excl - origregionRangeBegin - readLengthBeforeExtension - readLengthBeforeExtension);

            //find out orientation of extension with respect to genome region

            int hammingfwd = std::numeric_limits<int>::max();    
            int hammingrevc = std::numeric_limits<int>::max();  

            #if 1

            {
                assert(read1length >= 42);
                std::string_view firstReadView(extendedRead.data() + initialBoundaries[0], 42);

                hammingfwd = editDistance(
                    firstReadView,
                    std::string_view{origGenomeRegion.data(), 42}
                );

                if(hammingfwd > 0){ 
                    //align reverse complement of first read to end of genome range

                    std::string firstReadRevc 
                        = reverseComplementString(extendedRead.data() + initialBoundaries[0], 
                            initialBoundaries[1] - initialBoundaries[0]);

                    std::string_view dfgdff(firstReadRevc.data() + firstReadRevc.size() - 42, 42);

                    hammingrevc = editDistance(
                        dfgdff,
                        std::string_view{origGenomeRegion.data() + origRegionLength - 42, 42}
                    );
                }
            }  

            
            #else

            {

                std::string_view firstReadView(extendedRead.data() + initialBoundaries[0], 42);     

                hammingfwd = editDistance(
                    firstReadView,
                    std::string_view{origGenomeRegion.data(), 42}
                );

                hammingrevc = std::numeric_limits<int>::max();        
                
                if(hammingfwd > 0){          
                    std::string secondReadRevc 
                        = reverseComplementString(extendedRead.data() + initialBoundaries[2], 
                            initialBoundaries[3] - initialBoundaries[2]);

                    std::string_view dfgdff(secondReadRevc.data(), 42);

                    hammingrevc = editDistance(
                        dfgdff,
                        std::string_view{origGenomeRegion.data(), 42}
                    );
                }                

            }

            #endif

            std::optional<std::string> genomeRegionString;
            
            std::string_view genomeRegion;
            std::int64_t regionRangeBegin = -42;
            std::int64_t regionRangeEnd_excl = -42;
            bool clipped = false;
            std::array<int, 4> readBoundariesInGenomeRegion;

            if(hammingfwd <= hammingrevc){
                //read position initialBoundaries[0] maps to position 0 in orig genome range

                const int extensionLengthLeftOfFirstRead = initialBoundaries[0];
                const int extensionLengthRightOfSecondRead = extendedReadLength - initialBoundaries[3];

                //extend orig region to the left
                regionRangeBegin = origregionRangeBegin - extensionLengthLeftOfFirstRead;
                //extend orig region to the right
                regionRangeEnd_excl = origregionRangeEnd_excl + extensionLengthRightOfSecondRead;

                readBoundariesInGenomeRegion[0] = extensionLengthLeftOfFirstRead;
                readBoundariesInGenomeRegion[1] = readBoundariesInGenomeRegion[0] + read1length;
                readBoundariesInGenomeRegion[2] = readBoundariesInGenomeRegion[1] + origGapSize;
                readBoundariesInGenomeRegion[3] = readBoundariesInGenomeRegion[2] + read2length;

                //check out of bounds, and clip

                if(regionRangeBegin < 0){
                    const int difference = -regionRangeBegin;
                    extendedReadView.remove_prefix(difference);
                    regionRangeBegin = 0;
                    for(auto& b : boundaries){
                        b -= difference;
                    }
                    clipped = true;
                }

                if(std::size_t(regionRangeEnd_excl) > genome.data[genomeRegionName].size()){
                    const int difference = std::size_t(regionRangeEnd_excl) - genome.data[genomeRegionName].size();
                    extendedReadView.remove_suffix(difference);
                    regionRangeEnd_excl = genome.data[genomeRegionName].size();
                    for(auto& b : boundaries){
                        if(int(extendedReadView.size()) < b){
                            b -= difference;
                        }
                    }
                    clipped = true;
                }

                if(!clipped){

                    const int genomeRegionLength = regionRangeEnd_excl - regionRangeBegin;

                    genomeRegion = std::string_view(genome.data[genomeRegionName].data() + regionRangeBegin, genomeRegionLength);
                }

            }else{
                //read position initialBoundaries[3]-1 maps to position 0 in orig genome range, and decreasing positions in extended read correspond to increasing positions in genome range

                // //extend orig region to the left. this corresponds to the part between end of second read, and end of extendedread
                // regionRangeBegin -= extendedReadView.size() - initialBoundaries[3];
                // //extend orig region to the right
                // regionRangeEnd_excl = regionRangeBegin + extendedReadView.size();

                const int extensionLengthLeftOfFirstRead = initialBoundaries[0];
                const int extensionLengthRightOfSecondRead = extendedReadLength - initialBoundaries[3];

                regionRangeBegin = origregionRangeBegin - extensionLengthRightOfSecondRead;
                regionRangeEnd_excl = origregionRangeEnd_excl + extensionLengthLeftOfFirstRead;

                readBoundariesInGenomeRegion[0] = extensionLengthLeftOfFirstRead;
                readBoundariesInGenomeRegion[1] = readBoundariesInGenomeRegion[0] + read1length;
                readBoundariesInGenomeRegion[2] = readBoundariesInGenomeRegion[1] + origGapSize;
                readBoundariesInGenomeRegion[3] = readBoundariesInGenomeRegion[2] + read2length;

                if(regionRangeBegin < 0){
                    const int difference = -regionRangeBegin;
                    //remove from the right end of extended read
                    extendedReadView.remove_suffix(difference);
                    regionRangeBegin = 0;
                    for(auto& b : boundaries){
                        if(int(extendedReadView.size()) < b){
                            b -= difference;
                        }
                    }
                    clipped = true;
                }

                if(std::size_t(regionRangeEnd_excl) > genome.data[genomeRegionName].size()){
                    const int difference = std::size_t(regionRangeEnd_excl) - genome.data[genomeRegionName].size();
                    //remove from the left end of extended read
                    extendedReadView.remove_prefix(difference);
                    regionRangeEnd_excl = genome.data[genomeRegionName].size();
                    for(auto& b : boundaries){
                        b -= difference;
                    }
                    clipped = true;
                }

                if(!clipped){
                    const int genomeRegionLength = regionRangeEnd_excl - regionRangeBegin;

                    std::string regionReverseComplement = reverseComplementString(genome.data[genomeRegionName].data() + regionRangeBegin, genomeRegionLength);
                    genomeRegionString = std::move(regionReverseComplement);

                    genomeRegion = std::string_view(genomeRegionString->data(), genomeRegionLength);
                }
            }

            if(clipped){
                numClipped++;
            }else if(extendedReadView.size() < (boundaries[1] + (extendedReadView.size() - boundaries[2]))){
                numShrinkedPairs += 1;
            }else{
                const auto& region = genomeRegion;

                const int tid = omp_get_thread_num();

                std::vector<bool>& vecReached = vecvecReached[tid];
                std::vector<bool>& vecFoundAfterRepetition = vecvecFoundAfterRepetition[tid];
                std::vector<std::string>& vecS1 = vecvecS1[tid];
                std::vector<std::string>& vecS2 = vecvecS2[tid];

                vecReached.push_back(reachedMate);
                vecFoundAfterRepetition.push_back(foundAfterRepetition);

                if(!reachedMate){
                    //check parts of filled gap, and extension left of first read. (second read does not exist)
                    
                    std::string_view filledgapOfExtendedRead(extendedRead);
                    filledgapOfExtendedRead.remove_prefix(boundaries[1]);

                    std::string_view filledGapOfGenomeRegion(region);
                    filledGapOfGenomeRegion.remove_prefix(readBoundariesInGenomeRegion[1]);       
                    filledGapOfGenomeRegion.remove_suffix(filledGapOfGenomeRegion.size() - filledgapOfExtendedRead.size());

                    vecS1.emplace_back(filledgapOfExtendedRead);
                    vecS2.emplace_back(filledGapOfGenomeRegion);                 

                    std::string_view extensionleftRead(extendedReadView.data(), boundaries[0]);
                    std::string_view extensionleftGenome(region.data(), readBoundariesInGenomeRegion[0]);

                    vecS1.emplace_back(extensionleftRead);
                    vecS2.emplace_back(extensionleftGenome);

                }else{

                    std::string_view filledgapOfExtendedRead(extendedReadView);
                    filledgapOfExtendedRead.remove_prefix(boundaries[1]);
                    filledgapOfExtendedRead.remove_suffix(extendedReadView.size() - boundaries[2]);

                    std::string_view filledGapOfGenomeRegion(region);
                    filledGapOfGenomeRegion.remove_prefix(readBoundariesInGenomeRegion[1]);
                    filledGapOfGenomeRegion.remove_suffix(region.size() - readBoundariesInGenomeRegion[2]);

                    //const int edFilled = editDistance(filledgapOfExtendedRead, filledGapOfGenomeRegion);
                    vecS1.emplace_back(filledgapOfExtendedRead);
                    vecS2.emplace_back(filledGapOfGenomeRegion);

                    //check extension outside of original genome region

                    std::string_view extensionleftRead(extendedReadView.data(), boundaries[0]);
                    std::string_view extensionleftGenome(region.data(), readBoundariesInGenomeRegion[0]);

                    std::string_view extensionrightRead(extendedReadView.data() + boundaries[3], extendedReadView.size() - boundaries[3]);
                    std::string_view extensionrightGenome(region.data() + readBoundariesInGenomeRegion[3], region.size() - readBoundariesInGenomeRegion[3]);

                    vecS1.emplace_back(extensionleftRead);
                    vecS2.emplace_back(extensionleftGenome);

                    vecS1.emplace_back(extensionrightRead);
                    vecS2.emplace_back(extensionrightGenome);
                }
            }
            } catch (...){
                 #pragma omp critical
                 {
                 std::cerr << "Caught exception. Current read:\n";
                 std::cerr << i << " " << readIds[i] << "\n";
                 std::cerr << read.name << " " << read.comment << "\n" << read.sequence << "\n";
                 auto eptr = std::current_exception();
                    try {
                        if (eptr) {
                            std::rethrow_exception(eptr);
                        }
                    } catch(const std::exception& e) {
                        std::cerr << "Caught exception \"" << e.what() << "\"\n";
                    }
                 }
                 assert(false);
            }
        }

        nvtx::pop_range();

        nvtx::push_range("merge inputs", 1);

        std::vector<bool> vecReached;
        std::vector<bool> vecFoundAfterRepetition;
        std::vector<std::string> vecS1;
        std::vector<std::string> vecS2;

        for(auto&& vec : vecvecReached){
            vecReached.insert(vecReached.end(), vec.begin(), vec.end());
        }
        for(auto&& vec : vecvecFoundAfterRepetition){
            vecFoundAfterRepetition.insert(vecFoundAfterRepetition.end(), vec.begin(), vec.end());
        }
        for(auto&& vec : vecvecS1){
            vecS1.insert(vecS1.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
        }
        for(auto&& vec : vecvecS2){
            vecS2.insert(vecS2.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
        }

        nvtx::pop_range();

        auto gpuresults = editDistance_cuda(vecS1, vecS2);

        nvtx::push_range("process results", 2);

        std::size_t resultindex = 0;
        for(std::size_t s = 0; s < vecReached.size(); s++){
            if(!vecReached[s]){
                assert(resultindex <= gpuresults.size() - 2);
                const int edFilled = gpuresults[resultindex];
                const int edLeft = gpuresults[resultindex + 1];

                const int ed = edFilled + edLeft;

                    //only extended reads with extension outside of original region contribute.
                if(vecS1[resultindex + 1].size() > 0){
                    const int edBorder = edLeft;
                    hammingMapAfter[edBorder].first++;                      
                }

                hammingMapTotal[ed].first++;
                hammingMapGap[edFilled].first++;

                editDistanceStatistics[indexNotReached].addGap(vecS1[resultindex].size(), edFilled);
                editDistanceStatistics[indexNotReached].addOutside(vecS1[resultindex + 1].size(), edLeft);

                resultindex += 2;
            }else{
                if(!(resultindex <= gpuresults.size() - 3)){
                    std::cerr << "assertion will fail\n";
                }
                assert(resultindex <= gpuresults.size() - 3);

                const int edFilled = gpuresults[resultindex];
                const int edLeft = gpuresults[resultindex + 1];
                const int edRight = gpuresults[resultindex + 2];

                const int ed = edFilled + edLeft + edRight;

                //only extended reads with extension outside of original region contribute.
                if(vecS1[resultindex + 1].size() > 0 || vecS1[resultindex + 2].size() > 0){

                    const int edBorder = edLeft + edRight;
                    hammingMapAfter[edBorder].second++;                              
                }                    
                
                hammingMapTotal[ed].second++;
                hammingMapGap[edFilled].second++;

                //editsMapFull[editsInFullRead].second++;

                const int statisticsIndex = vecFoundAfterRepetition[s] ? indexReachedWithRepeat : indexReachedWithoutRepeat;

                //editDistanceStatistics[statisticsIndex].addFullRead(extendedRead.size(), editsInFullRead);
                editDistanceStatistics[statisticsIndex].addGap(vecS1[resultindex].size(), edFilled);
                editDistanceStatistics[statisticsIndex].addOutside(vecS1[resultindex + 1].size(), edLeft);
                editDistanceStatistics[statisticsIndex].addOutside(vecS1[resultindex + 2].size(), edRight);

                const int gapsizeDifference = vecS2[resultindex].size() - vecS1[resultindex].size();
                gapsizeDifferenceMap[gapsizeDifference].second++;

                resultindex += 3;
            }
        }
        nvtx::pop_range();
    }

    #endif


    void printReport() const{

        std::pair<std::int64_t, std::int64_t> remaining{};

        constexpr int printlimit = 20;

        std::cout << "Num read pairs: " << numReadPairs << "\n";
        std::cout << "Num reached mate: "  << numReachedMate << ", " << (double(numReachedMate) / double(numReadPairs)) * 100 << " % " << "\n";
        std::cout << "Num not extended: "  << numNotExtended << "\n";
        std::cout << "Num clipped: "  << numClipped << "\n";
        std::cout << "Num shrinked: " << numShrinkedPairs << "\n";
        std::cout << "Edit statistics, reached mate total: " << (editDistanceStatistics[indexReachedWithRepeat] + editDistanceStatistics[indexReachedWithoutRepeat]) << "\n";
        std::cout << "Edit statistics, reached mate without repeat: " << editDistanceStatistics[indexReachedWithoutRepeat] << "\n";
        std::cout << "Edit statistics, reached mate with repeat: " << editDistanceStatistics[indexReachedWithRepeat] << "\n";
        std::cout << "Edit statistics, not reached mate: " << editDistanceStatistics[indexNotReached] << "\n";
        std::cout << "Edit statistics, reached mate without repeat no merge: " << editDistanceStatistics[indexReachedWithoutRepeatNotMerged] << "\n";
        std::cout << "Edit statistics, reached mate without repeat with merge: " << editDistanceStatistics[indexReachedWithoutRepeatMerged] << "\n";
        std::cout << "Edit statistics, reached mate with repeat no merge: " << editDistanceStatistics[indexReachedWithRepeatNotMerged] << "\n";
        std::cout << "Edit statistics, reached mate with repeat with merge: " << editDistanceStatistics[indexReachedWithRepeatMerged] << "\n";

        std::cout << "Length of extended reads, reached mate total: " << (lengthStatistics[indexReachedWithoutRepeat] + lengthStatistics[indexReachedWithRepeat]) << "\n";
        std::cout << "Length of extended reads, reached without repeat: " << lengthStatistics[indexReachedWithoutRepeat] << "\n";
        std::cout << "Length of extended reads, reached with repeat: " << lengthStatistics[indexReachedWithRepeat] << "\n";
        std::cout << "Length of extended reads, not reached mate: " << lengthStatistics[indexNotReached] << "\n";

        std::cout << "Difference original gap size and filled gap size\n";

        std::pair<std::int64_t, std::int64_t> remainingless{0,0};
        std::pair<std::int64_t, std::int64_t> remaininggreater{0,0};
        
        for(const auto& x : gapsizeDifferenceMap){
            if(x.first < -8){
                remainingless.first += x.second.first;
                remainingless.second += x.second.second;
            }else if(x.first > 8){
                remaininggreater.first += x.second.first;
                remaininggreater.second += x.second.second;
            }else{
                std::cout << x.first << " : " << x.second.first << " - " << x.second.second << " (" << (x.second.first + x.second.second) << ")" <<"\n";
            }
        }
        std::cout << "< -8 : " << remainingless.first << " - " << remainingless.second << "\n";
        std::cout << "> 8 : " << remaininggreater.first << " - " << remaininggreater.second << "\n";
        std::cout << "\n\n";

        std::cout << "Edit distance in filled gap\n";

        remaining.first = 0;
        remaining.second = 0;

        for(const auto& x : hammingMapGap){
            if(x.first <= printlimit){
                std::cout << x.first << " : " << x.second.first << " - " << x.second.second << " (" << (x.second.first + x.second.second) << ")" <<"\n";
            }else{
                remaining.first += x.second.first;
                remaining.second += x.second.second;
            }
        }

        std::cout << "> " << printlimit <<" : " << remaining.first << " - " << remaining.second << "\n";


        std::cout << "\n\n";

        std::cout << "Edit distance in border extension\n";

        remaining.first = 0;
        remaining.second = 0;

        for(const auto& x : hammingMapAfter){
            if(x.first <= printlimit){
                std::cout << x.first << " : " << x.second.first << " - " << x.second.second << " (" << (x.second.first + x.second.second) << ")" <<"\n";
            }else{
                remaining.first += x.second.first;
                remaining.second += x.second.second;
            }
        }

        std::cout << "> " << printlimit <<" : " << remaining.first << " - " << remaining.second << "\n";


        std::cout << "\n\n";

        std::cout << "Edit distance in border extension + gap\n";

        remaining.first = 0;
        remaining.second = 0;

        for(const auto& x : hammingMapTotal){
            if(x.first <= printlimit){
                std::cout << x.first << " : " << x.second.first << " - " << x.second.second << " (" << (x.second.first + x.second.second) << ")" <<"\n";
            }else{
                remaining.first += x.second.first;
                remaining.second += x.second.second;
            }
        }

        std::cout << "> " << printlimit <<" : " << remaining.first << " - " << remaining.second << "\n";

        // std::cout << "\n\n";

        // std::cout << "fullEditDistanceMap\n";

        // remaining.first = 0;
        // remaining.second = 0;

        // for(const auto& x : fullEditDistanceMap){
        //     if(x.first <= printlimit){
        //         std::cout << x.first << " : " << x.second.first << " - " << x.second.second << " (" << (x.second.first + x.second.second) << ")" <<"\n";
        //     }else{
        //         remaining.first += x.second.first;
        //         remaining.second += x.second.second;
        //     }
        // }

        // std::cout << "> " << printlimit <<" : " << remaining.first << " - " << remaining.second << "\n";

        


        //std::cout << "\n\n";


        // std::cout << "Genome statistics for read which reached mate\n";
        // genomeHitsReached.printStatistics(); 

        // std::cout << "Genome statistics for read which did not reach mate\n";
        // genomeHitsNotReached.printStatistics(); 
    };
};


int main(int argc, char** argv){

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " readlength genome.fasta genomeregions.fasta extendedreads.txt\n";
        return 1;
    }

    const int readLengthBeforeExtension = std::atoi(argv[1]);

    std::string genomeFilename = argv[2];

    std::string genomeregionsFilename = argv[3]; // fasta file with one genome region per read pair in paired-end read file

    std::string extendedReadsFilename = argv[4]; 

    std::ifstream genomeregionsFile(genomeregionsFilename);

    assert(bool(genomeregionsFile));

    std::int64_t genomeRegionId = 0;

    kseqpp::KseqPP reader(extendedReadsFilename);
    Read read{};

    // GenomeHits genomeHitsReached(genomeFilename);
    // GenomeHits genomeHitsNotReached(genomeFilename);

    Genome genome = parseGenome(genomeFilename);

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

    constexpr int batchsize = 32000;
    Batch batch(batchsize);

    success = getNextRead();

    while(success){

        std::int64_t readId = readNumberInFile;
        try{
            readId = std::stoll(read.name);
        }catch(std::invalid_argument&){
            std::cerr << "error getting read id\n";
        }

        while(genomeRegionId < readId / 2){
            std::getline(genomeregionsFile, line); //fasta header

            assert(bool(genomeregionsFile));
            assert(line[0] == '>');

            std::getline(genomeregionsFile, line); //sequence
            assert(bool(genomeregionsFile));

            genomeRegionId++;
        }

        std::string genomeregionHeader;

        std::getline(genomeregionsFile, genomeregionHeader); //fasta header

        assert(bool(genomeregionsFile));
        assert(genomeregionHeader[0] == '>');

        std::getline(genomeregionsFile, line); //sequence
        assert(bool(genomeregionsFile));

        genomeRegionId++;
        readNumberInFile++;

        //file parsing is done

        batch.addElement(readId, std::move(read), str_toupper(genomeregionHeader));

        if(batch.isFull()){
            #ifdef __CUDACC__
            batch.process_gpu(readLengthBeforeExtension, genome);
            #else
            batch.process(readLengthBeforeExtension, genome);
            #endif
            batch.resetElements();
        }

        success = getNextRead(); //prepare next loop iteration
    }

    #ifdef __CUDACC__
    batch.process_gpu(readLengthBeforeExtension, genome);
    #else
    batch.process(readLengthBeforeExtension, genome);
    #endif

    batch.resetElements();

    batch.printReport();

    

    return 0;
}



