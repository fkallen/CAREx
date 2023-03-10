#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/cudaerrorcheck.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/gpumsamanaged.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <gpu/util_gpu.cuh>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>
#include <util.hpp>
#include <gpu/gpucpureadstorageadapter.cuh>
#include <gpu/gpucpuminhasheradapter.cuh>
#include <readextender_cpu.hpp>
#include <util_iterator.hpp>
#include <readextender_common.hpp>
#include <gpu/cubvector.cuh>
#include <gpu/cuda_block_select.cuh>
#include <mystringview.hpp>
#include <gpu/gpustringglueing.cuh>
#include <gpu/memcpykernel.cuh>
#include <gpu/cubwrappers.cuh>
#include <gpu/readextender_gpu_kernels.cuh>
#include <gpu/minhashqueryfilter.cuh>


#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <gpu/rmm_utilities.cuh>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


#define DO_ONLY_REMOVE_MATE_IDS



#if 0
    #define DEBUGDEVICESYNC { \
        CUDACHECK(cudaDeviceSynchronize()); \
    }

#else 
    #define DEBUGDEVICESYNC {}

#endif



namespace care{
namespace gpu{


// std::mutex someGlobalMutex;
// std::mutex someGlobalMutex2;

cudaError_t cudaEventRecordWrapper(cudaEvent_t event, cudaStream_t stream){
    //std::cerr << "record event " << event << " on stream " << stream << "\n";
    return cudaEventRecord(event, stream);
}

cudaError_t cudaEventSynchronizeWrapper(cudaEvent_t event){
    //std::cerr << "synchronize event " << event << "\n";
    return cudaEventSynchronize(event);
}

cudaError_t cudaStreamSynchronizeWrapper(cudaStream_t stream){
    //std::cerr << "synchronize stream " << stream << "\n";
    return cudaStreamSynchronize(stream);
}

cudaError_t cudaStreamWaitEventWrapper(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0){
    //std::cerr << "stream " << stream << " wait for event " << event << "\n";
    return cudaStreamWaitEvent(stream, event, flags);
}

template<int N>
struct ThrustTupleAddition;

template<>
struct ThrustTupleAddition<2>{

    template<class Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& l, const Tuple& r) const noexcept{
        return thrust::make_tuple(
            thrust::get<0>(l) + thrust::get<0>(r),
            thrust::get<1>(l) + thrust::get<1>(r)
        );
    }
};

template<>
struct ThrustTupleAddition<3>{

    template<class Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& l, const Tuple& r) const noexcept{
        return thrust::make_tuple(
            thrust::get<0>(l) + thrust::get<0>(r),
            thrust::get<1>(l) + thrust::get<1>(r),
            thrust::get<2>(l) + thrust::get<2>(r)
        );
    }
};




// __global__
// void checkSortedSegmentsKernel(
//     int numAnchors, 
//     int numCandidates,
//     const read_number* d_candidateReadIds,
//     const int* d_numCandidatesPerAnchor,
//     const int* d_numCandidatesPerAnchorPrefixSum
// ){
//     for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
//         __syncthreads();

//         for(int i = 1 + threadIdx.x; i < numAnchors + 1; i+=blockDim.x){
//             assert(d_numCandidatesPerAnchor[i-1] == 
//                 (d_numCandidatesPerAnchorPrefixSum[i] - d_numCandidatesPerAnchorPrefixSum[i - 1]));
//         }
//         if(!(numCandidates == d_numCandidatesPerAnchorPrefixSum[numAnchors])){
//             if(threadIdx.x == 0){
//                 printf("a %d, numCandidates %d, lastps %d\n", a, numCandidates, d_numCandidatesPerAnchorPrefixSum[numAnchors]);
//             }
//             __syncthreads();
//             assert(numCandidates == d_numCandidatesPerAnchorPrefixSum[numAnchors]);
//         }

//         __syncthreads();

//         const int offset = d_numCandidatesPerAnchorPrefixSum[a];
//         const int num = d_numCandidatesPerAnchor[a];

//         for(int i = 1 + threadIdx.x; i < num; i+=blockDim.x){
//             assert(d_candidateReadIds[offset + i - 1] < d_candidateReadIds[offset + i]);
//         }

//         __syncthreads();
//     }
// }


struct GpuReadExtender{

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    struct AnchorData{
        AnchorData(cudaStream_t stream, rmm::mr::device_memory_resource* mr_)
            : mr(mr_),
            d_anchorSequencesLength(0, stream, mr),
            d_accumExtensionsLengths(0, stream, mr),
            d_anchorSequencesDataDecoded(0, stream, mr),
            d_anchorQualityScores(0, stream, mr),
            d_anchorSequencesData(0, stream, mr){

            CUDACHECK(cudaStreamSynchronize(stream));
        }

        std::size_t encodedSequencePitchInInts = 0;
        std::size_t decodedSequencePitchInBytes = 0;
        std::size_t qualityPitchInBytes = 0;

        rmm::mr::device_memory_resource* mr;
        rmm::device_uvector<int> d_anchorSequencesLength;
        rmm::device_uvector<int> d_accumExtensionsLengths;
        rmm::device_uvector<char> d_anchorSequencesDataDecoded;
        rmm::device_uvector<char> d_anchorQualityScores;
        rmm::device_uvector<unsigned int> d_anchorSequencesData;
    };

    struct AnchorHashResult{
        AnchorHashResult(cudaStream_t stream, rmm::mr::device_memory_resource* mr_)
            : mr(mr_),
            d_candidateReadIds(0, stream),
            d_numCandidatesPerAnchor(0, stream),
            d_numCandidatesPerAnchorPrefixSum(0, stream){

            CUDACHECK(cudaStreamSynchronize(stream));
        }

        void resizeBuffersUninitialized(std::size_t newsize, cudaStream_t stream){
            resizeUninitialized(d_numCandidatesPerAnchor, newsize, stream);
            resizeUninitialized(d_numCandidatesPerAnchorPrefixSum, newsize + 1, stream);
        }

        PinnedBuffer<int> h_tmp{1};
        rmm::mr::device_memory_resource* mr;
        rmm::device_uvector<read_number> d_candidateReadIds;
        rmm::device_uvector<int> d_numCandidatesPerAnchor;
        rmm::device_uvector<int> d_numCandidatesPerAnchorPrefixSum;
    };

    struct RawExtendResult{
        bool noReadIsExtended{};
        int numResults{};
        std::size_t outputpitch{};
        std::size_t decodedSequencePitchInBytes{};
        CudaEvent event{cudaEventDisableTiming};
        PinnedBuffer<int> h_tmp{2};
        PinnedBuffer<char> h_inputAnchorsDecoded{};
        PinnedBuffer<extension::AbortReason> h_gpuabortReasons{};
        PinnedBuffer<extension::ExtensionDirection> h_gpudirections{};
        PinnedBuffer<int> h_gpuiterations{};
        PinnedBuffer<read_number> h_gpuReadIds{};
        PinnedBuffer<read_number> h_gpuMateReadIds{};
        PinnedBuffer<int> h_gpuAnchorLengths{};
        PinnedBuffer<int> h_gpuMateLengths{};
        PinnedBuffer<float> h_gpugoodscores{};
        PinnedBuffer<bool> h_gpuMateHasBeenFound{};

        PinnedBuffer<int> h_pairResultLengths{};


        int* h_pairResultRead1Begins{};
        int* h_pairResultRead2Begins{};
        char* h_pairResultSequences{};
        char* h_pairResultQualities{};
        bool* h_pairResultMateHasBeenFound{};
        bool* h_pairResultMergedDifferentStrands{};
        bool* h_pairResultAnchorIsLR{};

        PinnedBuffer<char> h_resultsperpseudoreadflat;
        void resizePseudoReadBuffers(std::size_t numPseudoReads, std::size_t sequenceAndQualPitch){
            std::size_t flatbuffers2size = sizeof(int) * numPseudoReads
                + sizeof(int) * numPseudoReads
                + sizeof(int) * numPseudoReads
                + sizeof(char) * numPseudoReads * sequenceAndQualPitch
                + sizeof(char) * numPseudoReads * sequenceAndQualPitch
                + sizeof(bool) * numPseudoReads
                + sizeof(bool) * numPseudoReads
                + sizeof(bool) * numPseudoReads;

            h_resultsperpseudoreadflat.resize(flatbuffers2size);
            h_pairResultRead1Begins = reinterpret_cast<int*>(h_resultsperpseudoreadflat.data());
            h_pairResultRead2Begins = reinterpret_cast<int*>(h_pairResultRead1Begins + numPseudoReads);
            h_pairResultSequences = reinterpret_cast<char*>(h_pairResultRead2Begins + numPseudoReads);
            h_pairResultQualities = reinterpret_cast<char*>(h_pairResultSequences + numPseudoReads * sequenceAndQualPitch);
            h_pairResultMateHasBeenFound = reinterpret_cast<bool*>(h_pairResultQualities + numPseudoReads * sequenceAndQualPitch);
            h_pairResultMergedDifferentStrands = reinterpret_cast<bool*>(h_pairResultMateHasBeenFound + numPseudoReads);
            h_pairResultAnchorIsLR = reinterpret_cast<bool*>(h_pairResultMergedDifferentStrands + numPseudoReads);
            outputpitch = sequenceAndQualPitch;
        }

    private:
    };

    struct IterationConfig{
        int maxextensionPerStep{1};
        int minCoverageForExtension{1};
    };

    struct TaskData{
        template<class T>
        using HostVector = std::vector<T>;

        int deviceId = 0;
        std::size_t entries = 0;
        std::size_t reservedEntries = 0;
        std::size_t encodedSequencePitchInInts = 0;
        std::size_t decodedSequencePitchInBytes = 0;
        std::size_t qualityPitchInBytes = 0;
        rmm::mr::device_memory_resource* mr;
        rmm::device_uvector<bool> pairedEnd;
        rmm::device_uvector<bool> mateHasBeenFound;
        rmm::device_uvector<int> id;
        rmm::device_uvector<int> pairId;
        rmm::device_uvector<int> iteration;
        rmm::device_uvector<float> goodscore;
        rmm::device_uvector<read_number> myReadId;
        rmm::device_uvector<read_number> mateReadId;
        rmm::device_uvector<extension::AbortReason> abortReason;
        rmm::device_uvector<extension::ExtensionDirection> direction;
        rmm::device_uvector<unsigned int> inputEncodedMate;
        rmm::device_uvector<unsigned int> inputAnchorsEncoded;
        rmm::device_uvector<char> soainputdecodedMateRevC;
        rmm::device_uvector<char> soainputmateQualityScoresReversed;
        rmm::device_uvector<int> soainputmateLengths;
        rmm::device_uvector<char> soainputAnchorsDecoded;
        rmm::device_uvector<char> soainputAnchorQualities;
        rmm::device_uvector<int> soainputAnchorLengths;
        rmm::device_uvector<int> soatotalDecodedAnchorsLengths;
        rmm::device_uvector<char> soatotalDecodedAnchorsFlat;
        rmm::device_uvector<char> soatotalAnchorQualityScoresFlat;
        rmm::device_uvector<int> soatotalAnchorBeginInExtendedRead;
        rmm::device_uvector<int> soaNumIterationResultsPerTask;
        rmm::device_uvector<int> soaNumIterationResultsPerTaskPrefixSum;

        rmm::device_uvector<read_number> d_usedReadIds;
        rmm::device_uvector<int> d_numUsedReadIdsPerTask;
        rmm::device_uvector<int> d_numUsedReadIdsPerTaskPrefixSum;

        // rmm::device_uvector<read_number> d_fullyUsedReadIds;
        // rmm::device_uvector<int> d_numFullyUsedReadIdsPerTask;
        // rmm::device_uvector<int> d_numFullyUsedReadIdsPerTaskPrefixSum;



        void consistencyCheck(cudaStream_t stream, bool verbose = false) const{
            assert(size() == entries);
            assert(pairedEnd.size() == size());
            assert(mateHasBeenFound.size() == size());
            assert(id.size() == size());
            assert(pairId.size() == size());
            assert(iteration.size() == size());
            assert(goodscore.size() == size());
            assert(myReadId.size() == size());
            assert(mateReadId.size() == size());
            assert(abortReason.size() == size());
            assert(direction.size() == size());
            assert(soainputmateLengths.size() == size());
            assert(soainputAnchorLengths.size() == size());
            assert(soaNumIterationResultsPerTask.size() == size());
            assert(soaNumIterationResultsPerTaskPrefixSum.size() == size() + 1);

            assert(d_numUsedReadIdsPerTask.size() == size());
            assert(d_numUsedReadIdsPerTaskPrefixSum.size() == size() + 1);
            // assert(d_numFullyUsedReadIdsPerTask.size() == size());
            // assert(d_numFullyUsedReadIdsPerTaskPrefixSum.size() == size() + 1);

            if(verbose){

                std::vector<int> nums(size());
                std::vector<int> numsPS(size()+1);
                CUDACHECK(cudaMemcpyAsync(nums.data(), d_numUsedReadIdsPerTask.data(), sizeof(int) * size(), D2H, stream));
                CUDACHECK(cudaMemcpyAsync(numsPS.data(), d_numUsedReadIdsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, stream));
                CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                std::cerr << "used nums\n";
                std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                std::cerr << "used numsPS\n";
                std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                // CUDACHECK(cudaMemcpyAsync(nums.data(), d_numFullyUsedReadIdsPerTask.data(), sizeof(int) * size(), D2H, stream));
                // CUDACHECK(cudaMemcpyAsync(numsPS.data(), d_numFullyUsedReadIdsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, stream));
                // CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                // std::cerr << "fully used nums\n";
                // std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                // std::cerr << "\n";

                // std::cerr << "fully used numsPS\n";
                // std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                // std::cerr << "\n";

                CUDACHECK(cudaMemcpyAsync(nums.data(), soaNumIterationResultsPerTask.data(), sizeof(int) * size(), D2H, stream));
                CUDACHECK(cudaMemcpyAsync(numsPS.data(), soaNumIterationResultsPerTaskPrefixSum.data(), sizeof(int) * (size()+1), D2H, stream));
                CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                std::cerr << "iteration result nums\n";
                std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";

                std::cerr << "iteration result numsPS\n";
                std::copy(numsPS.begin(), numsPS.end(), std::ostream_iterator<int>(std::cerr, ","));
                std::cerr << "\n";
            }

            #if 0
                //CUDACHECK(cudaDeviceSynchronize());

                // int numUsedIds = 0;
                // int numFullyUsedIds = 0;
                // CUDACHECK(cudaMemcpyAsync(&numUsedIds, d_numUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
                // CUDACHECK(cudaMemcpyAsync(&numFullyUsedIds, d_numFullyUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
                // CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                // if(numUsedIds != int(d_usedReadIds.size())){
                //     std::cerr << "numUsedIds " << numUsedIds << ", d_usedReadIds.size() " << d_usedReadIds.size() << "\n";
                // }

                // if(numFullyUsedIds != int(d_fullyUsedReadIds.size())){
                //     std::cerr << "numFullyUsedIds " << numFullyUsedIds << ", d_fullyUsedReadIds.size() " << d_fullyUsedReadIds.size() << "\n";
                // }

                // assert(numUsedIds == int(d_usedReadIds.size()));
                // assert(numFullyUsedIds == int(d_fullyUsedReadIds.size()));
            #endif
        }

        TaskData(rmm::mr::device_memory_resource* mr_) : TaskData(mr_, 0,0,0,0, (cudaStream_t)0) {}

        TaskData(
            rmm::mr::device_memory_resource* mr_, 
            int size, 
            std::size_t encodedSequencePitchInInts_, 
            std::size_t decodedSequencePitchInBytes_, 
            std::size_t qualityPitchInBytes_,
            cudaStream_t stream
        ) 
            : encodedSequencePitchInInts(encodedSequencePitchInInts_), 
                decodedSequencePitchInBytes(decodedSequencePitchInBytes_), 
                qualityPitchInBytes(qualityPitchInBytes_),
                mr(mr_),
                pairedEnd{0, stream, mr},
                mateHasBeenFound{0, stream, mr},
                id{0, stream, mr},
                pairId{0, stream, mr},
                iteration{0, stream, mr},
                goodscore{0, stream, mr},
                myReadId{0, stream, mr},
                mateReadId{0, stream, mr},
                abortReason{0, stream, mr},
                direction{0, stream, mr},
                inputEncodedMate{0, stream, mr},
                inputAnchorsEncoded{0, stream, mr},
                soainputdecodedMateRevC{0, stream, mr},
                soainputmateQualityScoresReversed{0, stream, mr},
                soainputmateLengths{0, stream, mr},
                soainputAnchorsDecoded{0, stream, mr},
                soainputAnchorQualities{0, stream, mr},
                soainputAnchorLengths{0, stream, mr},
                soatotalDecodedAnchorsLengths{0, stream, mr},
                soatotalDecodedAnchorsFlat{0, stream, mr},
                soatotalAnchorQualityScoresFlat{0, stream, mr},
                soatotalAnchorBeginInExtendedRead{0, stream, mr},
                soaNumIterationResultsPerTask{0, stream, mr},
                soaNumIterationResultsPerTaskPrefixSum{0, stream, mr},
                d_usedReadIds{0, stream, mr},
                d_numUsedReadIdsPerTask{0, stream, mr},
                d_numUsedReadIdsPerTaskPrefixSum{0, stream, mr}
                // d_fullyUsedReadIds{0, stream, mr},
                // d_numFullyUsedReadIdsPerTask{0, stream, mr},
                // d_numFullyUsedReadIdsPerTaskPrefixSum{0, stream, mr}
        {
            ////std::cerr << "task " << this << " constructor, stream " << stream << "\n";
            CUDACHECK(cudaGetDevice(&deviceId));
            resize(size, stream);

            //std::cerr << "after construct\n";
            consistencyCheck(stream);

        }

        std::size_t size() const noexcept{
            return entries;
        }

        std::size_t capacity() const noexcept{
            return reservedEntries;
        }

        void clearBuffers(cudaStream_t stream){
            //std::cerr << "task " << this << " clear, stream " << stream << "\n";
            clear(pairedEnd, stream);
            clear(mateHasBeenFound, stream);
            clear(id, stream);
            clear(pairId, stream);
            clear(iteration, stream);
            clear(goodscore, stream);
            clear(myReadId, stream);
            clear(mateReadId, stream);
            clear(abortReason, stream);
            clear(direction, stream);
            clear(inputEncodedMate, stream);
            clear(inputAnchorsEncoded, stream);
            clear(soainputdecodedMateRevC, stream);
            clear(soainputmateQualityScoresReversed, stream);
            clear(soainputmateLengths, stream);
            clear(soainputAnchorsDecoded, stream);
            clear(soainputAnchorQualities, stream);
            clear(soainputAnchorLengths, stream);
            destroy(soatotalDecodedAnchorsLengths, stream);
            destroy(soatotalDecodedAnchorsFlat, stream);
            destroy(soatotalAnchorQualityScoresFlat, stream);
            destroy(soatotalAnchorBeginInExtendedRead, stream);
            clear(soaNumIterationResultsPerTask, stream);
            resizeUninitialized(soaNumIterationResultsPerTaskPrefixSum, 1, stream);
            CUDACHECK(cudaMemsetAsync(soaNumIterationResultsPerTaskPrefixSum.data(), 0, sizeof(int), stream));

            destroy(d_usedReadIds, stream);
            clear(d_numUsedReadIdsPerTask, stream);
            resizeUninitialized(d_numUsedReadIdsPerTaskPrefixSum, 1, stream);
            CUDACHECK(cudaMemsetAsync(d_numUsedReadIdsPerTaskPrefixSum.data(), 0, sizeof(int), stream));

            // destroy(d_fullyUsedReadIds, stream);
            // clear(d_numFullyUsedReadIdsPerTask, stream);
            // resizeUninitialized(d_numFullyUsedReadIdsPerTaskPrefixSum, 1, stream);
            // CUDACHECK(cudaMemsetAsync(d_numFullyUsedReadIdsPerTaskPrefixSum.data(), 0, sizeof(int), stream));

            entries = 0;
            //std::cerr << "after clear\n";
            consistencyCheck(stream);
        }

        void reserveBuffers(std::size_t newsize, cudaStream_t stream){
            //std::cerr << "task " << this << " reserve, stream " << stream << "\n";
            reserve(pairedEnd, newsize, stream);
            reserve(mateHasBeenFound, newsize, stream);
            reserve(id, newsize, stream);
            reserve(pairId, newsize, stream);
            reserve(iteration, newsize, stream);
            reserve(goodscore, newsize, stream);
            reserve(myReadId, newsize, stream);
            reserve(mateReadId, newsize, stream);
            reserve(abortReason, newsize, stream);
            reserve(direction, newsize, stream);
            reserve(inputEncodedMate, newsize * encodedSequencePitchInInts, stream);
            reserve(inputAnchorsEncoded, newsize * encodedSequencePitchInInts, stream);
            reserve(soainputdecodedMateRevC, newsize * decodedSequencePitchInBytes, stream);
            reserve(soainputmateQualityScoresReversed, newsize * qualityPitchInBytes, stream);
            reserve(soainputmateLengths, newsize, stream);
            reserve(soainputAnchorsDecoded, newsize * decodedSequencePitchInBytes, stream);
            reserve(soainputAnchorQualities, newsize * qualityPitchInBytes, stream);
            reserve(soainputAnchorLengths, newsize, stream);
            reserve(soaNumIterationResultsPerTask, newsize, stream);
            reserve(soaNumIterationResultsPerTaskPrefixSum, newsize + 1, stream);

            reserve(d_numUsedReadIdsPerTask, newsize, stream);
            reserve(d_numUsedReadIdsPerTaskPrefixSum, newsize + 1, stream);

            // reserve(d_numFullyUsedReadIdsPerTask, newsize, stream);
            // reserve(d_numFullyUsedReadIdsPerTaskPrefixSum, newsize + 1, stream);

            reservedEntries = newsize;

            //std::cerr << "after reserve\n";
            consistencyCheck(stream);
        }

        void resize(std::size_t newsize, cudaStream_t stream){
            //std::cerr << "task " << this << " resize, stream " << stream << "\n";
            pairedEnd.resize(newsize, stream);
            mateHasBeenFound.resize(newsize, stream);
            id.resize(newsize, stream);
            pairId.resize(newsize, stream);
            iteration.resize(newsize, stream);
            goodscore.resize(newsize, stream);
            myReadId.resize(newsize, stream);
            mateReadId.resize(newsize, stream);
            abortReason.resize(newsize, stream);
            direction.resize(newsize, stream);
            inputEncodedMate.resize(newsize * encodedSequencePitchInInts, stream);
            inputAnchorsEncoded.resize(newsize * encodedSequencePitchInInts, stream);
            soainputdecodedMateRevC.resize(newsize * decodedSequencePitchInBytes, stream);
            soainputmateQualityScoresReversed.resize(newsize * qualityPitchInBytes, stream);
            soainputmateLengths.resize(newsize, stream);
            soainputAnchorsDecoded.resize(newsize * decodedSequencePitchInBytes, stream);
            soainputAnchorQualities.resize(newsize * qualityPitchInBytes, stream);
            soainputAnchorLengths.resize(newsize, stream);
            soaNumIterationResultsPerTask.resize(newsize, stream);
            soaNumIterationResultsPerTaskPrefixSum.resize(newsize + 1, stream);

            d_numUsedReadIdsPerTask.resize(newsize, stream);
            d_numUsedReadIdsPerTaskPrefixSum.resize(newsize + 1, stream);

            // d_numFullyUsedReadIdsPerTask.resize(newsize, stream);
            // d_numFullyUsedReadIdsPerTaskPrefixSum.resize(newsize + 1, stream);

            if(size() > 0){
                if(newsize > size()){

                    //repeat last element of prefix sum in newly added elements. fill numbers with 0

                    helpers::lambda_kernel<<<SDIV(newsize - size(), 128), 128, 0, stream>>>(
                        [
                            soaNumIterationResultsPerTaskPrefixSum = soaNumIterationResultsPerTaskPrefixSum.data(),
                            d_numUsedReadIdsPerTaskPrefixSum = d_numUsedReadIdsPerTaskPrefixSum.data(),
                            //d_numFullyUsedReadIdsPerTaskPrefixSum = d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                            soaNumIterationResultsPerTask = soaNumIterationResultsPerTask.data(),
                            d_numUsedReadIdsPerTask = d_numUsedReadIdsPerTask.data(),
                            //d_numFullyUsedReadIdsPerTask = d_numFullyUsedReadIdsPerTask.data(),
                            size = size(),
                            newsize = newsize
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                            const int stride = blockDim.x * gridDim.x;

                            for(int i = tid; i < newsize - size; i += stride){
                                soaNumIterationResultsPerTaskPrefixSum[size + 1 + i] = soaNumIterationResultsPerTaskPrefixSum[size];
                                d_numUsedReadIdsPerTaskPrefixSum[size + 1 + i] = d_numUsedReadIdsPerTaskPrefixSum[size];
                                //d_numFullyUsedReadIdsPerTaskPrefixSum[size + 1 + i] = d_numFullyUsedReadIdsPerTaskPrefixSum[size];

                                soaNumIterationResultsPerTask[size + i] = 0;
                                d_numUsedReadIdsPerTask[size + i] = 0;
                                //d_numFullyUsedReadIdsPerTask[size + i] = 0;
                            }
                        }
                    ); CUDACHECKASYNC;
                }
            }else{
                if(newsize > 0){
                    readextendergpukernels::fillKernel<<<SDIV(newsize, 128), 128, 0, stream>>>(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            soaNumIterationResultsPerTask.data(),
                            d_numUsedReadIdsPerTask.data()
                            //d_numFullyUsedReadIdsPerTask.data()
                        )),
                        newsize,
                        thrust::make_tuple(
                            0,
                            0
                            //0
                        )
                    ); CUDACHECKASYNC;
                }

                readextendergpukernels::fillKernel<<<SDIV(newsize+1, 128), 128, 0, stream>>>(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        soaNumIterationResultsPerTaskPrefixSum.data(),
                        d_numUsedReadIdsPerTaskPrefixSum.data()
                        //d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    newsize+1,
                    thrust::make_tuple(
                        0,
                        0
                        //0
                    )
                ); CUDACHECKASYNC;

            }

            entries = newsize;
            reservedEntries = std::max(entries, reservedEntries);

            //std::cerr << "after resize\n";
            consistencyCheck(stream);
        }

        bool checkPitch(const TaskData& rhs) const noexcept{
            if(encodedSequencePitchInInts != rhs.encodedSequencePitchInInts) return false;
            if(decodedSequencePitchInBytes != rhs.decodedSequencePitchInBytes) return false;
            if(qualityPitchInBytes != rhs.qualityPitchInBytes) return false;
            return true;
        }

        void aggregateAnchorData(AnchorData& anchorData, cudaStream_t stream){
            //std::cerr << "task " << this << " aggregateAnchorData, stream " << stream << "\n";
            resizeUninitialized(anchorData.d_anchorSequencesLength, size(), stream);
            resizeUninitialized(anchorData.d_accumExtensionsLengths, size(), stream);
            resizeUninitialized(anchorData.d_anchorSequencesDataDecoded, size() * decodedSequencePitchInBytes, stream);
            resizeUninitialized(anchorData.d_anchorQualityScores, size() * qualityPitchInBytes, stream);
            resizeUninitialized(anchorData.d_anchorSequencesData, size() * encodedSequencePitchInInts, stream);

            anchorData.encodedSequencePitchInInts = encodedSequencePitchInInts;
            anchorData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
            anchorData.qualityPitchInBytes = qualityPitchInBytes;

            if(size() > 0){
                // assert(!(soaNumIterationResultsPerTask.previousAllocStream.has_value() && stream != *soaNumIterationResultsPerTask.previousAllocStream));
                // assert(!(soaNumIterationResultsPerTaskPrefixSum.previousAllocStream.has_value() && stream != *soaNumIterationResultsPerTaskPrefixSum.previousAllocStream));
                // assert(!(soatotalAnchorBeginInExtendedRead.previousAllocStream.has_value() && stream != *soatotalAnchorBeginInExtendedRead.previousAllocStream));
                // assert(!(soatotalDecodedAnchorsLengths.previousAllocStream.has_value() && stream != *soatotalDecodedAnchorsLengths.previousAllocStream));
                // assert(!(soainputAnchorLengths.previousAllocStream.has_value() && stream != *soainputAnchorLengths.previousAllocStream));
                // assert(!(soatotalAnchorQualityScoresFlat.previousAllocStream.has_value() && stream != *soatotalAnchorQualityScoresFlat.previousAllocStream));
                // assert(!(soainputAnchorQualities.previousAllocStream.has_value() && stream != *soainputAnchorQualities.previousAllocStream));
                // assert(!(soatotalDecodedAnchorsFlat.previousAllocStream.has_value() && stream != *soatotalDecodedAnchorsFlat.previousAllocStream));
                // assert(!(soainputAnchorsDecoded.previousAllocStream.has_value() && stream != *soainputAnchorsDecoded.previousAllocStream));

                //compact some data of tasks into contiguous buffers 
                const int threads = size() * 32;
                readextendergpukernels::updateWorkingSetFromTasksKernel<256,32><<<SDIV(threads, 256), 256, 0, stream>>>(
                    size(),
                    qualityPitchInBytes,
                    decodedSequencePitchInBytes,
                    soaNumIterationResultsPerTask.data(),
                    soaNumIterationResultsPerTaskPrefixSum.data(),
                    anchorData.d_accumExtensionsLengths.data(),
                    anchorData.d_anchorSequencesLength.data(),
                    anchorData.d_anchorQualityScores.data(),
                    anchorData.d_anchorSequencesDataDecoded.data(),
                    soatotalAnchorBeginInExtendedRead.data(),
                    soatotalDecodedAnchorsLengths.data(),
                    soainputAnchorLengths.data(),
                    soatotalAnchorQualityScoresFlat.data(),
                    soainputAnchorQualities.data(),
                    soatotalDecodedAnchorsFlat.data(),
                    soainputAnchorsDecoded.data()
                ); CUDACHECKASYNC;

                callEncodeSequencesTo2BitKernel(
                    anchorData.d_anchorSequencesData.data(),
                    anchorData.d_anchorSequencesDataDecoded.data(),
                    anchorData.d_anchorSequencesLength.data(),
                    decodedSequencePitchInBytes,
                    encodedSequencePitchInInts,
                    size(),
                    8,
                    stream
                );
            }
        }

        void addTasks(
            int numReadPairs,
            // for the arrays, two consecutive numbers / sequences belong to same read pair
            const read_number* d_readpair_readIds,
            const int* d_readpair_readLengths,
            const unsigned int * d_readpair_sequences,
            const char* d_readpair_qualities,
            cudaStream_t stream
        ){
            ////std::cerr << "task " << this << " addTasks, stream " << stream << "\n";
            if(numReadPairs == 0) return;

            const int numAdditionalTasks = 4 * numReadPairs;

            TaskData newGpuSoaTaskData(mr, numAdditionalTasks, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

            readextendergpukernels::createGpuTaskData<128,8>
                <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, stream>>>(
                numReadPairs,
                d_readpair_readIds,
                d_readpair_readLengths,
                d_readpair_sequences,
                d_readpair_qualities,
                newGpuSoaTaskData.pairedEnd.data(),
                newGpuSoaTaskData.mateHasBeenFound.data(),
                newGpuSoaTaskData.id.data(),
                newGpuSoaTaskData.pairId.data(),
                newGpuSoaTaskData.iteration.data(),
                newGpuSoaTaskData.goodscore.data(),
                newGpuSoaTaskData.myReadId.data(),
                newGpuSoaTaskData.mateReadId.data(),
                newGpuSoaTaskData.abortReason.data(),
                newGpuSoaTaskData.direction.data(),
                newGpuSoaTaskData.inputEncodedMate.data(),
                newGpuSoaTaskData.soainputdecodedMateRevC.data(),
                newGpuSoaTaskData.soainputmateQualityScoresReversed.data(),
                newGpuSoaTaskData.soainputmateLengths.data(),
                newGpuSoaTaskData.inputAnchorsEncoded.data(),
                newGpuSoaTaskData.soainputAnchorsDecoded.data(),
                newGpuSoaTaskData.soainputAnchorQualities.data(),
                newGpuSoaTaskData.soainputAnchorLengths.data(),
                newGpuSoaTaskData.soaNumIterationResultsPerTask.data(),
                newGpuSoaTaskData.soaNumIterationResultsPerTaskPrefixSum.data(),
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                encodedSequencePitchInInts
            ); CUDACHECKASYNC;

            append(newGpuSoaTaskData, stream);
        }

        void append(const TaskData& rhs, cudaStream_t stream){
            //std::cerr << "task " << this << " append, stream " << stream << "\n";
            assert(checkPitch(rhs));

            nvtx::ScopedRange sr("soa append", 7);

            //std::cerr << "append check self\n";
            consistencyCheck(stream);

            //std::cerr << "append check rhs\n";
            rhs.consistencyCheck(stream);

            //create new arrays, copy both old arrays into it, then swap            
            if(rhs.size() > 0){
                const int newsize = size() + rhs.size();
                
                rmm::device_uvector<bool> newpairedEnd(newsize, stream, mr);
                rmm::device_uvector<bool> newmateHasBeenFound(newsize, stream, mr);
                rmm::device_uvector<int> newid(newsize, stream, mr);
                rmm::device_uvector<int> newpairId(newsize, stream, mr);
                rmm::device_uvector<int> newiteration(newsize, stream, mr);
                rmm::device_uvector<float> newgoodscore(newsize, stream, mr);
                rmm::device_uvector<read_number> newmyReadId(newsize, stream, mr);
                rmm::device_uvector<read_number> newmateReadId(newsize, stream, mr);
                rmm::device_uvector<extension::AbortReason> newabortReason(newsize, stream, mr);
                rmm::device_uvector<extension::ExtensionDirection> newdirection(newsize, stream, mr);
                rmm::device_uvector<int> newsoainputmateLengths(newsize, stream, mr);
                rmm::device_uvector<int> newsoainputAnchorLengths(newsize, stream, mr);
                rmm::device_uvector<int> newsoaNumIterationResultsPerTask(newsize, stream, mr);
                rmm::device_uvector<int> newnumUsedReadidsPerTask(newsize, stream, mr);
                //rmm::device_uvector<int> newnumFullyUsedReadidsPerTask(newsize, stream, mr);


                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        pairedEnd.data(),
                        mateHasBeenFound.data(),
                        id.data(),
                        pairId.data(),
                        iteration.data(),
                        goodscore.data(),
                        myReadId.data(),
                        mateReadId.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newpairedEnd.data(),
                        newmateHasBeenFound.data(),
                        newid.data(),
                        newpairId.data(),
                        newiteration.data(),
                        newgoodscore.data(),
                        newmyReadId.data(),
                        newmateReadId.data()
                    ))
                );

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.pairedEnd.data(),
                        rhs.mateHasBeenFound.data(),
                        rhs.id.data(),
                        rhs.pairId.data(),
                        rhs.iteration.data(),
                        rhs.goodscore.data(),
                        rhs.myReadId.data(),
                        rhs.mateReadId.data()
                    )),
                    rhs.size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newpairedEnd.data() + size(),
                        newmateHasBeenFound.data() + size(),
                        newid.data() + size(),
                        newpairId.data() + size(),
                        newiteration.data() + size(),
                        newgoodscore.data() + size(),
                        newmyReadId.data() + size(),
                        newmateReadId.data() + size()
                    ))
                );

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        abortReason.data(),
                        direction.data(),
                        soainputmateLengths.data(),
                        soainputAnchorLengths.data(),
                        soaNumIterationResultsPerTask.data(),
                        d_numUsedReadIdsPerTask.data()
                        //d_numFullyUsedReadIdsPerTask.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newabortReason.data(),
                        newdirection.data(),
                        newsoainputmateLengths.data(),
                        newsoainputAnchorLengths.data(),
                        newsoaNumIterationResultsPerTask.data(),
                        newnumUsedReadidsPerTask.data()
                        //newnumFullyUsedReadidsPerTask.data()
                    ))
                );                

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.abortReason.data(),
                        rhs.direction.data(),
                        rhs.soainputmateLengths.data(),
                        rhs.soainputAnchorLengths.data(),
                        rhs.soaNumIterationResultsPerTask.data(),
                        rhs.d_numUsedReadIdsPerTask.data()
                        //rhs.d_numFullyUsedReadIdsPerTask.data()
                    )),
                    rhs.size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newabortReason.data() + size(),
                        newdirection.data() + size(),
                        newsoainputmateLengths.data() + size(),
                        newsoainputAnchorLengths.data() + size(),
                        newsoaNumIterationResultsPerTask.data() + size(),
                        newnumUsedReadidsPerTask.data() + size()
                        //newnumFullyUsedReadidsPerTask.data() + size()
                    ))
                );

                std::swap(pairedEnd, newpairedEnd);
                std::swap(mateHasBeenFound, newmateHasBeenFound);
                std::swap(id, newid);
                std::swap(pairId, newpairId);
                std::swap(iteration, newiteration);
                std::swap(goodscore, newgoodscore);
                std::swap(myReadId, newmyReadId);
                std::swap(mateReadId, newmateReadId);
                std::swap(abortReason, newabortReason);
                std::swap(direction, newdirection);
                std::swap(soainputmateLengths, newsoainputmateLengths);
                std::swap(soainputAnchorLengths, newsoainputAnchorLengths);
                std::swap(soaNumIterationResultsPerTask, newsoaNumIterationResultsPerTask);
                std::swap(d_numUsedReadIdsPerTask, newnumUsedReadidsPerTask);
                //std::swap(d_numFullyUsedReadIdsPerTask, newnumFullyUsedReadidsPerTask);

                destroy(newpairedEnd, stream);
                destroy(newmateHasBeenFound, stream);
                destroy(newid, stream);
                destroy(newpairId, stream);
                destroy(newiteration, stream);
                destroy(newgoodscore, stream);
                destroy(newmyReadId, stream);
                destroy(newmateReadId, stream);
                destroy(newabortReason, stream);
                destroy(newdirection, stream);
                destroy(newsoainputmateLengths, stream);
                destroy(newsoainputAnchorLengths, stream);
                destroy(newsoaNumIterationResultsPerTask, stream);
                destroy(newnumUsedReadidsPerTask, stream);
                //destroy(newnumFullyUsedReadidsPerTask, stream);

                rmm::device_uvector<int> newsoaNumIterationResultsPerTaskPrefixSum(newsize + 1, stream, mr);
                rmm::device_uvector<int> newnumUsedReadidsPerTaskPrefixSum(newsize + 1, stream, mr);
                //rmm::device_uvector<int> newnumFullyUsedReadidsPerTaskPrefixSum(newsize + 1, stream, mr);

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        soaNumIterationResultsPerTaskPrefixSum.data(),
                        d_numUsedReadIdsPerTaskPrefixSum.data()
                        //d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    size(),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newsoaNumIterationResultsPerTaskPrefixSum.data(),
                        newnumUsedReadidsPerTaskPrefixSum.data()
                        //newnumFullyUsedReadidsPerTaskPrefixSum.data()
                    ))
                );                

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        rhs.soaNumIterationResultsPerTaskPrefixSum.data(),
                        rhs.d_numUsedReadIdsPerTaskPrefixSum.data()
                        //rhs.d_numFullyUsedReadIdsPerTaskPrefixSum.data()
                    )),
                    rhs.size() + 1,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newsoaNumIterationResultsPerTaskPrefixSum.data() + size(),
                        newnumUsedReadidsPerTaskPrefixSum.data() + size()
                        //newnumFullyUsedReadidsPerTaskPrefixSum.data() + size()
                    ))
                );

                std::swap(soaNumIterationResultsPerTaskPrefixSum, newsoaNumIterationResultsPerTaskPrefixSum);
                std::swap(d_numUsedReadIdsPerTaskPrefixSum, newnumUsedReadidsPerTaskPrefixSum);
                //std::swap(d_numFullyUsedReadIdsPerTaskPrefixSum, newnumFullyUsedReadidsPerTaskPrefixSum);

                destroy(newnumUsedReadidsPerTaskPrefixSum, stream);
                destroy(newsoaNumIterationResultsPerTaskPrefixSum, stream);
                //destroy(newnumFullyUsedReadidsPerTaskPrefixSum, stream);


                rmm::device_uvector<unsigned int> newinputEncodedMate(newsize * encodedSequencePitchInInts, stream, mr);
                rmm::device_uvector<unsigned int> newinputAnchorsEncoded(newsize * encodedSequencePitchInInts, stream, mr);

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        inputEncodedMate.data(),
                        inputAnchorsEncoded.data()
                    )),
                    size() * encodedSequencePitchInInts,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newinputEncodedMate.data(),
                        newinputAnchorsEncoded.data()
                    ))
                );

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        rhs.inputEncodedMate.data(),
                        rhs.inputAnchorsEncoded.data()
                    )),
                    rhs.size() * encodedSequencePitchInInts,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        newinputEncodedMate.data() + size() * encodedSequencePitchInInts,
                        newinputAnchorsEncoded.data() + size() * encodedSequencePitchInInts
                    ))
                );

                std::swap(inputEncodedMate, newinputEncodedMate);
                std::swap(inputAnchorsEncoded, newinputAnchorsEncoded);

                destroy(newinputEncodedMate, stream);
                destroy(newinputAnchorsEncoded, stream); 

                assert(decodedSequencePitchInBytes % sizeof(int) == 0);

                rmm::device_uvector<char> newsoainputdecodedMateRevC(newsize * decodedSequencePitchInBytes, stream, mr);
                rmm::device_uvector<char> newsoainputAnchorsDecoded(newsize * decodedSequencePitchInBytes, stream, mr);

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)soainputdecodedMateRevC.data(),
                        (int*)soainputAnchorsDecoded.data()
                    )),
                    size() * (decodedSequencePitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        (int*)newsoainputdecodedMateRevC.data(),
                        (int*)newsoainputAnchorsDecoded.data()
                    ))
                );

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)rhs.soainputdecodedMateRevC.data(),
                        (int*)rhs.soainputAnchorsDecoded.data()
                    )),
                    rhs.size() * (decodedSequencePitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        ((int*)newsoainputdecodedMateRevC.data()) + size() * (decodedSequencePitchInBytes / sizeof(int)),
                        ((int*)newsoainputAnchorsDecoded.data()) + size() * (decodedSequencePitchInBytes / sizeof(int))
                    ))
                );

                std::swap(soainputdecodedMateRevC, newsoainputdecodedMateRevC);
                std::swap(soainputAnchorsDecoded, newsoainputAnchorsDecoded);

                destroy(newsoainputdecodedMateRevC, stream);
                destroy(newsoainputAnchorsDecoded, stream); 

                assert(qualityPitchInBytes % sizeof(int) == 0);

                rmm::device_uvector<char> newsoainputmateQualityScoresReversed(newsize * qualityPitchInBytes, stream, mr);
                rmm::device_uvector<char> newsoainputAnchorQualities(newsize * qualityPitchInBytes, stream, mr);

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)soainputmateQualityScoresReversed.data(),
                        (int*)soainputAnchorQualities.data()
                    )),
                    size() * (qualityPitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        (int*)newsoainputmateQualityScoresReversed.data(),
                        (int*)newsoainputAnchorQualities.data()
                    ))
                );

                thrust::copy_n(
                    rmm::exec_policy_nosync(stream,mr),
                    thrust::make_zip_iterator(thrust::make_tuple(                    
                        (int*)rhs.soainputmateQualityScoresReversed.data(),
                        (int*)rhs.soainputAnchorQualities.data()
                    )),
                    rhs.size() * (qualityPitchInBytes / sizeof(int)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        ((int*)newsoainputmateQualityScoresReversed.data()) + size() * (qualityPitchInBytes / sizeof(int)),
                        ((int*)newsoainputAnchorQualities.data()) + size() * (qualityPitchInBytes / sizeof(int))
                    ))
                );

                std::swap(soainputmateQualityScoresReversed, newsoainputmateQualityScoresReversed);
                std::swap(soainputAnchorQualities, newsoainputAnchorQualities);

                destroy(newsoainputmateQualityScoresReversed, stream);
                destroy(newsoainputAnchorQualities, stream); 

                #if 0

                append(soatotalDecodedAnchorsLengths, rhs.soatotalDecodedAnchorsLengths.data(), rhs.soatotalDecodedAnchorsLengths.data() + rhs.soatotalDecodedAnchorsLengths.size(), stream);
                append(soatotalAnchorBeginInExtendedRead, rhs.soatotalAnchorBeginInExtendedRead.data(), rhs.soatotalAnchorBeginInExtendedRead.data() + rhs.soatotalAnchorBeginInExtendedRead.size(), stream);

                append(soatotalDecodedAnchorsFlat, rhs.soatotalDecodedAnchorsFlat.data(), rhs.soatotalDecodedAnchorsFlat.data() + rhs.soatotalDecodedAnchorsFlat.size(), stream);
                append(soatotalAnchorQualityScoresFlat, rhs.soatotalAnchorQualityScoresFlat.data(), rhs.soatotalAnchorQualityScoresFlat.data() + rhs.soatotalAnchorQualityScoresFlat.size(), stream);
            
                append(d_usedReadIds, rhs.d_usedReadIds.data(), rhs.d_usedReadIds.data() + rhs.d_usedReadIds.size(), stream);
                //append(d_fullyUsedReadIds, rhs.d_fullyUsedReadIds.data(), rhs.d_fullyUsedReadIds.data() + rhs.d_fullyUsedReadIds.size(), stream);

                #else 

                rmm::device_uvector<int> newsoatotalDecodedAnchorsLengths(soatotalDecodedAnchorsLengths.size() + rhs.soatotalDecodedAnchorsLengths.size(), stream, mr);
                rmm::device_uvector<int> newsoatotalAnchorBeginInExtendedRead(soatotalAnchorBeginInExtendedRead.size() + rhs.soatotalAnchorBeginInExtendedRead.size(), stream, mr);

                rmm::device_uvector<read_number> newd_usedReadIds(d_usedReadIds.size() + rhs.d_usedReadIds.size(), stream, mr);
                //rmm::device_uvector<read_number> newd_fullyUsedReadIds(d_fullyUsedReadIds.size() + rhs.d_fullyUsedReadIds.size(), stream, mr);

                std::size_t elements = 0;
                elements = std::max(elements, soatotalDecodedAnchorsLengths.size());
                elements = std::max(elements, rhs.soatotalDecodedAnchorsLengths.size());
                elements = std::max(elements, soatotalAnchorBeginInExtendedRead.size());
                elements = std::max(elements, rhs.soatotalAnchorBeginInExtendedRead.size());
                elements = std::max(elements, d_usedReadIds.size());
                elements = std::max(elements, rhs.d_usedReadIds.size());
                //elements = std::max(elements, d_fullyUsedReadIds.size());
                //elements = std::max(elements, rhs.d_fullyUsedReadIds.size());

                if(elements > 0){

                    using care::gpu::MemcpyParams;

                    auto memcpyParams1 = cuda::std::tuple_cat(
                        cuda::std::make_tuple(MemcpyParams(newsoatotalDecodedAnchorsLengths.data(), soatotalDecodedAnchorsLengths.data(), sizeof(int) * soatotalDecodedAnchorsLengths.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalDecodedAnchorsLengths.data() + soatotalDecodedAnchorsLengths.size(), rhs.soatotalDecodedAnchorsLengths.data(), sizeof(int) * rhs.soatotalDecodedAnchorsLengths.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalAnchorBeginInExtendedRead.data(), soatotalAnchorBeginInExtendedRead.data(), sizeof(int) * soatotalAnchorBeginInExtendedRead.size())),
                        cuda::std::make_tuple(MemcpyParams(newsoatotalAnchorBeginInExtendedRead.data() + soatotalAnchorBeginInExtendedRead.size(), rhs.soatotalAnchorBeginInExtendedRead.data(), sizeof(int) * rhs.soatotalAnchorBeginInExtendedRead.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_usedReadIds.data(), d_usedReadIds.data(), sizeof(read_number) * d_usedReadIds.size())),
                        cuda::std::make_tuple(MemcpyParams(newd_usedReadIds.data() + d_usedReadIds.size(), rhs.d_usedReadIds.data(), sizeof(read_number) * rhs.d_usedReadIds.size()))
                        //cuda::std::make_tuple(MemcpyParams(newd_fullyUsedReadIds.data(), d_fullyUsedReadIds.data(), sizeof(read_number) * d_fullyUsedReadIds.size())),
                        //cuda::std::make_tuple(MemcpyParams(newd_fullyUsedReadIds.data() + d_fullyUsedReadIds.size(), rhs.d_fullyUsedReadIds.data(), sizeof(read_number) * rhs.d_fullyUsedReadIds.size()))
                    );

                    care::gpu::memcpyKernel<int><<<SDIV(elements, 256), 256, 0, stream>>>(memcpyParams1); CUDACHECKASYNC;
                }

                std::swap(soatotalDecodedAnchorsLengths, newsoatotalDecodedAnchorsLengths);
                std::swap(soatotalAnchorBeginInExtendedRead, newsoatotalAnchorBeginInExtendedRead);
                std::swap(d_usedReadIds, newd_usedReadIds);
                //std::swap(d_fullyUsedReadIds, newd_fullyUsedReadIds);

                destroy(newsoatotalDecodedAnchorsLengths, stream);
                destroy(newsoatotalAnchorBeginInExtendedRead, stream);
                destroy(newd_usedReadIds, stream);
                //destroy(newd_fullyUsedReadIds, stream);

                ::append(soatotalDecodedAnchorsFlat, rhs.soatotalDecodedAnchorsFlat.data(), rhs.soatotalDecodedAnchorsFlat.data() + rhs.soatotalDecodedAnchorsFlat.size(), stream);
                ::append(soatotalAnchorQualityScoresFlat, rhs.soatotalAnchorQualityScoresFlat.data(), rhs.soatotalAnchorQualityScoresFlat.data() + rhs.soatotalAnchorQualityScoresFlat.size(), stream);

                

                #endif
            }


            if(rhs.size() > 0){
                //fix appended prefixsums
                // readextendergpukernels::taskFixAppendedPrefixSumsKernel<128><<<SDIV(rhs.size()+1, 128), 128, 0, stream>>>(
                //     soaNumIterationResultsPerTaskPrefixSum.data(),
                //     d_numUsedReadIdsPerTaskPrefixSum.data(),
                //     d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
                //     soaNumIterationResultsPerTask.data(),
                //     d_numUsedReadIdsPerTask.data(),
                //     d_numFullyUsedReadIdsPerTask.data(),
                //     size(),
                //     rhs.size()
                // ); CUDACHECKASYNC;
                readextendergpukernels::taskFixAppendedPrefixSumsKernel<128><<<SDIV(rhs.size()+1, 128), 128, 0, stream>>>(
                    soaNumIterationResultsPerTaskPrefixSum.data(),
                    d_numUsedReadIdsPerTaskPrefixSum.data(),
                    soaNumIterationResultsPerTask.data(),
                    d_numUsedReadIdsPerTask.data(),
                    size(),
                    rhs.size()
                ); CUDACHECKASYNC;

            }

            entries += rhs.size();
            reservedEntries = std::max(entries, reservedEntries);

            //std::cerr << "after append\n";
            consistencyCheck(stream);
        }

        void iterationIsFinished(cudaStream_t stream){
            //std::cerr << "task " << this << " iterationIsFinished, stream " << stream << "\n\n";
            if(size() > 0){
                // assert(!(direction.previousAllocStream.has_value() && stream != *direction.previousAllocStream));
                // assert(!(pairedEnd.previousAllocStream.has_value() && stream != *pairedEnd.previousAllocStream));
                // assert(!(mateHasBeenFound.previousAllocStream.has_value() && stream != *mateHasBeenFound.previousAllocStream));
                // assert(!(pairId.previousAllocStream.has_value() && stream != *pairId.previousAllocStream));
                // assert(!(id.previousAllocStream.has_value() && stream != *id.previousAllocStream));
                // assert(!(abortReason.previousAllocStream.has_value() && stream != *abortReason.previousAllocStream));
                // assert(!(iteration.previousAllocStream.has_value() && stream != *iteration.previousAllocStream));

                consistencyCheck(stream);

                readextendergpukernels::taskIncrementIterationKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                    size(),
                    direction.data(),
                    pairedEnd.data(),
                    mateHasBeenFound.data(),
                    pairId.data(),
                    id.data(),
                    abortReason.data(), 
                    iteration.data()
                ); CUDACHECKASYNC;
            }
        }

        static constexpr std::size_t getHostTempStorageSize() noexcept{
            return 128;
        }

        template<class FlagIter>
        TaskData select(FlagIter d_selectionFlags, cudaStream_t stream, void* hostTempStorage){
            //std::cerr << "task " << this << " select, stream " << stream << "\n";
            nvtx::ScopedRange sr("soa_select", 1);
            rmm::device_uvector<int> positions(entries, stream, mr);
            rmm::device_scalar<int> d_numSelected(stream, mr);

            CUDACHECKASYNC; //DEBUG

            CubCallWrapper(mr).cubSelectFlagged(
                thrust::make_counting_iterator(0),
                d_selectionFlags,
                positions.begin(),
                d_numSelected.data(),
                entries,
                stream
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream)); //DEBUG

            int* numSelected = reinterpret_cast<int*>(hostTempStorage);
            CUDACHECK(cudaMemcpyAsync(numSelected, d_numSelected.data(), sizeof(int), D2H, stream));
            CUDACHECK(cudaStreamSynchronizeWrapper(stream));

            TaskData selection = gather(positions.begin(), positions.begin() + (*numSelected), stream, hostTempStorage);

            //std::cerr << "check selected\n";
            selection.consistencyCheck(stream);

            return selection;
        }

        template<class MapIter>
        TaskData gather(MapIter d_mapBegin, MapIter d_mapEnd, cudaStream_t stream, void* hostTempStorage){
            //std::cerr << "task " << this << " gather, stream " << stream << "\n";

            nvtx::ScopedRange sr("soa_gather", 2);

            auto gathersize = thrust::distance(d_mapBegin, d_mapEnd);

            TaskData selection(mr, gathersize, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

            
            auto inputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                pairedEnd.begin(),
                mateHasBeenFound.begin(),
                id.begin(),
                pairId.begin(),
                iteration.begin(),
                goodscore.begin(),
                myReadId.begin()
            ));

            auto outputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.pairedEnd.begin(),
                selection.mateHasBeenFound.begin(),
                selection.id.begin(),
                selection.pairId.begin(),
                selection.iteration.begin(),
                selection.goodscore.begin(),
                selection.myReadId.begin()
            ));

            thrust::gather(
                rmm::exec_policy_nosync(stream, mr),
                d_mapBegin,
                d_mapBegin + gathersize,
                inputScalars1Begin,
                outputScalars1Begin
            );

            auto inputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                mateReadId.begin(),
                abortReason.begin(),
                direction.begin(),
                soainputmateLengths.begin(),
                soainputAnchorLengths.begin()
            ));

            auto outputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.mateReadId.begin(),
                selection.abortReason.begin(),
                selection.direction.begin(),
                selection.soainputmateLengths.begin(),
                selection.soainputAnchorLengths.begin()
            ));

            thrust::gather(
                rmm::exec_policy_nosync(stream, mr),
                d_mapBegin,
                d_mapBegin + gathersize,
                inputScalars2Begin,
                outputScalars2Begin
            );    

            gatherSoaData(selection, d_mapBegin, d_mapEnd, stream, hostTempStorage);   

            //std::cerr << "check gather\n";

            selection.consistencyCheck(stream);

            return selection;
        }

        template<class MapIter>
        void gatherSoaData(TaskData& selection, MapIter d_mapBegin, MapIter d_mapEnd, cudaStream_t stream, void* hostTempStorage){
            ////std::cerr << "task " << this << " gatherSoaData, stream " << stream << "\n";
            assert(checkPitch(selection));

            auto gathersize = thrust::distance(d_mapBegin, d_mapEnd);

            selection.soaNumIterationResultsPerTask.resize(gathersize, stream);
            selection.soaNumIterationResultsPerTaskPrefixSum.resize(gathersize + 1, stream);

            selection.d_numUsedReadIdsPerTask.resize(gathersize, stream);
            selection.d_numUsedReadIdsPerTaskPrefixSum.resize(gathersize + 1, stream);

            //selection.d_numFullyUsedReadIdsPerTask.resize(gathersize, stream);
            //selection.d_numFullyUsedReadIdsPerTaskPrefixSum.resize(gathersize + 1, stream);

            selection.soainputdecodedMateRevC.resize(gathersize * decodedSequencePitchInBytes, stream);
            selection.soainputmateQualityScoresReversed.resize(gathersize * qualityPitchInBytes, stream);
            selection.soainputAnchorsDecoded.resize(gathersize * decodedSequencePitchInBytes, stream);
            selection.soainputAnchorQualities.resize(gathersize * qualityPitchInBytes, stream);

            selection.inputEncodedMate.resize(gathersize * encodedSequencePitchInInts, stream);
            selection.inputAnchorsEncoded.resize(gathersize * encodedSequencePitchInInts, stream);

            thrust::gather(
                rmm::exec_policy_nosync(stream, mr),
                d_mapBegin,
                d_mapBegin + gathersize,
                thrust::make_zip_iterator(thrust::make_tuple(
                    soaNumIterationResultsPerTask.begin(),
                    d_numUsedReadIdsPerTask.begin()
                    //d_numFullyUsedReadIdsPerTask.begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    selection.soaNumIterationResultsPerTask.begin(),
                    selection.d_numUsedReadIdsPerTask.begin()
                    //selection.d_numFullyUsedReadIdsPerTask.begin()
                ))
            );

            //Fused two scans
            CubCallWrapper(mr).cubInclusiveScan(
                thrust::make_zip_iterator(thrust::make_tuple(
                    selection.soaNumIterationResultsPerTask.begin(), 
                    selection.d_numUsedReadIdsPerTask.begin()
                    //selection.d_numFullyUsedReadIdsPerTask.begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    selection.soaNumIterationResultsPerTaskPrefixSum.begin() + 1, 
                    selection.d_numUsedReadIdsPerTaskPrefixSum.begin() + 1
                    //selection.d_numFullyUsedReadIdsPerTaskPrefixSum.begin() + 1
                )),
                ThrustTupleAddition<2>{}, // 3 with fully used
                gathersize,
                stream
            );

            //set first element of prefix sums to 0
            helpers::lambda_kernel<<<1,1,0,stream>>>([
                a = selection.soaNumIterationResultsPerTaskPrefixSum.begin(),
                b = selection.d_numUsedReadIdsPerTaskPrefixSum.begin()
                //c = selection.d_numFullyUsedReadIdsPerTaskPrefixSum.begin()
            ] __device__(){
                a[0] = 0;
                b[0] = 0;
                //c[0] = 0;
            }); CUDACHECKASYNC;

            if(gathersize > 0){

                int* selectedNumIterationResults = reinterpret_cast<int*>(hostTempStorage);
                int* selectedNumUsedIds = selectedNumIterationResults + 1;
                //int* selectedNumFullyUsedIds = selectedNumUsedIds + 1;
                CUDACHECK(cudaMemcpyAsync(selectedNumIterationResults, selection.soaNumIterationResultsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream));
                CUDACHECK(cudaMemcpyAsync(selectedNumUsedIds, selection.d_numUsedReadIdsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream));
               // CUDACHECK(cudaMemcpyAsync(selectedNumFullyUsedIds, selection.d_numFullyUsedReadIdsPerTaskPrefixSum.data() + gathersize, sizeof(int), D2H, stream));
                CUDACHECK(cudaStreamSynchronizeWrapper(stream));


                ::resizeUninitialized(selection.soatotalDecodedAnchorsLengths, *selectedNumIterationResults, stream);
                ::resizeUninitialized(selection.soatotalAnchorBeginInExtendedRead, *selectedNumIterationResults, stream);
                ::resizeUninitialized(selection.soatotalDecodedAnchorsFlat, *selectedNumIterationResults * decodedSequencePitchInBytes, stream);
                ::resizeUninitialized(selection.soatotalAnchorQualityScoresFlat, *selectedNumIterationResults * qualityPitchInBytes, stream);

                ::resizeUninitialized(selection.d_usedReadIds, *selectedNumUsedIds, stream);
                //::resizeUninitialized(selection.d_fullyUsedReadIds, *selectedNumFullyUsedIds, stream);
            }else{
                ::resizeUninitialized(selection.soatotalDecodedAnchorsLengths, 0, stream);
                ::resizeUninitialized(selection.soatotalAnchorBeginInExtendedRead, 0, stream);
                ::resizeUninitialized(selection.soatotalDecodedAnchorsFlat, 0 * decodedSequencePitchInBytes, stream);
                ::resizeUninitialized(selection.soatotalAnchorQualityScoresFlat, 0 * qualityPitchInBytes, stream);

                ::resizeUninitialized(selection.d_usedReadIds, 0, stream);
                //::resizeUninitialized(selection.d_fullyUsedReadIds, 0, stream);

                return;
            }

            readextendergpukernels::taskGatherKernel1<128, 32><<<gathersize, 128, 0, stream>>>(
                d_mapBegin,
                d_mapEnd,
                gathersize,
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                encodedSequencePitchInInts,
                selection.soainputAnchorLengths.data(),
                soainputAnchorLengths.data(),
                selection.soainputAnchorQualities.data(),
                soainputAnchorQualities.data(),
                selection.soainputmateQualityScoresReversed.data(),
                soainputmateQualityScoresReversed.data(),
                selection.soainputAnchorsDecoded.data(),
                soainputAnchorsDecoded.data(),
                selection.soainputdecodedMateRevC.data(),
                soainputdecodedMateRevC.data(),
                selection.inputEncodedMate.data(),
                inputEncodedMate.data(),
                selection.inputAnchorsEncoded.data(),
                inputAnchorsEncoded.data()
            ); CUDACHECKASYNC;


            // readextendergpukernels::taskGatherKernel2<128,32><<<gathersize, 128, 0, stream>>>(
            //     d_mapBegin,
            //     d_mapEnd,
            //     gathersize,
            //     decodedSequencePitchInBytes,
            //     qualityPitchInBytes,
            //     selection.soaNumIterationResultsPerTaskPrefixSum.data(),
            //     soaNumIterationResultsPerTaskPrefixSum.data(),
            //     soaNumIterationResultsPerTask.data(),
            //     selection.soatotalDecodedAnchorsLengths.data(),
            //     soatotalDecodedAnchorsLengths.data(),
            //     selection.soatotalAnchorBeginInExtendedRead.data(),
            //     soatotalAnchorBeginInExtendedRead.data(),
            //     selection.soatotalDecodedAnchorsFlat.data(),
            //     soatotalDecodedAnchorsFlat.data(),
            //     selection.soatotalAnchorQualityScoresFlat.data(),
            //     soatotalAnchorQualityScoresFlat.data(),
            //     selection.d_numUsedReadIdsPerTaskPrefixSum.data(),
            //     d_numUsedReadIdsPerTaskPrefixSum.data(),
            //     d_numUsedReadIdsPerTask.data(),
            //     selection.d_usedReadIds.data(),
            //     d_usedReadIds.data(),
            //     selection.d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
            //     d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
            //     d_numFullyUsedReadIdsPerTask.data(),
            //     selection.d_fullyUsedReadIds.data(),
            //     d_fullyUsedReadIds.data()
            // ); CUDACHECKASYNC;

            readextendergpukernels::taskGatherKernel2<128,32><<<gathersize, 128, 0, stream>>>(
                d_mapBegin,
                d_mapEnd,
                gathersize,
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                selection.soaNumIterationResultsPerTaskPrefixSum.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                soaNumIterationResultsPerTask.data(),
                selection.soatotalDecodedAnchorsLengths.data(),
                soatotalDecodedAnchorsLengths.data(),
                selection.soatotalAnchorBeginInExtendedRead.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                selection.soatotalDecodedAnchorsFlat.data(),
                soatotalDecodedAnchorsFlat.data(),
                selection.soatotalAnchorQualityScoresFlat.data(),
                soatotalAnchorQualityScoresFlat.data(),
                selection.d_numUsedReadIdsPerTaskPrefixSum.data(),
                d_numUsedReadIdsPerTaskPrefixSum.data(),
                d_numUsedReadIdsPerTask.data(),
                selection.d_usedReadIds.data(),
                d_usedReadIds.data()
            ); CUDACHECKASYNC;

        }

        void addScalarIterationResultData(
            const float* d_goodscores,
            const extension::AbortReason* d_abortReasons,
            const bool* d_mateHasBeenFound,
            cudaStream_t stream
        ){
            //std::cerr << "task " << this << " addScalarIterationResultData, stream " << stream << "\n";
            if(size() > 0){
                readextendergpukernels::taskUpdateScalarIterationResultsKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                    size(),
                    goodscore.data(),
                    abortReason.data(),
                    mateHasBeenFound.data(),
                    d_goodscores,
                    d_abortReasons,
                    d_mateHasBeenFound
                ); CUDACHECKASYNC;
            }
        }

        
        void addSoAIterationResultData(
            const int* d_addNumEntriesPerTask,
            const int* d_addNumEntriesPerTaskPrefixSum,
            const int* d_addTotalDecodedAnchorsLengths,
            const int* d_addTotalAnchorBeginInExtendedRead,
            const char* d_addTotalDecodedAnchorsFlat,
            const char* d_addTotalAnchorQualityScoresFlat,
            std::size_t addSequencesPitchInBytes,
            std::size_t addQualityPitchInBytes,
            cudaStream_t stream,
            void* hostTempStorage
        ){
            //std::cerr << "task " << this << " addSoAIterationResultData, stream " << stream << "\n";
            if(size() == 0) return;

            rmm::device_uvector<int> newNumEntriesPerTask(size(), stream, mr);
            rmm::device_uvector<int> newNumEntriesPerTaskPrefixSum(size() + 1, stream, mr);

            readextendergpukernels::vectorAddKernel<<<SDIV(size(), 128), 128, 0, stream>>>(
                d_addNumEntriesPerTask, 
                soaNumIterationResultsPerTask.begin(), 
                newNumEntriesPerTask.begin(), 
                size()
            ); CUDACHECKASYNC;

            CubCallWrapper(mr).cubInclusiveSum(
                newNumEntriesPerTask.begin(),
                newNumEntriesPerTaskPrefixSum.begin() + 1,
                size(),
                stream
            );

            cudaMemsetAsync(
                newNumEntriesPerTaskPrefixSum.data(),
                0, 
                sizeof(int),
                stream
            ); CUDACHECKASYNC;

            std::size_t newirregularsize = 0;
            if(size() > 0){
                int* result = reinterpret_cast<int*>(hostTempStorage);
                CUDACHECK(cudaMemcpyAsync(result, newNumEntriesPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
                CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                newirregularsize = *result;
            }

            rmm::device_uvector<int> newsoatotalDecodedAnchorsLengths(newirregularsize, stream, mr);
            rmm::device_uvector<int> newsoatotalAnchorBeginInExtendedRead(newirregularsize, stream, mr);
            rmm::device_uvector<char> newsoatotalDecodedAnchorsFlat(newirregularsize * decodedSequencePitchInBytes, stream, mr);
            rmm::device_uvector<char> newsoatotalAnchorQualityScoresFlat(newirregularsize * qualityPitchInBytes, stream, mr);

            readextendergpukernels::taskAddIterationResultsKernel<128,32><<<size(), 128, 0, stream>>>(
                size(),
                decodedSequencePitchInBytes,
                qualityPitchInBytes,
                addSequencesPitchInBytes,
                addQualityPitchInBytes,
                newNumEntriesPerTaskPrefixSum.data(),
                newsoatotalDecodedAnchorsFlat.data(),
                newsoatotalAnchorQualityScoresFlat.data(),
                newsoatotalDecodedAnchorsLengths.data(),
                newsoatotalAnchorBeginInExtendedRead.data(),
                soaNumIterationResultsPerTask.data(),
                soaNumIterationResultsPerTaskPrefixSum.data(),
                soatotalDecodedAnchorsLengths.data(),
                soatotalAnchorBeginInExtendedRead.data(),
                soatotalDecodedAnchorsFlat.data(),
                soatotalAnchorQualityScoresFlat.data(),
                d_addNumEntriesPerTask,
                d_addNumEntriesPerTaskPrefixSum,
                d_addTotalDecodedAnchorsLengths,
                d_addTotalAnchorBeginInExtendedRead,
                d_addTotalDecodedAnchorsFlat,
                d_addTotalAnchorQualityScoresFlat
            ); CUDACHECKASYNC;

            std::swap(soaNumIterationResultsPerTask, newNumEntriesPerTask);
            std::swap(soaNumIterationResultsPerTaskPrefixSum, newNumEntriesPerTaskPrefixSum);
            std::swap(soatotalDecodedAnchorsLengths, newsoatotalDecodedAnchorsLengths);
            std::swap(soatotalAnchorBeginInExtendedRead, newsoatotalAnchorBeginInExtendedRead);
            std::swap(soatotalDecodedAnchorsFlat, newsoatotalDecodedAnchorsFlat);
            std::swap(soatotalAnchorQualityScoresFlat, newsoatotalAnchorQualityScoresFlat);

            consistencyCheck(stream);

        }

        // void updateUsedReadIdsAndFullyUsedReadIds(
        //     const read_number* d_candidateReadIds,
        //     const int* d_numCandidatesPerAnchor,
        //     const int* d_numCandidatesPerAnchorPrefixSum,
        //     const bool* d_isFullyUsedId,
        //     int numCandidateIds,
        //     cudaStream_t stream,
        //     void* hostTempStorage
        // ){
        //     //std::cerr << "task " << this << " updateUsedReadIdsAndFullyUsedReadIds, stream " << stream << "\n";

        //     int* const tmpptr1 = reinterpret_cast<int*>(hostTempStorage);
        //     int* const tmpptr2 = tmpptr1 + 1;

        //     int numUsedIds = 0;
        //     int numFullyUsedIds = 0;
        //     CUDACHECK(cudaMemcpyAsync(tmpptr1, d_numUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
        //     CUDACHECK(cudaMemcpyAsync(tmpptr2, d_numFullyUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
        //     CUDACHECK(cudaStreamSynchronizeWrapper(stream));
        //     numUsedIds = *tmpptr1;
        //     numFullyUsedIds = *tmpptr2;

        //     if(numUsedIds != int(d_usedReadIds.size())){
        //         std::cerr << "numUsedIds " << numUsedIds << ", d_usedReadIds.size() " << d_usedReadIds.size() << "\n";
        //     }

        //     if(numFullyUsedIds != int(d_fullyUsedReadIds.size())){
        //         std::cerr << "numFullyUsedIds " << numFullyUsedIds << ", d_fullyUsedReadIds.size() " << d_fullyUsedReadIds.size() << "\n";
        //     }

        //     assert(numUsedIds == int(d_usedReadIds.size()));
        //     //assert(numFullyUsedIds == int(d_fullyUsedReadIds.size()));

        //     const int maxoutputsize = numCandidateIds + numUsedIds;

        //     rmm::device_uvector<read_number> d_newUsedReadIds(maxoutputsize, stream, mr);
        //     rmm::device_uvector<int> d_newnumUsedReadIdsPerTask(size(), stream, mr);

        //     auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
        //         d_candidateReadIds,
        //         d_numCandidatesPerAnchor,
        //         d_numCandidatesPerAnchorPrefixSum,
        //         numCandidateIds,
        //         size(),
        //         d_usedReadIds.data(),
        //         d_numUsedReadIdsPerTask.data(),
        //         d_numUsedReadIdsPerTaskPrefixSum.data(),
        //         numUsedIds,
        //         size(),   
        //         d_newUsedReadIds.data(),
        //         d_newnumUsedReadIdsPerTask.data(),
        //         size(),
        //         stream,
        //         mr
        //     );

        //     const int newNumUsedIds = std::distance(d_newUsedReadIds.data(), d_newUsedReadIds_end);

        //     ::erase(d_newUsedReadIds, d_newUsedReadIds.begin() + newNumUsedIds, d_newUsedReadIds.end(), stream);
        //     std::swap(d_usedReadIds, d_newUsedReadIds);
        //     std::swap(d_numUsedReadIdsPerTask, d_newnumUsedReadIdsPerTask);

        //     destroy(d_newUsedReadIds, stream);
        //     destroy(d_newnumUsedReadIdsPerTask, stream);

        //     rmm::device_uvector<read_number> d_currentFullyUsedReadIds(numCandidateIds, stream, mr);
        //     rmm::device_uvector<int> d_currentNumFullyUsedreadIdsPerAnchor(size(), stream, mr);
        //     rmm::device_uvector<int> d_currentNumFullyUsedreadIdsPerAnchorPS(size(), stream, mr);
        //     rmm::device_scalar<int> d_addNumFullyUsed(stream, mr);
            
        //     //make compact list of current fully used candidates
        //     CubCallWrapper(mr).cubSelectFlagged(
        //         d_candidateReadIds,
        //         d_isFullyUsedId,
        //         d_currentFullyUsedReadIds.data(),
        //         d_addNumFullyUsed.data(),
        //         numCandidateIds,
        //         stream
        //     );

        //     //compute current number of fully used candidates per segment
        //     CubCallWrapper(mr).cubSegmentedReduceSum(
        //         d_isFullyUsedId,
        //         d_currentNumFullyUsedreadIdsPerAnchor.data(),
        //         size(),
        //         d_numCandidatesPerAnchorPrefixSum,
        //         d_numCandidatesPerAnchorPrefixSum + 1,
        //         stream
        //     );

        //     //compute prefix sum of current number of fully used candidates per segment

        //     CubCallWrapper(mr).cubExclusiveSum(
        //         d_currentNumFullyUsedreadIdsPerAnchor.data(), 
        //         d_currentNumFullyUsedreadIdsPerAnchorPS.data(), 
        //         size(),
        //         stream
        //     );

        //     int addNumFullyUsed = 0;
        //     CUDACHECK(cudaMemcpyAsync(tmpptr1, d_addNumFullyUsed.data(), sizeof(int), D2H, stream));
        //     CUDACHECK(cudaStreamSynchronizeWrapper(stream));
        //     addNumFullyUsed = *tmpptr1;

        //     const int maxoutputsize2 = addNumFullyUsed + numFullyUsedIds;

        //     rmm::device_uvector<read_number> d_newFullyUsedReadIds(maxoutputsize2, stream, mr);
        //     rmm::device_uvector<int> d_newNumFullyUsedreadIdsPerAnchor(size(), stream, mr);

        //     auto d_newFullyUsedReadIds_end = GpuSegmentedSetOperation::set_union(
        //         d_currentFullyUsedReadIds.data(),
        //         d_currentNumFullyUsedreadIdsPerAnchor.data(),
        //         d_currentNumFullyUsedreadIdsPerAnchorPS.data(),
        //         addNumFullyUsed,
        //         size(),
        //         d_fullyUsedReadIds.data(),
        //         d_numFullyUsedReadIdsPerTask.data(),
        //         d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
        //         numFullyUsedIds,
        //         size(),       
        //         d_newFullyUsedReadIds.data(),
        //         d_newNumFullyUsedreadIdsPerAnchor.data(),
        //         size(),
        //         stream,
        //         mr
        //     );

        //     const int newNumFullyUsedIds = std::distance(d_newFullyUsedReadIds.data(), d_newFullyUsedReadIds_end);

        //     ::erase(d_newFullyUsedReadIds, d_newFullyUsedReadIds.begin() + newNumFullyUsedIds, d_newFullyUsedReadIds.end(), stream);

        //     std::swap(d_fullyUsedReadIds, d_newFullyUsedReadIds);
        //     std::swap(d_numFullyUsedReadIdsPerTask, d_newNumFullyUsedreadIdsPerAnchor);

        //     destroy(d_newFullyUsedReadIds, stream);
        //     destroy(d_newNumFullyUsedreadIdsPerAnchor, stream);


        //     //merged two prefix sums of relatively small arrays into single cub call

        //     CubCallWrapper(mr).cubInclusiveScan(
        //         thrust::make_zip_iterator(thrust::make_tuple(
        //             d_numFullyUsedReadIdsPerTask.data(), 
        //             d_numUsedReadIdsPerTask.data()
        //         )),
        //         thrust::make_zip_iterator(thrust::make_tuple(
        //             d_numFullyUsedReadIdsPerTaskPrefixSum.data() + 1, 
        //             d_numUsedReadIdsPerTaskPrefixSum.data() + 1
        //         )),
        //         ThrustTupleAddition<2>{},
        //         size(),
        //         stream
        //     );

        //     //std::cerr << "after update used\n";
        //     consistencyCheck(stream);
        // }

        void updateUsedReadIds(
            const read_number* d_candidateReadIds,
            const int* d_numCandidatesPerAnchor,
            const int* d_numCandidatesPerAnchorPrefixSum,
            int numCandidateIds,
            cudaStream_t stream,
            void* hostTempStorage
        ){
            //std::cerr << "task " << this << " updateUsedReadIdsAndFullyUsedReadIds, stream " << stream << "\n";

            int* const tmpptr1 = reinterpret_cast<int*>(hostTempStorage);

            int numUsedIds = 0;
            CUDACHECK(cudaMemcpyAsync(tmpptr1, d_numUsedReadIdsPerTaskPrefixSum.data() + size(), sizeof(int), D2H, stream));
            CUDACHECK(cudaStreamSynchronizeWrapper(stream));
            numUsedIds = *tmpptr1;

            if(numUsedIds != int(d_usedReadIds.size())){
                std::cerr << "numUsedIds " << numUsedIds << ", d_usedReadIds.size() " << d_usedReadIds.size() << "\n";
            }

            assert(numUsedIds == int(d_usedReadIds.size()));

            const int maxoutputsize = numCandidateIds + numUsedIds;

            rmm::device_uvector<read_number> d_newUsedReadIds(maxoutputsize, stream, mr);
            rmm::device_uvector<int> d_newnumUsedReadIdsPerTask(size(), stream, mr);

            auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                d_candidateReadIds,
                d_numCandidatesPerAnchor,
                d_numCandidatesPerAnchorPrefixSum,
                numCandidateIds,
                size(),
                d_usedReadIds.data(),
                d_numUsedReadIdsPerTask.data(),
                d_numUsedReadIdsPerTaskPrefixSum.data(),
                numUsedIds,
                size(),   
                d_newUsedReadIds.data(),
                d_newnumUsedReadIdsPerTask.data(),
                size(),
                stream,
                mr
            );

            const int newNumUsedIds = std::distance(d_newUsedReadIds.data(), d_newUsedReadIds_end);

            ::erase(d_newUsedReadIds, d_newUsedReadIds.begin() + newNumUsedIds, d_newUsedReadIds.end(), stream);
            std::swap(d_usedReadIds, d_newUsedReadIds);
            std::swap(d_numUsedReadIdsPerTask, d_newnumUsedReadIdsPerTask);

            destroy(d_newUsedReadIds, stream);
            destroy(d_newnumUsedReadIdsPerTask, stream);

            CubCallWrapper(mr).cubInclusiveSum(
                d_numUsedReadIdsPerTask.data(),
                d_numUsedReadIdsPerTaskPrefixSum.data() + 1,
                size(),
                stream
            );

            //std::cerr << "after update used\n";
            consistencyCheck(stream);
        }

        void getActiveFlags(bool* d_flags, int minFragmentSize, int maxFragmentSize, cudaStream_t stream) const{
            //std::cerr << "task " << this << " getActiveFlags, stream " << stream << "\n";
            if(size() > 0){
                readextendergpukernels::taskComputeActiveFlagsKernel<128><<<SDIV(size(), 128), 128, 0, stream>>>(
                    size(),
                    minFragmentSize,
                    maxFragmentSize,
                    d_flags,
                    iteration.data(),
                    soaNumIterationResultsPerTask.data(),
                    soaNumIterationResultsPerTaskPrefixSum.data(),
                    soatotalAnchorBeginInExtendedRead.data(),
                    soainputmateLengths.data(),
                    abortReason.data(),
                    mateHasBeenFound.data()
                ); CUDACHECKASYNC;

                consistencyCheck(stream);
            }
        }

    };




    struct Hasher{
        const gpu::GpuMinhasher* gpuMinhasher{};
        mutable MinhasherHandle minhashHandle{};
        helpers::SimpleAllocationPinnedHost<int> h_numCandidates{};
        rmm::mr::device_memory_resource* mr{};

        Hasher(const gpu::GpuMinhasher& gpuMinhasher_, rmm::mr::device_memory_resource* mr_)
            : gpuMinhasher(&gpuMinhasher_),
            minhashHandle(gpuMinhasher_.makeMinhasherHandle()),
            mr(mr_){

            h_numCandidates.resize(1);
        }

        ~Hasher(){
            gpuMinhasher->destroyHandle(minhashHandle);
        }

        void getCandidateReadIds(const AnchorData& anchorData, AnchorHashResult& results, cudaStream_t stream){
            const int numAnchors = anchorData.d_anchorSequencesLength.size();

            ::resizeUninitialized(results.d_numCandidatesPerAnchor, numAnchors, stream);
            ::resizeUninitialized(results.d_numCandidatesPerAnchorPrefixSum, numAnchors + 1, stream);

            int totalNumValues = 0;

            DEBUGSTREAMSYNC(stream);

            gpuMinhasher->determineNumValues(
                minhashHandle,
                anchorData.d_anchorSequencesData.data(),
                anchorData.encodedSequencePitchInInts,
                anchorData.d_anchorSequencesLength.data(),
                numAnchors,
                results.d_numCandidatesPerAnchor.data(),
                totalNumValues,
                stream,
                mr
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));

            ::resizeUninitialized(results.d_candidateReadIds, totalNumValues, stream);    

            if(totalNumValues == 0){
                *h_numCandidates = 0;
                CUDACHECK(cudaMemsetAsync(results.d_numCandidatesPerAnchor.data(), 0, sizeof(int) * numAnchors , stream));
                CUDACHECK(cudaMemsetAsync(results.d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int) * (1 + numAnchors), stream));

                DEBUGSTREAMSYNC(stream);
            }else{

                DEBUGSTREAMSYNC(stream);

                gpuMinhasher->retrieveValues(
                    minhashHandle,
                    numAnchors,              
                    totalNumValues,
                    results.d_candidateReadIds.data(),
                    results.d_numCandidatesPerAnchor.data(),
                    results.d_numCandidatesPerAnchorPrefixSum.data(),
                    stream,
                    mr
                );

                rmm::device_uvector<read_number> d_candidate_read_ids2(totalNumValues, stream, mr);
                rmm::device_uvector<int> d_candidates_per_anchor2(numAnchors, stream, mr);
                rmm::device_uvector<int> d_candidates_per_anchor_prefixsum2(1 + numAnchors, stream, mr);

                cub::DoubleBuffer<read_number> d_items{results.d_candidateReadIds.data(), d_candidate_read_ids2.data()};
                cub::DoubleBuffer<int> d_numItemsPerSegment{results.d_numCandidatesPerAnchor.data(), d_candidates_per_anchor2.data()};
                cub::DoubleBuffer<int> d_numItemsPerSegmentPrefixSum{results.d_numCandidatesPerAnchorPrefixSum.data(), d_candidates_per_anchor_prefixsum2.data()};

                GpuMinhashQueryFilter::keepDistinct(
                    d_items,
                    d_numItemsPerSegment,
                    d_numItemsPerSegmentPrefixSum, //numSegments + 1
                    numAnchors,
                    totalNumValues,
                    stream,
                    mr
                );

                if(d_items.Current() != results.d_candidateReadIds.data()){
                    //std::cerr << "swap d_candidate_read_ids\n";
                    std::swap(results.d_candidateReadIds, d_candidate_read_ids2);
                }
                if(d_numItemsPerSegment.Current() != results.d_numCandidatesPerAnchor.data()){
                    //std::cerr << "swap d_candidates_per_anchor\n";
                    std::swap(results.d_numCandidatesPerAnchor, d_candidates_per_anchor2);
                }
                if(d_numItemsPerSegmentPrefixSum.Current() != results.d_numCandidatesPerAnchorPrefixSum.data()){
                    //std::cerr << "swap d_candidates_per_anchor_prefixsum\n";
                    std::swap(results.d_numCandidatesPerAnchorPrefixSum, d_candidates_per_anchor_prefixsum2);
                }

                CUDACHECK(cudaMemcpyAsync(
                    h_numCandidates.data(),
                    results.d_numCandidatesPerAnchorPrefixSum.data() + numAnchors,
                    sizeof(int),
                    D2H,
                    stream
                ));

                CUDACHECK(cudaStreamSynchronizeWrapper(stream));

                ::erase(results.d_candidateReadIds, results.d_candidateReadIds.begin() + (*h_numCandidates), results.d_candidateReadIds.end(), stream);
            }

            DEBUGSTREAMSYNC(stream);
        }
    
        template<class Flags>
        void getCandidateReadIdsWithExtraExtensionHash(
            const AnchorData& anchorData, 
            AnchorHashResult& results, 
            const IterationConfig& iterationConfig, 
            Flags d_extraFlags,
            cudaStream_t stream
        ){
            //std::lock_guard<std::mutex> lg(someGlobalMutex2);

            //Compute the extra sequences
            const int numAnchors = anchorData.d_anchorSequencesLength.size();
            const int kmersize = gpuMinhasher->getKmerSize();
            const int extralength = SDIV(iterationConfig.maxextensionPerStep + kmersize - 1, 4) * 4; //rounded up to multiple of 4
            assert(extralength > 0);
            const std::size_t extraDecodedPitch = SDIV(extralength, 256) * 256; //rounded up to multiple of 256            

            rmm::device_uvector<char> extraDecodedSequences(extraDecodedPitch * numAnchors, stream, mr);
            rmm::device_uvector<int> extraSequenceLengths(numAnchors, stream, mr);

            DEBUGSTREAMSYNC(stream);

            readextendergpukernels::copyLastNCharactersOfStrings<256>
            <<<SDIV(numAnchors, (256 / 32)), 256, 0, stream>>>(
                numAnchors,
                anchorData.d_anchorSequencesDataDecoded.data(),
                anchorData.d_anchorSequencesLength.data(),
                anchorData.decodedSequencePitchInBytes,
                extraDecodedSequences.data(),
                extraSequenceLengths.data(),
                extraDecodedPitch,
                d_extraFlags,
                extralength
            );

            DEBUGSTREAMSYNC(stream);

            const std::size_t extraEncodedPitchInInts = SequenceHelpers::getEncodedNumInts2Bit(extralength);

            rmm::device_uvector<unsigned int> extraEncodedSequences(extraEncodedPitchInInts * numAnchors, stream, mr);

            callEncodeSequencesTo2BitKernel(
                extraEncodedSequences.data(),
                extraDecodedSequences.data(),
                extraSequenceLengths.data(),
                extraDecodedPitch,
                extraEncodedPitchInInts,
                numAnchors,
                8,
                stream
            );

            destroy(extraDecodedSequences, stream);

            //hash extra sequences
            rmm::device_uvector<int> d_numCandidatesPerAnchorExtra(numAnchors, stream, mr);
            
            int totalNumValuesExtraSequences = 0;

            DEBUGSTREAMSYNC(stream);

            gpuMinhasher->determineNumValues(
                minhashHandle,
                extraEncodedSequences.data(),
                extraEncodedPitchInInts,
                extraSequenceLengths.data(),
                numAnchors,
                d_numCandidatesPerAnchorExtra.data(),
                totalNumValuesExtraSequences,
                stream,
                mr
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));

            destroy(extraEncodedSequences, stream);
            destroy(extraSequenceLengths, stream);

            rmm::device_uvector<read_number> d_candidateReadIdsExtra(totalNumValuesExtraSequences, stream, mr);
            rmm::device_uvector<int> d_numCandidatesPerAnchorPrefixSumExtra(numAnchors + 1, stream, mr);

            if(totalNumValuesExtraSequences == 0){
                CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorExtra.data(), 0, sizeof(int) * numAnchors , stream));
                CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSumExtra.data(), 0, sizeof(int) * (1 + numAnchors), stream));

                DEBUGSTREAMSYNC(stream);

            }else{

                DEBUGSTREAMSYNC(stream);

                gpuMinhasher->retrieveValues(
                    minhashHandle,
                    numAnchors,              
                    totalNumValuesExtraSequences,
                    d_candidateReadIdsExtra.data(),
                    d_numCandidatesPerAnchorExtra.data(),
                    d_numCandidatesPerAnchorPrefixSumExtra.data(),
                    stream,
                    mr
                );

                DEBUGSTREAMSYNC(stream);

                {
                    rmm::device_uvector<read_number> d_candidate_read_ids2(totalNumValuesExtraSequences, stream, mr);
                    rmm::device_uvector<int> d_candidates_per_anchor2(numAnchors, stream, mr);
                    rmm::device_uvector<int> d_candidates_per_anchor_prefixsum2(1 + numAnchors, stream, mr);

                    cub::DoubleBuffer<read_number> d_items{d_candidateReadIdsExtra.data(), d_candidate_read_ids2.data()};
                    cub::DoubleBuffer<int> d_numItemsPerSegment{d_numCandidatesPerAnchorExtra.data(), d_candidates_per_anchor2.data()};
                    cub::DoubleBuffer<int> d_numItemsPerSegmentPrefixSum{d_numCandidatesPerAnchorPrefixSumExtra.data(), d_candidates_per_anchor_prefixsum2.data()};

                    GpuMinhashQueryFilter::keepDistinct(
                        d_items,
                        d_numItemsPerSegment,
                        d_numItemsPerSegmentPrefixSum, //numSegments + 1
                        numAnchors,
                        totalNumValuesExtraSequences,
                        stream,
                        mr
                    );

                    if(d_items.Current() != d_candidateReadIdsExtra.data()){
                        //std::cerr << "swap d_candidate_read_ids\n";
                        std::swap(d_candidateReadIdsExtra, d_candidate_read_ids2);
                    }
                    if(d_numItemsPerSegment.Current() != d_numCandidatesPerAnchorExtra.data()){
                        //std::cerr << "swap d_candidates_per_anchor\n";
                        std::swap(d_numCandidatesPerAnchorExtra, d_candidates_per_anchor2);
                    }
                    if(d_numItemsPerSegmentPrefixSum.Current() != d_numCandidatesPerAnchorPrefixSumExtra.data()){
                        //std::cerr << "swap d_candidates_per_anchor_prefixsum\n";
                        std::swap(d_numCandidatesPerAnchorPrefixSumExtra, d_candidates_per_anchor_prefixsum2);
                    }
                }


                CUDACHECK(cudaMemcpyAsync(
                    h_numCandidates.data(),
                    d_numCandidatesPerAnchorPrefixSumExtra.data() + numAnchors,
                    sizeof(int),
                    D2H,
                    stream
                ));

                CUDACHECK(cudaStreamSynchronizeWrapper(stream));
                totalNumValuesExtraSequences = *h_numCandidates;
            }

            //hash anchor sequences
            rmm::device_uvector<int> d_numCandidatesPerAnchor(numAnchors, stream, mr);
            
            int totalNumValuesAnchorSequences = 0;

            DEBUGSTREAMSYNC(stream);

            gpuMinhasher->determineNumValues(
                minhashHandle,
                anchorData.d_anchorSequencesData.data(),
                anchorData.encodedSequencePitchInInts,
                anchorData.d_anchorSequencesLength.data(),
                numAnchors,
                d_numCandidatesPerAnchor.data(),
                totalNumValuesAnchorSequences,
                stream,
                mr
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));

            rmm::device_uvector<read_number> d_candidateReadIds(totalNumValuesAnchorSequences, stream, mr);
            rmm::device_uvector<int> d_numCandidatesPerAnchorPrefixSum(numAnchors + 1, stream, mr);

            if(totalNumValuesAnchorSequences == 0){
                CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchor.data(), 0, sizeof(int) * numAnchors , stream));
                CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int) * (1 + numAnchors), stream));

                DEBUGSTREAMSYNC(stream);

            }else{

                DEBUGSTREAMSYNC(stream);

                gpuMinhasher->retrieveValues(
                    minhashHandle,
                    numAnchors,              
                    totalNumValuesAnchorSequences,
                    d_candidateReadIds.data(),
                    d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum.data(),
                    stream,
                    mr
                );

                DEBUGSTREAMSYNC(stream);

                {
                    rmm::device_uvector<read_number> d_candidate_read_ids2(totalNumValuesAnchorSequences, stream, mr);
                    rmm::device_uvector<int> d_candidates_per_anchor2(numAnchors, stream, mr);
                    rmm::device_uvector<int> d_candidates_per_anchor_prefixsum2(1 + numAnchors, stream, mr);

                    cub::DoubleBuffer<read_number> d_items{d_candidateReadIds.data(), d_candidate_read_ids2.data()};
                    cub::DoubleBuffer<int> d_numItemsPerSegment{d_numCandidatesPerAnchor.data(), d_candidates_per_anchor2.data()};
                    cub::DoubleBuffer<int> d_numItemsPerSegmentPrefixSum{d_numCandidatesPerAnchorPrefixSum.data(), d_candidates_per_anchor_prefixsum2.data()};

                    GpuMinhashQueryFilter::keepDistinct(
                        d_items,
                        d_numItemsPerSegment,
                        d_numItemsPerSegmentPrefixSum, //numSegments + 1
                        numAnchors,
                        totalNumValuesAnchorSequences,
                        stream,
                        mr
                    );

                    if(d_items.Current() != d_candidateReadIds.data()){
                        //std::cerr << "swap d_candidate_read_ids\n";
                        std::swap(d_candidateReadIds, d_candidate_read_ids2);
                    }
                    if(d_numItemsPerSegment.Current() != d_numCandidatesPerAnchor.data()){
                        //std::cerr << "swap d_candidates_per_anchor\n";
                        std::swap(d_numCandidatesPerAnchor, d_candidates_per_anchor2);
                    }
                    if(d_numItemsPerSegmentPrefixSum.Current() != d_numCandidatesPerAnchorPrefixSum.data()){
                        //std::cerr << "swap d_candidates_per_anchor_prefixsum\n";
                        std::swap(d_numCandidatesPerAnchorPrefixSum, d_candidates_per_anchor_prefixsum2);
                    }
                }


                CUDACHECK(cudaMemcpyAsync(
                    h_numCandidates.data(),
                    d_numCandidatesPerAnchorPrefixSum.data() + numAnchors,
                    sizeof(int),
                    D2H,
                    stream
                ));

                CUDACHECK(cudaStreamSynchronizeWrapper(stream));
                totalNumValuesAnchorSequences = *h_numCandidates;
            }

            //for each anchor compute set_union with extra candidates

            DEBUGSTREAMSYNC(stream);

            ::resizeUninitialized(results.d_numCandidatesPerAnchor, numAnchors, stream);
            ::resizeUninitialized(results.d_candidateReadIds, totalNumValuesAnchorSequences + totalNumValuesExtraSequences, stream);
            DEBUGSTREAMSYNC(stream);

            int deviceId = 0;
            CUDACHECK(cudaGetDevice(&deviceId));
            
            rmm::mr::thrust_allocator<char> thrustCachingAllocator(stream, mr);

            DEBUGSTREAMSYNC(stream);

            auto newend = GpuSegmentedSetOperation::set_union(
                d_candidateReadIds.data(),
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                totalNumValuesAnchorSequences,
                numAnchors,
                d_candidateReadIdsExtra.data(),
                d_numCandidatesPerAnchorExtra.data(),
                d_numCandidatesPerAnchorPrefixSumExtra.data(),
                totalNumValuesExtraSequences,
                numAnchors,
                results.d_candidateReadIds.data(),
                results.d_numCandidatesPerAnchor.data(),
                numAnchors,
                stream,
                mr
            );

            assert(newend <= results.d_candidateReadIds.end());

            DEBUGSTREAMSYNC(stream);

            ::erase(results.d_candidateReadIds, newend, results.d_candidateReadIds.end(), stream);

            DEBUGSTREAMSYNC(stream);

            destroy(d_candidateReadIdsExtra, stream);
            destroy(d_numCandidatesPerAnchorExtra, stream);
            destroy(d_numCandidatesPerAnchorPrefixSumExtra, stream);

            destroy(d_candidateReadIds, stream);
            destroy(d_numCandidatesPerAnchor, stream);
            destroy(d_numCandidatesPerAnchorPrefixSum, stream);

            ::resizeUninitialized(results.d_numCandidatesPerAnchorPrefixSum, numAnchors + 1, stream);

            DEBUGSTREAMSYNC(stream);


            CUDACHECK(cudaMemsetAsync(results.d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream));

            DEBUGSTREAMSYNC(stream);


            CubCallWrapper(mr).cubInclusiveSum(
                results.d_numCandidatesPerAnchor.data(),
                results.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                numAnchors,
                stream
            );

            DEBUGSTREAMSYNC(stream);
        }
    };


    enum class State{
        UpdateWorkingSet,
        BeforeHash,
        BeforeRemoveIds,
        BeforeComputePairFlags,
        BeforeLoadCandidates,
        BeforeEraseData,
        BeforeAlignment,
        BeforeAlignmentFilter,
        BeforeMSA,
        BeforeExtend,
        BeforePrepareNextIteration,
        Finished,
        None
    };

    static std::string to_string(State s){
        switch(s){
            case State::UpdateWorkingSet: return "UpdateWorkingSet";
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeRemoveIds: return "BeforeRemoveIds";
            case State::BeforeComputePairFlags: return "BeforeComputePairFlags";
            case State::BeforeLoadCandidates: return "BeforeLoadCandidates";
            case State::BeforeEraseData: return "BeforeEraseData";
            case State::BeforeAlignment: return "BeforeAlignment";
            case State::BeforeAlignmentFilter: return "BeforeAlignmentFilter";
            case State::BeforeMSA: return "BeforeMSA";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforePrepareNextIteration: return "BeforePrepareNextIteration";
            case State::Finished: return "Finished";
            case State::None: return "None";
            default: return "Missing case GpuReadExtender::to_string(State)\n";
        };
    }

    bool isEmpty() const noexcept{
        return tasks->size() == 0;
    }

    void setState(State newstate){      
        if(false){
            std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate);
            std::cerr << "\n";
        }

        state = newstate;
    }

    GpuReadExtender(
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        std::size_t msaColumnPitchInElements_,
        bool isPairedEnd_,
        const gpu::GpuReadStorage& rs, 
        const ProgramOptions& programOptions_,
        const cpu::QualityScoreConversion& qualityConversion_,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr_
    ) : 
        pairedEnd(isPairedEnd_),
        mr(mr_),
        gpuReadStorage(&rs),
        programOptions(&programOptions_),
        qualityConversion(&qualityConversion_),
        readStorageHandle(gpuReadStorage->makeHandle()),
        d_mateIdHasBeenRemoved(0, stream, mr),
        d_candidateSequencesData(0, stream, mr),
        d_candidateSequencesLength(0, stream, mr),    
        d_candidateReadIds(0, stream, mr),
        d_isPairedCandidate(0, stream, mr),
        d_alignment_overlaps(0, stream, mr),
        d_alignment_shifts(0, stream, mr),
        d_alignment_nOps(0, stream, mr),
        d_alignment_best_alignment_flags(0, stream, mr),
        d_numCandidatesPerAnchor(0, stream, mr),
        d_numCandidatesPerAnchorPrefixSum(0, stream, mr),
        d_anchorSequencesDataDecoded(0, stream, mr),
        d_anchorQualityScores(0, stream, mr),
        d_anchorSequencesLength(0, stream, mr),
        d_anchorSequencesData(0, stream, mr),
        d_accumExtensionsLengths(0, stream, mr),
        multiMSA(stream, mr)
    {

        CUDACHECK(cudaGetDevice(&deviceId));

        h_numAnchors.resize(1);
        h_numCandidates.resize(1);
        h_numAnchorsWithRemovedMates.resize(1);

        h_minmax.resize(2);
        h_numPositions.resize(2);


        *h_numAnchors = 0;
        *h_numCandidates = 0;
        *h_numAnchorsWithRemovedMates = 0;

        encodedSequencePitchInInts = encodedSequencePitchInInts_;
        decodedSequencePitchInBytes = decodedSequencePitchInBytes_;
        qualityPitchInBytes = qualityPitchInBytes_;
        msaColumnPitchInElements = msaColumnPitchInElements_;

        CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }

    void processOneIteration(
        TaskData& inputTasks, 
        AnchorData& currentAnchorData, 
        AnchorHashResult& currentHashResults, 
        TaskData& outputFinishedTasks, 
        const IterationConfig& iterationConfig_,
        cudaStream_t stream
    ){
        std::lock_guard<std::mutex> lockguard(mutex);
        //std::cerr << "processOneIteration enter thread " << std::this_thread::get_id() << "\n";


        if(inputTasks.size() > 0){

            tasks = &inputTasks;
            initialNumCandidates = currentHashResults.d_candidateReadIds.size();
            iterationConfig = &iterationConfig_;

            DEBUGSTREAMSYNC(stream);

            copyAnchorDataFrom(currentAnchorData, stream);
            copyHashResultsFrom(currentHashResults, stream);

            state = GpuReadExtender::State::BeforeRemoveIds;

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("removeUsedIdsAndMateIds", 0);
            removeUsedIdsAndMateIds(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("computePairFlagsGpu", 1);
            computePairFlagsGpu(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("loadCandidateSequenceData", 2);
            loadCandidateSequenceData(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("eraseDataOfRemovedMates", 3);
            eraseDataOfRemovedMates(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("calculateAlignments", 4);
            calculateAlignments(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("filterAlignments", 5);
            filterAlignments(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("computeMSAs", 6);
            computeMSAs(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("computeExtendedSequencesFromMSAs", 7);
            computeExtendedSequencesFromMSAs(stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            nvtx::push_range("prepareNextIteration", 8);
            prepareNextIteration(inputTasks, outputFinishedTasks, stream);
            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            destroyDeviceBuffers(stream);

            DEBUGSTREAMSYNC(stream);

        }else{
            std::cerr << "inputTasks empty\n";
            addFinishedGpuSoaTasks(inputTasks, outputFinishedTasks, stream);
            inputTasks.clearBuffers(stream);
        }

        //std::cerr << "exit thread " << std::this_thread::get_id() << "\n";
    }


    void removeUsedIdsAndMateIds(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeRemoveIds);
        assert(tasks->size() > 0);

        rmm::device_uvector<bool> d_shouldBeKept(initialNumCandidates, stream, mr);
        rmm::device_uvector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, mr);        

        resizeUninitialized(d_mateIdHasBeenRemoved, tasks->size(), stream);

        thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_shouldBeKept.begin(), initialNumCandidates, false);
        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<tasks->size(), 128, 0, stream>>>(
            d_candidateReadIds.data(),
            tasks->myReadId.data(),
            tasks->mateReadId.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchor.data(),
            d_shouldBeKept.data(),
            d_mateIdHasBeenRemoved.data(),
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            pairedEnd
        );
        CUDACHECKASYNC;  

        //copy selected candidate ids

        rmm::device_uvector<read_number> d_candidateReadIds2(initialNumCandidates, stream, mr);
        assert(h_numCandidates.data() != nullptr);

        CubCallWrapper(mr).cubSelectFlagged(
            d_candidateReadIds.data(),
            d_shouldBeKept.data(),
            d_candidateReadIds2.data(),
            h_numCandidates.data(),
            initialNumCandidates,
            stream
        );

        CUDACHECK(cudaEventRecordWrapper(h_numCandidatesEvent, stream));

        destroy(d_shouldBeKept, stream);

        //::erase(d_candidateReadIds2, d_candidateReadIds2.begin() + *h_numCandidates, d_candidateReadIds2.end(), stream);

        rmm::device_uvector<int> d_numCandidatesPerAnchorPrefixSum2(tasks->size() + 1, stream, mr);

        //compute prefix sum of number of candidates per anchor
        CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum2.data(), 0, sizeof(int), stream));

        CubCallWrapper(mr).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            tasks->size(),
            stream
        );

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent)); //wait for h_numCandidates
   
        #ifdef DO_ONLY_REMOVE_MATE_IDS
            std::swap(d_candidateReadIds, d_candidateReadIds2);
            std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2);
        #else         

            //compute segmented set difference.  d_candidateReadIds = d_candidateReadIds2 \ d_fullyUsedReadIds
            // auto d_candidateReadIds_end = GpuSegmentedSetOperation::set_difference(
            //     d_candidateReadIds2.data(),
            //     d_numCandidatesPerAnchor2.data(),
            //     d_numCandidatesPerAnchorPrefixSum2.data(),
            //     *h_numCandidates,
            //     tasks->size(),
            //     tasks->d_fullyUsedReadIds.data(),
            //     tasks->d_numFullyUsedReadIdsPerTask.data(),
            //     tasks->d_numFullyUsedReadIdsPerTaskPrefixSum.data(),
            //     tasks->d_fullyUsedReadIds.size(),
            //     tasks->size(),        
            //     d_candidateReadIds.data(),
            //     d_numCandidatesPerAnchor.data(),
            //     tasks->size(),
            //     stream,
            //     mr
            // );

            // *h_numCandidates = std::distance(d_candidateReadIds.data(), d_candidateReadIds_end);

        #endif

        h_candidateReadIds.resize(*h_numCandidates);
        CUDACHECK(cudaEventRecordWrapper(events[0], stream));
        CUDACHECK(cudaStreamWaitEventWrapper(hostOutputStream, events[0], 0));

        CUDACHECK(cudaMemcpyAsync(
            h_candidateReadIds.data(),
            d_candidateReadIds.data(),
            sizeof(read_number) * (*h_numCandidates),
            D2H,
            hostOutputStream
        ));

        CUDACHECK(cudaEventRecordWrapper(h_candidateReadIdsEvent, hostOutputStream));        

        destroy(d_numCandidatesPerAnchor2, stream);
        destroy(d_numCandidatesPerAnchorPrefixSum2, stream);        
        
        CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream));
        //compute prefix sum of new segment sizes
        CubCallWrapper(mr).cubInclusiveSum(
            d_numCandidatesPerAnchor.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(),
            stream
        );

        setState(GpuReadExtender::State::BeforeComputePairFlags);
    }

    void computePairFlagsGpu(cudaStream_t stream) {
        assert(state == GpuReadExtender::State::BeforeComputePairFlags);
        assert(tasks->size() > 0);

        resizeUninitialized(d_isPairedCandidate, initialNumCandidates, stream);
        rmm::device_uvector<int> d_firstTasksOfPairsToCheck(tasks->size(), stream, mr);
        rmm::device_uvector<bool> d_flags(tasks->size(), stream, mr);
        rmm::device_scalar<int> d_numChecks(stream, mr);

        thrust::fill_n(rmm::exec_policy_nosync(stream,mr), d_isPairedCandidate.begin(), initialNumCandidates, false);
        thrust::fill_n(rmm::exec_policy_nosync(stream,mr), d_flags.begin(), tasks->size(), false);

        readextendergpukernels::flagFirstTasksOfConsecutivePairedTasks<128><<<SDIV(tasks->size(), 128), 128, 0, stream>>>(
            tasks->size(),
            d_flags.data(),
            tasks->id.data()
        ); CUDACHECKASYNC;

        CubCallWrapper(mr).cubSelectFlagged(
            thrust::make_counting_iterator(0),
            d_flags.data(),
            d_firstTasksOfPairsToCheck.data(),
            d_numChecks.data(),
            tasks->size(),
            stream
        );

        destroy(d_flags, stream);

        readextendergpukernels::flagPairedCandidatesKernel<128,4096><<<tasks->size(), 128, 0, stream>>>(
            d_numChecks.data(),
            d_firstTasksOfPairsToCheck.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateReadIds.data(),
            tasks->d_numUsedReadIdsPerTask.data(),
            tasks->d_numUsedReadIdsPerTaskPrefixSum.data(),
            tasks->d_usedReadIds.data(),
            d_isPairedCandidate.data()
        ); CUDACHECKASYNC;

        setState(GpuReadExtender::State::BeforeLoadCandidates);

    }

    void loadCandidateSequenceData(cudaStream_t stream) {
        assert(state == GpuReadExtender::State::BeforeLoadCandidates);

        ::resizeUninitialized(d_candidateSequencesLength, initialNumCandidates, stream);
        ::resizeUninitialized(d_candidateSequencesData, encodedSequencePitchInInts * initialNumCandidates, stream);

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            makeAsyncConstBufferWrapper(h_candidateReadIds.data(), h_candidateReadIdsEvent),
            d_candidateReadIds.data(), //device accessible
            *h_numCandidates,
            stream,
            mr
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.data(),
            d_candidateReadIds.data(),
            *h_numCandidates,
            stream
        );

        setState(GpuReadExtender::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeEraseData);
        assert(tasks->size() > 0);

        rmm::device_uvector<bool> d_keepflags(initialNumCandidates, stream, mr);

        //compute flags of candidates which should not be removed. Candidates which should be removed are identical to mate sequence
        thrust::fill_n(rmm::exec_policy_nosync(stream,mr), d_keepflags.begin(), initialNumCandidates, true);

        const int* d_currentNumCandidates = d_numCandidatesPerAnchorPrefixSum.data() + tasks->size();

        constexpr int groupsize = 32;
        constexpr int blocksize = 128;
        constexpr int groupsperblock = blocksize / groupsize;
        dim3 block(blocksize,1,1);
        dim3 grid(SDIV(tasks->size() * groupsize, blocksize), 1, 1);
        const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

        readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
            tasks->inputEncodedMate.data(),
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_mateIdHasBeenRemoved.data(),
            tasks->size(),
            d_keepflags.data(),
            initialNumCandidates,
            d_currentNumCandidates
        ); CUDACHECKASYNC;

        compactCandidateDataByFlagsExcludingAlignments(
            d_keepflags.data(),
            false,
            stream
        );

        setState(GpuReadExtender::State::BeforeAlignment);
    }

    void calculateAlignments(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeAlignment);

        rmm::device_uvector<int> d_segmentIdsOfCandidates = getSegmentIdsPerElement(
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            tasks->size(), 
            initialNumCandidates,
            stream,
            mr
        );

        ::resizeUninitialized(d_alignment_overlaps, initialNumCandidates, stream);
        ::resizeUninitialized(d_alignment_shifts, initialNumCandidates, stream);
        ::resizeUninitialized(d_alignment_nOps, initialNumCandidates, stream);
        ::resizeUninitialized(d_alignment_best_alignment_flags, initialNumCandidates, stream);

        h_numAnchors[0] = tasks->size();

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int currentNumAnchors = tasks->size();
        const int currentNumCandidates = *h_numCandidates; //this does not need to be exact, but it must be >= d_numCandidatesPerAnchorPrefixSum[tasks->size()]
        const int maximumSequenceLength = encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = programOptions->min_overlap;
        const float maxErrorRate = programOptions->maxErrorRate;
        const float min_overlap_ratio = programOptions->min_overlap_ratio;
        const float maxRatioOpsOverOverlapForOrientationFilter = 0.06f;

        callRightShiftedHammingDistanceKernel(
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            d_anchorSequencesData.data(),
            d_candidateSequencesData.data(),
            d_anchorSequencesLength.data(),
            d_candidateSequencesLength.data(),
            d_segmentIdsOfCandidates.data(),
            currentNumAnchors,
            currentNumCandidates,
            d_anchorContainsN,
            removeAmbiguousAnchors,
            d_candidateContainsN,
            removeAmbiguousCandidates,
            maximumSequenceLength,
            maximumSequenceLength,
            encodedSequencePitchInInts2Bit,
            encodedSequencePitchInInts2Bit,
            min_overlap,
            maxErrorRate,
            min_overlap_ratio,
            maxRatioOpsOverOverlapForOrientationFilter,
            stream
        );

        setState(GpuReadExtender::State::BeforeAlignmentFilter);
    }

    void filterAlignments(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeAlignmentFilter);
        assert(tasks->size() > 0);

        rmm::device_uvector<bool> d_keepflags(initialNumCandidates, stream, mr);

        const int* const d_currentNumCandidates = d_numCandidatesPerAnchorPrefixSum.data() + tasks->size();

        readextendergpukernels::flagGoodAlignmentsKernel<128><<<tasks->size(), 128, 0, stream>>>(
        //readextendergpukernels::flagGoodAlignmentsKernel<1><<<1, 1, 0, stream>>>(
            tasks->id.data(),
            tasks->iteration.data(),
            d_alignment_best_alignment_flags.data(),
            d_alignment_shifts.data(),
            d_alignment_overlaps.data(),
            d_anchorSequencesLength.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_isPairedCandidate.data(),
            d_keepflags.data(),
            programOptions->min_overlap_ratio,
            tasks->size(),
            d_currentNumCandidates,
            initialNumCandidates
        ); CUDACHECKASYNC;

        compactCandidateDataByFlags(
            d_keepflags.data(),
            true, //copy candidate read ids to host because they might be needed to load quality scores
            stream
        );

        setState(GpuReadExtender::State::BeforeMSA);
    }

    void computeMSAs(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeMSA);
        assert(tasks->size() > 0);

        rmm::device_uvector<char> d_candidateQualityScores(qualityPitchInBytes * initialNumCandidates, stream, mr);

        loadCandidateQualityScores(stream, d_candidateQualityScores.data());

        rmm::device_uvector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, mr);

        rmm::device_uvector<int> indices1(initialNumCandidates, stream, mr);
        rmm::device_uvector<int> indices2(initialNumCandidates, stream, mr);
        rmm::device_scalar<int> d_numCandidates2(stream, mr);

        const int threads = 32 * tasks->size();
        readextendergpukernels::segmentedIotaKernel<32><<<SDIV(threads, 128), 128, 0, stream>>>(
            indices1.data(),
            tasks->size(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data()
        ); CUDACHECKASYNC;

        *h_numAnchors = tasks->size();

        const bool useQualityScoresForMSA = true;

        multiMSA.construct(
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_anchorSequencesData.data(),
            d_anchorQualityScores.data(),
            tasks->size(),
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            programOptions->maxErrorRate,
            gpu::MSAColumnCount(msaColumnPitchInElements),
            stream
        );

        multiMSA.refine(
            indices2.data(),
            d_numCandidatesPerAnchor2.data(),
            d_numCandidates2.data(),
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_anchorSequencesData.data(),
            d_anchorQualityScores.data(),
            tasks->size(),
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            initialNumCandidates,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            programOptions->maxErrorRate,
            programOptions->estimatedCoverage,
            getNumRefinementIterations(),
            stream
        );
 
        rmm::device_uvector<bool> d_shouldBeKept(initialNumCandidates, stream, mr);

        thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_shouldBeKept.begin(), initialNumCandidates, false);

        const int numThreads2 = tasks->size() * 32;
        readextendergpukernels::convertLocalIndicesInSegmentsToGlobalFlags<128,32>
        <<<SDIV(numThreads2, 128), 128, 0, stream>>>(
            d_shouldBeKept.data(),
            indices2.data(),
            d_numCandidatesPerAnchor2.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            tasks->size()
        ); CUDACHECKASYNC
     
        destroy(indices1, stream);
        destroy(indices2, stream);
        destroy(d_numCandidatesPerAnchor2, stream);

        destroy(d_candidateQualityScores, stream);

        compactCandidateDataByFlags(
            d_shouldBeKept.data(),
            false,
            stream
        );
        setState(GpuReadExtender::State::BeforeExtend);
    }

    void computeExtendedSequencesFromMSAs(cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforeExtend);
        assert(tasks->size() > 0);

        outputAnchorPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;
        outputAnchorQualityPitchInBytes = SDIV(qualityPitchInBytes, 128) * 128;
        decodedMatesRevCPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;

        rmm::device_uvector<int> d_accumExtensionsLengthsOUT(tasks->size(), stream, mr);
        rmm::device_uvector<int> d_sizeOfGapToMate(tasks->size(), stream, mr);
        rmm::device_uvector<float> d_goodscores(tasks->size(), stream, mr);

        rmm::device_uvector<char> d_outputAnchors(tasks->size() * outputAnchorPitchInBytes, stream, mr);
        rmm::device_uvector<char> d_outputAnchorQualities(tasks->size() * outputAnchorQualityPitchInBytes, stream, mr);
        rmm::device_uvector<bool> d_outputMateHasBeenFound(tasks->size(), stream, mr);
        rmm::device_uvector<extension::AbortReason> d_abortReasons(tasks->size(), stream, mr);
        rmm::device_uvector<int> d_outputAnchorLengths(tasks->size(), stream, mr);
        //rmm::device_uvector<bool> d_isFullyUsedCandidate(initialNumCandidates, stream, mr);

        thrust::fill_n(
            rmm::exec_policy_nosync(stream, mr),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_outputMateHasBeenFound.data(),
                d_abortReasons.data(),
                d_goodscores.data()
            )),
            tasks->size(),
            thrust::make_tuple(
                false, 
                extension::AbortReason::None,
                0.0f
            )
        );

        //thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_isFullyUsedCandidate.begin(), initialNumCandidates, false);
      
        //compute extensions

        readextendergpukernels::computeExtensionStepFromMsaKernel<128><<<tasks->size(), 128, 0, stream>>>(
            programOptions->minFragmentSize,
            programOptions->maxFragmentSize,
            multiMSA.multiMSAView(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_accumExtensionsLengths.data(),
            tasks->soainputmateLengths.data(),
            d_abortReasons.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_outputAnchors.data(),
            outputAnchorPitchInBytes,
            d_outputAnchorQualities.data(),
            outputAnchorQualityPitchInBytes,
            d_outputAnchorLengths.data(),
            tasks->pairedEnd.data(),
            tasks->inputEncodedMate.data(),
            encodedSequencePitchInInts,
            decodedMatesRevCPitchInBytes,
            d_outputMateHasBeenFound.data(),
            d_sizeOfGapToMate.data(),
            iterationConfig->minCoverageForExtension,
            iterationConfig->maxextensionPerStep
        ); CUDACHECKASYNC;

        readextendergpukernels::computeExtensionStepQualityKernel<128><<<tasks->size(), 128, 0, stream>>>(
            d_goodscores.data(),
            multiMSA.multiMSAView(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_accumExtensionsLengths.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_anchorSequencesLength.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateSequencesLength.data(),
            d_alignment_shifts.data(),
            d_alignment_best_alignment_flags.data(),
            d_candidateSequencesData.data(),
            multiMSA.getColumnProperties(),
            encodedSequencePitchInInts
        ); CUDACHECKASYNC;

        // readextendergpukernels::flagFullyUsedCandidatesKernel<128>
        // <<<tasks->size(), 128, 0, stream>>>(
        //     tasks->id.data(),
        //     tasks->iteration.data(),
        //     tasks->size(),
        //     d_numCandidatesPerAnchor.data(),
        //     d_numCandidatesPerAnchorPrefixSum.data(),
        //     d_candidateSequencesLength.data(),
        //     d_alignment_shifts.data(),
        //     d_anchorSequencesLength.data(),
        //     d_accumExtensionsLengths.data(),
        //     d_accumExtensionsLengthsOUT.data(),
        //     d_abortReasons.data(),
        //     d_outputMateHasBeenFound.data(),
        //     d_isFullyUsedCandidate.data()
        // ); CUDACHECKASYNC;

        nvtx::push_range("gpuunpack", 3);

        tasks->addScalarIterationResultData(
            d_goodscores.data(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            stream
        );

        rmm::device_uvector<int> d_addNumEntriesPerTask(tasks->size(), stream, mr);
        rmm::device_uvector<int> d_addNumEntriesPerTaskPrefixSum(tasks->size()+1, stream, mr);

        //init first prefixsum element with 0.
        //compute number of soa iteration results per task
        readextendergpukernels::computeNumberOfSoaIterationResultsPerTaskKernel<128>
        <<<SDIV(tasks->size(), 128), 128, 0, stream>>>(
            tasks->size(),
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_sizeOfGapToMate.data()
        ); CUDACHECKASYNC;
 
        CubCallWrapper(mr).cubInclusiveSum(
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data() + 1,
            tasks->size(),
            stream
        );

        // int* const tmpptr1 = reinterpret_cast<int*>(h_tempForMemcopies.data());
        // int numAdd = 0;
        // CUDACHECK(cudaMemcpyAsync(
        //     tmpptr1,
        //     d_addNumEntriesPerTaskPrefixSum.data() + tasks->size(),
        //     sizeof(int),
        //     D2H,
        //     stream
        // ));
        // CUDACHECK(cudaStreamSynchronizeWrapper(stream));
        // numAdd = *tmpptr1;

        //tasks->size() is upper bound, use it to avoid synchronization
        rmm::device_uvector<char> d_addTotalDecodedAnchorsFlat(tasks->size() * outputAnchorPitchInBytes, stream, mr);
        rmm::device_uvector<char> d_addTotalAnchorQualityScoresFlat(tasks->size() * outputAnchorQualityPitchInBytes, stream, mr);
        rmm::device_uvector<int> d_addAnchorLengths(tasks->size(), stream, mr);
        rmm::device_uvector<int> d_addAnchorBeginsInExtendedRead(tasks->size(), stream, mr);
        //std::cerr << "numAdd = " << numAdd << "\n";

        assert(tasks->decodedSequencePitchInBytes >= outputAnchorPitchInBytes);
        assert(tasks->qualityPitchInBytes >= outputAnchorQualityPitchInBytes);

        readextendergpukernels::makeSoAIterationResultsKernel<128><<<tasks->size(), 128, 0, stream>>>(
            tasks->id.data(),
            tasks->iteration.data(),
            tasks->size(),
            outputAnchorPitchInBytes,
            outputAnchorQualityPitchInBytes,
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data(),
            d_addTotalDecodedAnchorsFlat.data(),
            d_addTotalAnchorQualityScoresFlat.data(),
            d_addAnchorLengths.data(),
            d_addAnchorBeginsInExtendedRead.data(),
            tasks->decodedSequencePitchInBytes,
            tasks->qualityPitchInBytes,
            tasks->abortReason.data(),
            tasks->mateHasBeenFound.data(),
            tasks->soainputdecodedMateRevC.data(),
            tasks->soainputmateQualityScoresReversed.data(),
            tasks->soainputmateLengths.data(),
            d_sizeOfGapToMate.data(),
            d_outputAnchorLengths.data(),
            d_outputAnchors.data(),
            d_outputAnchorQualities.data(),
            d_accumExtensionsLengthsOUT.data()
        ); CUDACHECKASYNC;

        tasks->addSoAIterationResultData(
            d_addNumEntriesPerTask.data(),
            d_addNumEntriesPerTaskPrefixSum.data(),
            d_addAnchorLengths.data(),
            d_addAnchorBeginsInExtendedRead.data(),
            d_addTotalDecodedAnchorsFlat.data(),
            d_addTotalAnchorQualityScoresFlat.data(),
            outputAnchorPitchInBytes,
            outputAnchorQualityPitchInBytes,
            stream,
            h_tempForMemcopies.data()
        );

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));

        // tasks->updateUsedReadIdsAndFullyUsedReadIds(
        //     d_candidateReadIds.data(),
        //     d_numCandidatesPerAnchor.data(),
        //     d_numCandidatesPerAnchorPrefixSum.data(),
        //     d_isFullyUsedCandidate.data(),
        //     *h_numCandidates,
        //     stream,
        //     h_tempForMemcopies.data()
        // );
        tasks->updateUsedReadIds(
            d_candidateReadIds.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            *h_numCandidates,
            stream,
            h_tempForMemcopies.data()
        );

        //increment iteration and check early exit of tasks
        tasks->iterationIsFinished(stream);

        nvtx::pop_range();

        std::swap(d_accumExtensionsLengths, d_accumExtensionsLengthsOUT);

        setState(GpuReadExtender::State::BeforePrepareNextIteration);
    }
    
    void prepareNextIteration(TaskData& outputActiveTasks, TaskData& outputFinishedTasks, cudaStream_t stream){
        assert(state == GpuReadExtender::State::BeforePrepareNextIteration);

        const int totalTasksBefore = tasks->size() + outputFinishedTasks.size();

        rmm::device_uvector<bool> d_activeFlags(tasks->size(), stream, mr);
        tasks->getActiveFlags(d_activeFlags.data(), programOptions->minFragmentSize, programOptions->maxFragmentSize, stream);

        TaskData newgpuSoaActiveTasks = tasks->select(
            d_activeFlags.data(),
            stream,
            h_tempForMemcopies.data()
        );

        auto inactiveFlags = thrust::make_transform_iterator(
            d_activeFlags.data(),
            thrust::logical_not<bool>{}
        );

        TaskData newlygpuSoaFinishedTasks = tasks->select(
            inactiveFlags,
            stream,
            h_tempForMemcopies.data()
        );

        addFinishedGpuSoaTasks(newlygpuSoaFinishedTasks, outputFinishedTasks, stream);
        std::swap(outputActiveTasks, newgpuSoaActiveTasks);

        const int totalTasksAfter = tasks->size() + outputFinishedTasks.size();
        assert(totalTasksAfter == totalTasksBefore);

        if(!isEmpty()){
            setState(GpuReadExtender::State::UpdateWorkingSet);
        }else{
            setStateToFinished(stream);
        }
        
    }


    TaskData getFinishedGpuSoaTasksOfFinishedPairsAndRemoveThem(TaskData& finishedTasks, cudaStream_t stream) const{
        //determine tasks in groups of 4

        if(finishedTasks.size() > 0){
            rmm::device_uvector<int> d_positions4(finishedTasks.size(), stream, mr);
            rmm::device_uvector<int> d_positionsNot4(finishedTasks.size(), stream, mr);
            rmm::device_uvector<int> d_numPositions(2, stream, mr);

            thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_numPositions.begin(), 2, 0);

            if(computeTaskSplitGatherIndicesSmallInput.computationPossible(finishedTasks.size())){
                computeSplitGatherIndicesOfFinishedTasksSmall(
                    finishedTasks,
                    d_positions4.data(), 
                    d_positionsNot4.data(), 
                    d_numPositions.data(), 
                    d_numPositions.data() + 1,
                    stream
                );
            }else{
                computeSplitGatherIndicesOfFinishedTasksDefault(
                    finishedTasks,
                    d_positions4.data(), 
                    d_positionsNot4.data(), 
                    d_numPositions.data(), 
                    d_numPositions.data() + 1,
                    stream
                );
            }

            CUDACHECK(cudaMemcpyAsync(
                h_numPositions.data(),
                d_numPositions.data(),
                sizeof(int) * 2,
                D2H,
                stream
            ));

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));

            if(h_numPositions[0] > 0){

                TaskData gpufinishedTasks4 = finishedTasks.gather(
                    d_positions4.data(), 
                    d_positions4.data() + h_numPositions[0],
                    stream,
                    h_tempForMemcopies.data()
                );

                TaskData gpufinishedTasksNot4 = finishedTasks.gather(
                    d_positionsNot4.data(), 
                    d_positionsNot4.data() + h_numPositions[1],
                    stream,
                    h_tempForMemcopies.data()
                );

                std::swap(finishedTasks, gpufinishedTasksNot4);

                return gpufinishedTasks4;
            }else{
                return TaskData(mr); //empty. no finished tasks to process
            }
        }else{
            return TaskData(mr); //empty. no finished tasks to process
        }
    }

    void computeSplitGatherIndicesOfFinishedTasksSmall(
        const TaskData& finishedTasks,
        int* d_positions4, 
        int* d_positionsNot4, 
        int* d_numPositions4, 
        int* d_numPositionsNot4,
        cudaStream_t stream
    ) const {
        assert(computeTaskSplitGatherIndicesSmallInput.computationPossible(finishedTasks.size()));

        if(finishedTasks.size() == 0){
            CUDACHECK(cudaMemsetAsync(d_numPositions4, 0, sizeof(int), stream));
            thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_numPositionsNot4, 1, int(finishedTasks.size()));
            return;
        }

        rmm::device_uvector<int> d_minmax(2, stream, mr);

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
        //readextendergpukernels::minmaxSingleBlockKernel<128><<<1, 128, 0, stream>>>(
            finishedTasks.pairId.data(),
            finishedTasks.size(),
            d_minmax.data()
        ); CUDACHECKASYNC;       

        computeTaskSplitGatherIndicesSmallInput.compute(
            finishedTasks.size(),
            d_positions4,
            d_positionsNot4,
            d_numPositions4,
            d_numPositionsNot4,
            finishedTasks.pairId.data(),
            finishedTasks.id.data(),
            d_minmax.data(),
            stream
        );
    }



    void computeSplitGatherIndicesOfFinishedTasksDefault(
        const TaskData& finishedTasks,
        int* d_positions4, 
        int* d_positionsNot4, 
        int* d_numPositions4, 
        int* d_numPositionsNot4,
        cudaStream_t stream
    ) const {
        if(finishedTasks.size() == 0){
            CUDACHECK(cudaMemsetAsync(d_numPositions4, 0, sizeof(int), stream));
            thrust::fill_n(rmm::exec_policy_nosync(stream, mr), d_numPositionsNot4, 1, int(finishedTasks.size()));
            return;
        }

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
            finishedTasks.pairId.data(),
            finishedTasks.size(),
            h_minmax.data()
        ); CUDACHECKASYNC;

        CUDACHECK(cudaStreamSynchronizeWrapper(stream));

        rmm::device_uvector<int> d_pairIds1(finishedTasks.size(), stream, mr);
        rmm::device_uvector<int> d_pairIds2(finishedTasks.size(), stream, mr);
        rmm::device_uvector<int> d_indices1(finishedTasks.size(), stream, mr);
        rmm::device_uvector<int> d_incices2(finishedTasks.size(), stream, mr);

        //decrease pair ids by smallest pair id to improve radix sort performance
        thrust::transform(
            rmm::exec_policy_nosync(stream, mr),
            finishedTasks.pairId.begin(),
            finishedTasks.pairId.end(),
            thrust::make_constant_iterator(-h_minmax[0]),
            d_pairIds1.begin(),
            thrust::plus<int>{}
        );

        thrust::sequence(rmm::exec_policy_nosync(stream, mr), d_indices1.begin(), d_indices1.end(), 0);
       
        cub::DoubleBuffer<int> d_keys(d_pairIds1.data(), d_pairIds2.data());
        cub::DoubleBuffer<int> d_values(d_indices1.data(), d_incices2.data());

        const int begin_bit = 0;
        const int end_bit = std::ceil(std::log2(h_minmax[1] - h_minmax[0]));

        cudaError_t status = cudaSuccess;
        std::size_t tempbytes = 0;
        status = cub::DeviceRadixSort::SortPairs(
            nullptr,
            tempbytes,
            d_keys,
            d_values,
            finishedTasks.size(), 
            begin_bit, 
            end_bit, 
            stream
        );
        CUDACHECK(status);

        rmm::device_uvector<char> d_temp(tempbytes, stream, mr);

        status = cub::DeviceRadixSort::SortPairs(
            d_temp.data(),
            tempbytes,
            d_keys,
            d_values,
            finishedTasks.size(), 
            begin_bit, 
            end_bit, 
            stream
        );
        CUDACHECK(status);
        destroy(d_temp, stream);       

        const int* d_theSortedPairIds = d_keys.Current();
        const int* d_theSortedIndices = d_values.Current();

        rmm::device_uvector<int> d_counts_out(finishedTasks.size(), stream, mr);
        rmm::device_scalar<int> d_num_runs_out(stream);

        CubCallWrapper(mr).cubReduceByKey(
            d_theSortedPairIds, 
            cub::DiscardOutputIterator<>{},
            thrust::make_constant_iterator(1),
            d_counts_out.data(),
            d_num_runs_out.data(),
            thrust::plus<int>{},
            finishedTasks.size(),
            stream
        );

        destroy(d_pairIds1, stream);
        destroy(d_pairIds2, stream);

        //compute prefix sums to have stable output
        rmm::device_uvector<int> d_outputoffsetsPos4(finishedTasks.size() + 1, stream, mr);
        rmm::device_uvector<int> d_outputoffsetsNotPos4(finishedTasks.size() + 1, stream, mr);
        rmm::device_uvector<int> d_countsInclusivePrefixSum(finishedTasks.size(), stream, mr);

        //compute two exclusive prefix sums and one inclusive prefix sum. the three operations are fused into a single call

        helpers::lambda_kernel<<<1,1,0,stream>>>([
            d_outputoffsetsPos4 = d_outputoffsetsPos4.data(),
            d_outputoffsetsNotPos4 = d_outputoffsetsNotPos4.data()
        ] __device__ (){
            d_outputoffsetsPos4[0] = 0;
            d_outputoffsetsNotPos4[0] = 0;
        }); CUDACHECKASYNC;

        auto inputIterator1 = thrust::make_transform_iterator(
            d_counts_out.data(),
            [] __host__ __device__ (int count){
                if(count == 4){
                    return count;
                }else{
                    return 0;
                }
            }
        );

        auto outputIterator1 = d_outputoffsetsPos4.data() + 1; // exclusive sum, so inclusive sum starts at position 1

        auto inputIterator2 = thrust::make_transform_iterator(
            d_counts_out.data(),
            [] __host__ __device__ (int count){
                if(count != 4){
                    return count;
                }else{
                    return 0;
                }
            }
        );

        auto outputIterator2 = d_outputoffsetsNotPos4.data() + 1; // exclusive sum, so inclusive sum starts at position 1

        auto inputIterator3 = d_counts_out.data();
        auto outputIterator3 = d_countsInclusivePrefixSum.data();

        CubCallWrapper(mr).cubInclusiveScan(
            thrust::make_zip_iterator(thrust::make_tuple(
                inputIterator1, inputIterator2, inputIterator3
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                outputIterator1, outputIterator2, outputIterator3
            )),
            ThrustTupleAddition<3>{},
            finishedTasks.size(),
            stream
        );

        readextendergpukernels::computeTaskSplitGatherIndicesDefaultKernel<256><<<SDIV(finishedTasks.size(), 256), 256, 0, stream>>>(
            finishedTasks.size(),
            d_positions4,
            d_positionsNot4,
            d_numPositions4,
            d_numPositionsNot4,
            d_countsInclusivePrefixSum.data(),
            d_num_runs_out.data(),
            d_theSortedIndices,
            finishedTasks.id.data(),
            d_outputoffsetsPos4.data(),
            d_outputoffsetsNotPos4.data()
        ); CUDACHECKASYNC;
    }

    void constructRawResults(TaskData& finishedTasks, RawExtendResult& rawResults, cudaStream_t stream){

        nvtx::ScopedRange sr("constructRawResults", 5);
        std::lock_guard<std::mutex> lockguard(mutex);
        //std::cerr << "constructRawResults enter thread " << std::this_thread::get_id() << "\n";

        auto finishedTasks4 = getFinishedGpuSoaTasksOfFinishedPairsAndRemoveThem(finishedTasks, stream);
        CUDACHECK(cudaStreamSynchronizeWrapper(stream));

        const int numFinishedTasks = finishedTasks4.size();
        rawResults.noReadIsExtended = false;
        rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
        rawResults.numResults = numFinishedTasks / 4;

        if(numFinishedTasks == 0){            
            return;
        }

        CUDACHECK(cudaMemcpyAsync(
            rawResults.h_tmp.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data() + numFinishedTasks,
            sizeof(int),
            D2H,
            stream
        ));
        CUDACHECK(cudaEventRecordWrapper(rawResults.event, stream));

        //copy data from device to host in second stream
        
        rawResults.h_gpuabortReasons.resize(numFinishedTasks);
        rawResults.h_gpudirections.resize(numFinishedTasks);
        rawResults.h_gpuiterations.resize(numFinishedTasks);
        rawResults.h_gpuReadIds.resize(numFinishedTasks);
        rawResults.h_gpuMateReadIds.resize(numFinishedTasks);
        rawResults.h_gpuAnchorLengths.resize(numFinishedTasks);
        rawResults.h_gpuMateLengths.resize(numFinishedTasks);
        rawResults.h_gpugoodscores.resize(numFinishedTasks);
        rawResults.h_gpuMateHasBeenFound.resize(numFinishedTasks);

        using care::gpu::MemcpyParams;

        auto memcpyParams1 = cuda::std::tuple_cat(
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuabortReasons.data(), finishedTasks4.abortReason.data(), sizeof(extension::AbortReason) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpudirections.data(), finishedTasks4.direction.data(), sizeof(extension::ExtensionDirection) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuiterations.data(), finishedTasks4.iteration.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuReadIds.data(), finishedTasks4.myReadId.data(), sizeof(read_number) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateReadIds.data(), finishedTasks4.mateReadId.data(), sizeof(read_number) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuAnchorLengths.data(), finishedTasks4.soainputAnchorLengths.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateLengths.data(), finishedTasks4.soainputmateLengths.data(), sizeof(int) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpugoodscores.data(), finishedTasks4.goodscore.data(), sizeof(float) * numFinishedTasks)),
            cuda::std::make_tuple(MemcpyParams(rawResults.h_gpuMateHasBeenFound.data(), finishedTasks4.mateHasBeenFound.data(), sizeof(bool) * numFinishedTasks))
        );

        care::gpu::memcpyKernel<int><<<SDIV(numFinishedTasks, 256), 256, 0, stream>>>(memcpyParams1); CUDACHECKASYNC;
       
        CUDACHECK(cudaEventSynchronizeWrapper(rawResults.event));

        const int totalNumIterationResultsOfFinishedTasks = *rawResults.h_tmp;

        assert(totalNumIterationResultsOfFinishedTasks >= 0);

        //if there are no candidates, the resulting sequences will be identical to the input anchors. no computing required
        if(totalNumIterationResultsOfFinishedTasks == 0){
            //finishedTasks4.consistencyCheck(true);

            rawResults.h_inputAnchorsDecoded.resize(finishedTasks4.size() * decodedSequencePitchInBytes);

            assert(finishedTasks4.soainputAnchorsDecoded.size() >= sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes);
            assert(rawResults.h_inputAnchorsDecoded.size() >= sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes);

            CUDACHECK(cudaMemcpyAsync(
                rawResults.h_inputAnchorsDecoded.data(),
                finishedTasks4.soainputAnchorsDecoded.data(),
                sizeof(char) * finishedTasks4.size() * decodedSequencePitchInBytes,
                D2H,
                stream
            ));

            rawResults.noReadIsExtended = true;
            rawResults.decodedSequencePitchInBytes = decodedSequencePitchInBytes;

            return;
        }

        rmm::device_uvector<unsigned int> d_extendedIterationSequences(totalNumIterationResultsOfFinishedTasks * encodedSequencePitchInInts, stream, mr);

        callEncodeSequencesTo2BitKernel(
            d_extendedIterationSequences.data(),
            finishedTasks4.soatotalDecodedAnchorsFlat.data(),
            finishedTasks4.soatotalDecodedAnchorsLengths.data(),
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            totalNumIterationResultsOfFinishedTasks,
            8,
            stream
        );


        //sequence data has been transfered to gpu. now set up remaining msa input data

        rmm::device_uvector<int> d_alignment_overlaps_tmp(totalNumIterationResultsOfFinishedTasks, stream, mr);
        rmm::device_uvector<int> d_alignment_nOps_tmp(totalNumIterationResultsOfFinishedTasks, stream, mr);
        rmm::device_uvector<AlignmentOrientation> d_alignment_best_alignment_flags_tmp(totalNumIterationResultsOfFinishedTasks, stream, mr);
        rmm::device_uvector<bool> d_isPairedCandidate_tmp(totalNumIterationResultsOfFinishedTasks, stream, mr);

        //fill the arrays such that msa will have good quality without pairedness
        thrust::fill_n(
            rmm::exec_policy_nosync(stream, mr),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_alignment_overlaps_tmp.begin(),
                d_alignment_nOps_tmp.begin(),
                d_alignment_best_alignment_flags_tmp.begin(),
                d_isPairedCandidate_tmp.begin()
            )), 
            totalNumIterationResultsOfFinishedTasks, 
            thrust::make_tuple(
                gpuReadStorage->getSequenceLengthUpperBound(),
                0,
                AlignmentOrientation::Forward,
                false
            )
        );
                
        //all input data ready. now set up msa

        rmm::device_uvector<int> indices1(totalNumIterationResultsOfFinishedTasks, stream, mr);

        const int threads = 32 * numFinishedTasks;

        readextendergpukernels::segmentedIotaKernel<32><<<SDIV(threads, 128), 128, 0, stream>>>(
            indices1.data(),
            numFinishedTasks,
            finishedTasks4.soaNumIterationResultsPerTask.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data()
        ); CUDACHECKASYNC;

        multiMSA.construct(
            d_alignment_overlaps_tmp.data(),
            finishedTasks4.soatotalAnchorBeginInExtendedRead.data(),
            d_alignment_nOps_tmp.data(),
            d_alignment_best_alignment_flags_tmp.data(),
            indices1.data(),
            finishedTasks4.soaNumIterationResultsPerTask.data(),
            finishedTasks4.soaNumIterationResultsPerTaskPrefixSum.data(),
            finishedTasks4.soainputAnchorLengths.data(),
            finishedTasks4.inputAnchorsEncoded.data(),
            nullptr, //anchor qualities
            numFinishedTasks,
            finishedTasks4.soatotalDecodedAnchorsLengths.data(),
            d_extendedIterationSequences.data(),
            nullptr, //candidate qualities
            d_isPairedCandidate_tmp.data(),
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            false, //useQualityScores
            programOptions->maxErrorRate,
            gpu::MSAColumnCount::unknown(),
            stream
        );

        assert(multiMSA.getNumMSAs() == numFinishedTasks);

        destroy(indices1, stream);
        destroy(d_extendedIterationSequences, stream);
        destroy(d_alignment_overlaps_tmp, stream);
        destroy(d_alignment_nOps_tmp, stream);
        destroy(d_alignment_best_alignment_flags_tmp, stream);
        destroy(d_isPairedCandidate_tmp, stream);

        const std::size_t resultMSAColumnPitchInElements = multiMSA.getMaximumMsaWidth();

        //compute quality of consensus
        rmm::device_uvector<char> d_consensusQuality(numFinishedTasks * resultMSAColumnPitchInElements, stream, mr);
        rmm::device_uvector<char> d_decodedConsensus(numFinishedTasks * resultMSAColumnPitchInElements, stream, mr);
        rmm::device_uvector<int> d_resultLengths(numFinishedTasks, stream, mr);
        
        multiMSA.computeConsensusQuality(
            d_consensusQuality.data(),
            resultMSAColumnPitchInElements,
            stream
        );

        multiMSA.computeConsensus(
            d_decodedConsensus.data(),
            resultMSAColumnPitchInElements,
            stream
        );

        multiMSA.computeMsaSizes(
            d_resultLengths.data(),
            stream
        );

        multiMSA.destroy(stream);


        const int numResults = numFinishedTasks / 4;

        rmm::device_uvector<int> d_pairResultLengths(numResults, stream, mr);

        //compute pair result output sizes and use them to determine required memory
        readextendergpukernels::makePairResultsFromFinishedTasksDryRunKernel<128><<<numResults, 128, 0, stream>>>(
            numResults,
            d_pairResultLengths.data(),
            finishedTasks4.soainputAnchorLengths.data(), 
            d_resultLengths.data(),
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            finishedTasks4.mateHasBeenFound.data(),
            finishedTasks4.goodscore.data(),
            resultMSAColumnPitchInElements,
            programOptions->minFragmentSize,
            programOptions->maxFragmentSize
        ); CUDACHECKASYNC;

        int* const minmaxPairResultLengths = rawResults.h_tmp.data();

        readextendergpukernels::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
        //readextendergpukernels::minmaxSingleBlockKernel<128><<<1, 128, 0, stream>>>(
            d_pairResultLengths.data(),
            numResults,
            minmaxPairResultLengths
        ); CUDACHECKASYNC;

        CUDACHECK(cudaEventRecordWrapper(rawResults.event, stream));

        //replace positions which are covered by anchor and mate with the original data
        readextendergpukernels::applyOriginalReadsToExtendedReads<128,32>
        <<<SDIV(numFinishedTasks, 4), 128, 0, stream>>>(
            resultMSAColumnPitchInElements,
            numFinishedTasks,
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            d_resultLengths.data(),
            finishedTasks4.inputAnchorsEncoded.data(),
            finishedTasks4.soainputAnchorLengths.data(),
            finishedTasks4.soainputAnchorQualities.data(),
            finishedTasks4.mateHasBeenFound.data(),
            encodedSequencePitchInInts,
            qualityPitchInBytes
        ); CUDACHECKASYNC;

        CUDACHECK(cudaEventSynchronizeWrapper(rawResults.event));

        const int outputPitch = SDIV(minmaxPairResultLengths[1], 4) * 4; //round up maximum output size to 4 bytes

        // rmm::device_uvector<int> d_pairResultRead1Begins(numResults, stream, mr);
        // rmm::device_uvector<int> d_pairResultRead2Begins(numResults, stream, mr);
        // rmm::device_uvector<char> d_pairResultSequences(numResults * outputPitch, stream, mr);
        // rmm::device_uvector<char> d_pairResultQualities(numResults * outputPitch, stream, mr);
        // rmm::device_uvector<bool> d_pairResultMateHasBeenFound(numResults, stream, mr);
        // rmm::device_uvector<bool> d_pairResultMergedDifferentStrands(numResults, stream, mr);
        // rmm::device_uvector<bool> d_pairResultAnchorIsLR(numResults, stream, mr);

        std::size_t flatbuffers2size = sizeof(int) * numResults
            + sizeof(int) * numResults
            + sizeof(int) * numResults
            + sizeof(char) * numResults * outputPitch
            + sizeof(char) * numResults * outputPitch
            + sizeof(bool) * numResults
            + sizeof(bool) * numResults
            + sizeof(bool) * numResults;

        rmm::device_uvector<char> d_resultsperpseudoreadflat(flatbuffers2size, stream, mr);
        int* d_pairResultRead1Begins = reinterpret_cast<int*>(d_resultsperpseudoreadflat.data());
        int* d_pairResultRead2Begins = reinterpret_cast<int*>(d_pairResultRead1Begins + numResults);
        char* d_pairResultSequences = reinterpret_cast<char*>(d_pairResultRead2Begins + numResults);
        char* d_pairResultQualities = reinterpret_cast<char*>(d_pairResultSequences + numResults * outputPitch);
        bool* d_pairResultMateHasBeenFound = reinterpret_cast<bool*>(d_pairResultQualities + numResults * outputPitch);
        bool* d_pairResultMergedDifferentStrands = reinterpret_cast<bool*>(d_pairResultMateHasBeenFound + numResults);
        bool* d_pairResultAnchorIsLR = reinterpret_cast<bool*>(d_pairResultMergedDifferentStrands + numResults);


        
        const std::size_t smem = 3 * outputPitch;

        readextendergpukernels::makePairResultsFromFinishedTasksKernel<128><<<numResults, 128, smem, stream>>>(
            numResults,
            d_pairResultAnchorIsLR,
            d_pairResultSequences,
            d_pairResultQualities,
            d_pairResultLengths.data(),
            d_pairResultRead1Begins,
            d_pairResultRead2Begins,
            d_pairResultMateHasBeenFound,
            d_pairResultMergedDifferentStrands,
            outputPitch,
            finishedTasks4.soainputAnchorLengths.data(), 
            d_resultLengths.data(),
            d_decodedConsensus.data(),
            d_consensusQuality.data(),
            finishedTasks4.mateHasBeenFound.data(),
            finishedTasks4.goodscore.data(),
            resultMSAColumnPitchInElements,
            programOptions->minFragmentSize,
            programOptions->maxFragmentSize
        ); CUDACHECKASYNC;

        destroy(d_consensusQuality, stream);
        destroy(d_decodedConsensus, stream);
        destroy(d_resultLengths, stream);

        rawResults.h_pairResultLengths.resize(numResults);
        rawResults.resizePseudoReadBuffers(numResults, outputPitch);

        CUDACHECK(cudaMemcpyAsync(
            rawResults.h_pairResultLengths.data(),
            d_pairResultLengths.data(),
            sizeof(int) * numResults,
            D2H,
            stream
        ));
        CUDACHECK(cudaMemcpyAsync(
            rawResults.h_resultsperpseudoreadflat.data(),
            d_resultsperpseudoreadflat.data(),
            flatbuffers2size,
            D2H,
            stream
        ));

        //std::cerr << "exit thread " << std::this_thread::get_id() << "\n";
    }

    std::vector<extension::ExtendResult> convertRawExtendResults(const RawExtendResult& rawResults) const{
        nvtx::ScopedRange sr("convertRawExtendResults", 7);

        std::vector<extension::ExtendResult> gpuResultVector(rawResults.numResults);

        if(!rawResults.noReadIsExtended){

            for(int k = 0; k < rawResults.numResults; k++){
                auto& gpuResult = gpuResultVector[k];

                const int index = k;

                const char* gpuSeq = &rawResults.h_pairResultSequences[k * rawResults.outputpitch];
                const char* gpuQual = &rawResults.h_pairResultQualities[k * rawResults.outputpitch];
                const int gpuLength = rawResults.h_pairResultLengths[k];
                const int read1begin = rawResults.h_pairResultRead1Begins[k];
                const int read2begin = rawResults.h_pairResultRead2Begins[k];
                const bool anchorIsLR = rawResults.h_pairResultAnchorIsLR[k]; 
                const bool mateHasBeenFound = rawResults.h_pairResultMateHasBeenFound[k];
                const bool mergedDifferentStrands = rawResults.h_pairResultMergedDifferentStrands[k];

                std::string s1(gpuSeq, gpuLength);
                std::string s2(gpuQual, gpuLength);

                const int i0 = 4 * index + 0;
                const int i2 = 4 * index + 2;

                int srcindex = i0;
                if(!anchorIsLR){
                    srcindex = i2;
                }

                if(mateHasBeenFound){
                    gpuResult.abortReason = extension::AbortReason::None;
                }else{
                    gpuResult.abortReason = rawResults.h_gpuabortReasons[srcindex];
                }

                gpuResult.direction = anchorIsLR ? extension::ExtensionDirection::LR : extension::ExtensionDirection::RL;
                gpuResult.numIterations = rawResults.h_gpuiterations[srcindex];
                gpuResult.aborted = gpuResult.abortReason != extension::AbortReason::None;
                gpuResult.readId1 = rawResults.h_gpuReadIds[srcindex];
                gpuResult.readId2 = rawResults.h_gpuMateReadIds[srcindex];
                gpuResult.originalLength = rawResults.h_gpuAnchorLengths[srcindex];
                gpuResult.originalMateLength = rawResults.h_gpuMateLengths[srcindex];
                gpuResult.read1begin = read1begin;
                gpuResult.goodscore = rawResults.h_gpugoodscores[srcindex];
                gpuResult.read2begin = read2begin;
                gpuResult.mateHasBeenFound = mateHasBeenFound;
                gpuResult.extendedRead = std::move(s1);
                gpuResult.qualityScores = std::move(s2);
                gpuResult.mergedFromReadsWithoutMate = mergedDifferentStrands;
            }
        }else{
            for(int p = 0; p < rawResults.numResults; p++){
                //LR search
                const int i0 = 4 * p + 0;

                auto& result = gpuResultVector[p];

                result.direction = extension::ExtensionDirection::LR;
                result.numIterations = rawResults.h_gpuiterations[i0];
                result.aborted = rawResults.h_gpuabortReasons[i0] != extension::AbortReason::None;
                result.abortReason = rawResults.h_gpuabortReasons[i0];
                result.readId1 = rawResults.h_gpuReadIds[i0];
                result.readId2 = rawResults.h_gpuMateReadIds[i0];
                result.originalLength = rawResults.h_gpuAnchorLengths[i0];
                result.originalMateLength = rawResults.h_gpuMateLengths[i0];
                result.read1begin = 0;
                result.goodscore = rawResults.h_gpugoodscores[i0];
                result.read2begin = -1;
                result.mateHasBeenFound = false;
                result.extendedRead.assign(
                    rawResults.h_inputAnchorsDecoded.begin() + i0 * rawResults.decodedSequencePitchInBytes,
                    rawResults.h_inputAnchorsDecoded.begin() + i0 * rawResults.decodedSequencePitchInBytes + rawResults.h_gpuAnchorLengths[i0]
                );
                result.qualityScores.resize(rawResults.h_gpuAnchorLengths[i0]);
                std::fill(result.qualityScores.begin(), result.qualityScores.end(), 'I');
            }
        }

        return gpuResultVector;
    }




    //helpers

    void loadCandidateQualityScores(cudaStream_t stream, char* d_qualityscores){
        char* outputQualityScores = d_qualityscores;

        if(programOptions->useQualityScores){

            CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));

            gpuReadStorage->gatherQualities(
                readStorageHandle,
                outputQualityScores,
                qualityPitchInBytes,
                makeAsyncConstBufferWrapper(h_candidateReadIds.data(), h_candidateReadIdsEvent),
                d_candidateReadIds.data(),
                *h_numCandidates,
                stream,
                mr
            );

        }else{
            thrust::fill_n(
                rmm::exec_policy_nosync(stream, mr),
                outputQualityScores, 
                qualityPitchInBytes * initialNumCandidates, 
                'I'
            );
        }        
    }

    void compactCandidateDataByFlagsExcludingAlignments(
        const bool* d_keepFlags,
        bool updateHostCandidateReadIds,
        cudaStream_t stream
    ){
        rmm::device_uvector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, mr);

        CubCallWrapper(mr).cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        rmm::device_uvector<int> d_candidateSequencesLength2(initialNumCandidates, stream, mr);
        rmm::device_uvector<read_number> d_candidateReadIds2(initialNumCandidates, stream, mr);
        rmm::device_uvector<bool> d_isPairedCandidate2(initialNumCandidates, stream, mr);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));
        const int currentNumCandidates = *h_numCandidates;

        //compact 1d arrays

        CubCallWrapper(mr).cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            initialNumCandidates, 
            stream
        );

        CUDACHECK(cudaEventRecordWrapper(h_numCandidatesEvent, stream));

        if(updateHostCandidateReadIds){
            CUDACHECK(cudaStreamWaitEventWrapper(hostOutputStream, h_numCandidatesEvent, 0));      

            CUDACHECK(cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds2.data(),
                sizeof(read_number) * currentNumCandidates,
                D2H,
                hostOutputStream
            ));

            CUDACHECK(cudaEventRecordWrapper(h_candidateReadIdsEvent, hostOutputStream));  
        }

        CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream));
        CubCallWrapper(mr).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(), 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        destroy(d_numCandidatesPerAnchor2, stream);

        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        destroy(d_candidateSequencesLength2, stream);
        destroy(d_candidateReadIds2, stream);
        destroy(d_isPairedCandidate2, stream);
        
        //update candidate sequences data
        rmm::device_uvector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * initialNumCandidates, stream, mr);

        CubCallWrapper(mr).cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            initialNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        destroy(d_candidateSequencesData2, stream);
    }


    void compactCandidateDataByFlags(
        const bool* d_keepFlags,
        bool updateHostCandidateReadIds,
        cudaStream_t stream
    ){

        DEBUGSTREAMSYNC(stream);

        rmm::device_uvector<int> d_numCandidatesPerAnchor2(tasks->size(), stream, mr);

        DEBUGSTREAMSYNC(stream);

        CubCallWrapper(mr).cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            tasks->size(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        DEBUGSTREAMSYNC(stream);

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps.data(),
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        DEBUGSTREAMSYNC(stream);

        rmm::device_uvector<int> d_alignment_overlaps2(initialNumCandidates, stream, mr);
        rmm::device_uvector<int> d_alignment_shifts2(initialNumCandidates, stream, mr);
        rmm::device_uvector<int> d_alignment_nOps2(initialNumCandidates, stream, mr);
        rmm::device_uvector<AlignmentOrientation> d_alignment_best_alignment_flags2(initialNumCandidates, stream, mr);
        rmm::device_uvector<int> d_candidateSequencesLength2(initialNumCandidates, stream, mr);
        rmm::device_uvector<read_number> d_candidateReadIds2(initialNumCandidates, stream, mr);
        rmm::device_uvector<bool> d_isPairedCandidate2(initialNumCandidates, stream, mr);

        DEBUGSTREAMSYNC(stream);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps2.data(),
                d_alignment_overlaps2.data(),
                d_alignment_shifts2.data(),
                d_alignment_best_alignment_flags2.data(),
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        CUDACHECK(cudaEventSynchronizeWrapper(h_numCandidatesEvent));
        const int currentNumCandidates = *h_numCandidates;

        //compact 1d arrays

        CubCallWrapper(mr).cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            initialNumCandidates, 
            stream
        );

        CUDACHECK(cudaEventRecordWrapper(h_numCandidatesEvent, stream));

        if(updateHostCandidateReadIds){
            CUDACHECK(cudaStreamWaitEventWrapper(hostOutputStream, h_numCandidatesEvent, 0));           

            CUDACHECK(cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds2.data(),
                sizeof(read_number) * currentNumCandidates,
                D2H,
                hostOutputStream
            ));

            CUDACHECK(cudaEventRecordWrapper(h_candidateReadIdsEvent, hostOutputStream));  
        }

        CUDACHECK(cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream));
        CubCallWrapper(mr).cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            tasks->size(), 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        destroy(d_numCandidatesPerAnchor2, stream);

        std::swap(d_alignment_nOps, d_alignment_nOps2);
        std::swap(d_alignment_overlaps, d_alignment_overlaps2);
        std::swap(d_alignment_shifts, d_alignment_shifts2);
        std::swap(d_alignment_best_alignment_flags, d_alignment_best_alignment_flags2);
        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        destroy(d_alignment_overlaps2, stream);
        destroy(d_alignment_shifts2, stream);
        destroy(d_alignment_nOps2, stream);
        destroy(d_alignment_best_alignment_flags2, stream);
        destroy(d_candidateSequencesLength2, stream);
        destroy(d_candidateReadIds2, stream);
        destroy(d_isPairedCandidate2, stream);
        
        //update candidate sequences data
        rmm::device_uvector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * initialNumCandidates, stream, mr);

        CubCallWrapper(mr).cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            initialNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        destroy(d_candidateSequencesData2, stream);

        //update candidate quality scores
        // assert(qualityPitchInBytes % sizeof(int) == 0);
        // rmm::device_uvector<char> d_candidateQualities2(qualityPitchInBytes * initialNumCandidates, stream, mr);

        // cubSelectFlagged(
        //     (const int*)d_candidateQualityScores.data(),
        //     thrust::make_transform_iterator(
        //         thrust::make_counting_iterator(0),
        //         make_iterator_multiplier(d_keepFlags, qualityPitchInBytes / sizeof(int))
        //     ),
        //     (int*)d_candidateQualities2.data(),
        //     thrust::make_discard_iterator(),
        //     initialNumCandidates * qualityPitchInBytes / sizeof(int),
        //     firstStream
        // );

        // std::swap(d_candidateQualityScores, d_candidateQualities2);
    }

    void setStateToFinished(cudaStream_t stream){
        tasks->clearBuffers(stream);

        CUDACHECK(cudaStreamSynchronizeWrapper(stream));

        setState(GpuReadExtender::State::Finished);
    }
    
    void addFinishedGpuSoaTasks(TaskData& tasksToAdd, TaskData& finishedTasks, cudaStream_t stream) const{
        finishedTasks.append(tasksToAdd, stream);
        //std::cerr << "addFinishedSoaTasks. soaFinishedTasks size " << soaFinishedTasks.entries << "\n";
    }

    void copyHashResultsFrom(const AnchorHashResult& results, cudaStream_t stream){
        ::resizeUninitialized(d_candidateReadIds, results.d_candidateReadIds.size(), stream);
        ::resizeUninitialized(d_numCandidatesPerAnchor, results.d_numCandidatesPerAnchor.size(), stream);
        ::resizeUninitialized(d_numCandidatesPerAnchorPrefixSum, results.d_numCandidatesPerAnchorPrefixSum.size(), stream);

        CUDACHECK(cudaMemcpyAsync(
            d_candidateReadIds.data(),
            results.d_candidateReadIds.data(),
            sizeof(read_number) * d_candidateReadIds.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_numCandidatesPerAnchor.data(),
            results.d_numCandidatesPerAnchor.data(),
            sizeof(int) * d_numCandidatesPerAnchor.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_numCandidatesPerAnchorPrefixSum.data(),
            results.d_numCandidatesPerAnchorPrefixSum.data(),
            sizeof(int) * d_numCandidatesPerAnchorPrefixSum.size(),
            D2D,
            stream
        ));
    }

    void copyAnchorDataFrom(const AnchorData& results, cudaStream_t stream){
        ::resizeUninitialized(d_anchorSequencesDataDecoded, results.d_anchorSequencesDataDecoded.size(), stream);
        ::resizeUninitialized(d_anchorQualityScores, results.d_anchorQualityScores.size(), stream);
        ::resizeUninitialized(d_anchorSequencesLength, results.d_anchorSequencesLength.size(), stream);
        ::resizeUninitialized(d_anchorSequencesData, results.d_anchorSequencesData.size(), stream);
        ::resizeUninitialized(d_accumExtensionsLengths, results.d_accumExtensionsLengths.size(), stream);

        CUDACHECK(cudaMemcpyAsync(
            d_anchorSequencesDataDecoded.data(),
            results.d_anchorSequencesDataDecoded.data(),
            sizeof(char) * d_anchorSequencesDataDecoded.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_anchorQualityScores.data(),
            results.d_anchorQualityScores.data(),
            sizeof(char) * d_anchorQualityScores.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_anchorSequencesLength.data(),
            results.d_anchorSequencesLength.data(),
            sizeof(int) * d_anchorSequencesLength.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_anchorSequencesData.data(),
            results.d_anchorSequencesData.data(),
            sizeof(unsigned int) * d_anchorSequencesData.size(),
            D2D,
            stream
        ));

        CUDACHECK(cudaMemcpyAsync(
            d_accumExtensionsLengths.data(),
            results.d_accumExtensionsLengths.data(),
            sizeof(int) * d_accumExtensionsLengths.size(),
            D2D,
            stream
        ));
    }

    bool pairedEnd = false;
    State state = State::None;
    int someId = 0;
    int alltimeMaximumNumberOfTasks = 0;
    std::size_t alltimetotalTaskBytes = 0;

    int initialNumCandidates = 0;

    int deviceId{};

    rmm::mr::device_memory_resource* mr{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const ProgramOptions* programOptions{};
    const cpu::QualityScoreConversion* qualityConversion{};
    mutable ReadStorageHandle readStorageHandle{};

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    std::size_t outputAnchorPitchInBytes = 0;
    std::size_t outputAnchorQualityPitchInBytes = 0;
    std::size_t decodedMatesRevCPitchInBytes = 0;

    const IterationConfig* iterationConfig{};

    PinnedBuffer<char> h_tempForMemcopies{TaskData::getHostTempStorageSize()};
    
    PinnedBuffer<read_number> h_candidateReadIds{};
    PinnedBuffer<int> h_numCandidatesPerAnchor{};
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};
    PinnedBuffer<int> h_numAnchors{};
    PinnedBuffer<int> h_numCandidates{};
    PinnedBuffer<int> h_numAnchorsWithRemovedMates{};
    PinnedBuffer<int> h_minmax{};
    PinnedBuffer<int> h_numPositions{};


    void destroyDeviceBuffers(cudaStream_t stream){

        ::destroy(d_candidateSequencesData, stream);
        ::destroy(d_candidateSequencesLength, stream);
        ::destroy(d_candidateReadIds, stream);
        ::destroy(d_isPairedCandidate, stream);
        ::destroy(d_alignment_overlaps, stream);
        ::destroy(d_alignment_shifts, stream);
        ::destroy(d_alignment_nOps, stream);
        ::destroy(d_alignment_best_alignment_flags, stream);
        ::destroy(d_numCandidatesPerAnchor, stream);
        ::destroy(d_numCandidatesPerAnchorPrefixSum, stream);
        ::destroy(d_anchorSequencesDataDecoded, stream);
        ::destroy(d_anchorQualityScores, stream);
        ::destroy(d_anchorSequencesLength, stream);
        ::destroy(d_anchorSequencesData, stream);
        ::destroy(d_accumExtensionsLengths, stream);

        multiMSA.destroy(stream);

        CUDACHECK(cudaStreamSynchronize(stream));
    }

    rmm::device_uvector<bool> d_mateIdHasBeenRemoved;
    // ----- candidate data
    rmm::device_uvector<unsigned int> d_candidateSequencesData;
    rmm::device_uvector<int> d_candidateSequencesLength;    
    rmm::device_uvector<read_number> d_candidateReadIds;
    rmm::device_uvector<bool> d_isPairedCandidate;
    rmm::device_uvector<int> d_alignment_overlaps;
    rmm::device_uvector<int> d_alignment_shifts;
    rmm::device_uvector<int> d_alignment_nOps;
    rmm::device_uvector<AlignmentOrientation> d_alignment_best_alignment_flags;

    rmm::device_uvector<int> d_numCandidatesPerAnchor;
    rmm::device_uvector<int> d_numCandidatesPerAnchorPrefixSum;
    // ----- 
    
    // ----- input data
    rmm::device_uvector<char> d_anchorSequencesDataDecoded;
    rmm::device_uvector<char> d_anchorQualityScores;
    rmm::device_uvector<int> d_anchorSequencesLength;
    rmm::device_uvector<unsigned int> d_anchorSequencesData;
    rmm::device_uvector<int> d_accumExtensionsLengths;
    // -----

    
    // ----- MSA data
    gpu::ManagedGPUMultiMSA multiMSA;
    // -----


    // ----- Ready-events for pinned outputs
    CudaEvent h_numAnchorsEvent{};
    CudaEvent h_numCandidatesEvent{};
    CudaEvent h_numAnchorsWithRemovedMatesEvent{};
    CudaEvent h_numUsedReadIdsEvent{};
    CudaEvent h_candidateReadIdsEvent{};

    // -----

    CudaStream hostOutputStream{};

    readextendergpukernels::ComputeTaskSplitGatherIndicesSmallInput computeTaskSplitGatherIndicesSmallInput{};
    
    std::array<CudaEvent, 1> events{};

    TaskData* tasks = nullptr;
    
    mutable std::mutex mutex{};
};


} //namespace gpu
} //namespace care


#endif