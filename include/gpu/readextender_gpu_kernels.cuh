#ifndef CARE_READEXTENDER_GPU_KERNELS_CUH
#define CARE_READEXTENDER_GPU_KERNELS_CUH

#include <readextender_common.hpp>

#include <gpu/cudaerrorcheck.cuh>
#include <gpu/groupmemcpy.cuh>
#include <msasplits.hpp>

#include <cassert>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace care{
namespace gpu{



namespace readextendergpukernels{

    //Utility


    template<class Iter1, class Iter2, class Iter3>
    __global__
    void vectorAddKernel(Iter1 input1, Iter2 input2, Iter3 output, int N){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(output + i) = *(input1 + i) + *(input2 + i);
        }
    }


    template<class InputIter, class ConstantIter, class OutputIter>
    __global__
    void vectorAddConstantKernel(InputIter input1, ConstantIter constantiter, OutputIter output, int N){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(output + i) = *(input1 + i) + *constantiter;
        }
    }

    template<class Sum, class T, class U>
    struct ConstantSumIterator{
        using value_type = Sum;

        T input1;
        U input2;

        __host__ __device__
        ConstantSumIterator(T i1, U i2) : input1(i1), input2(i2){}

        __host__ __device__
        value_type operator[](std::size_t i) const{
            return operator*();
        }

        __host__ __device__
        value_type operator*() const{
            return (*input1) + (*input2);
        }

        __host__ __device__
        value_type operator->() const{
            return (*input1) + (*input2);
        }        
    };

    template<class Sum, class T, class U>
    ConstantSumIterator<Sum, T, U> makeConstantSumIterator(T i1, U i2) {
        return ConstantSumIterator<Sum, T, U>{i1, i2};
    }


    template<class Iter1, class T>
    __global__
    void iotaKernel(Iter1 outputbegin, Iter1 outputend, T init){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        const int N = outputend - outputbegin;

        for(int i = tid; i < N; i += stride){
            *(outputbegin + i) = init + static_cast<T>(i);
        }
    }

    template<int groupsize, class Iter, class SizeIter, class OffsetIter>
    __global__
    void segmentedIotaKernel(
        Iter outputbegin, 
        int numSegments, 
        SizeIter segmentsizes, 
        OffsetIter segmentBeginOffsets
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        
        const int numGroups = stride / groupsize;
        const int groupId = tid / groupsize;

        for(int s = groupId; s < numSegments; s += numGroups){
            const int num = segmentsizes[s];
            const int offset = segmentBeginOffsets[s];
            
            for(int i = group.thread_rank(); i < num; i += group.size()){
                *(outputbegin + offset + i) = i;
            }
        }
    }

    template<class Iter1, class T>
    __global__
    void fillKernel(Iter1 begin, int N, T value){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(begin + i) = value;
        }
    }

    template<int blocksize, class InputIter, class OutputIter>
    __global__
    void minmaxSingleBlockKernel(InputIter begin, int N, OutputIter minmax){
        using value_type_in = typename std::iterator_traits<InputIter>::value_type;
        using value_type_out = typename std::iterator_traits<OutputIter>::value_type;
        static_assert(std::is_same_v<value_type_in, value_type_out>);

        using value_type = value_type_in;

        using BlockReduce = cub::BlockReduce<value_type, blocksize>;
        __shared__ typename BlockReduce::TempStorage temp1;
        __shared__ typename BlockReduce::TempStorage temp2;

        if(blockIdx.x == 0){

            const int tid = threadIdx.x;
            const int stride = blockDim.x;

            value_type myMin = std::numeric_limits<value_type>::max();
            value_type myMax = 0;

            for(int i = tid; i < N; i += stride){
                const value_type val = *(begin + i);
                myMin = min(myMin, val);
                myMax = max(myMax, val);
            }

            myMin = BlockReduce(temp1).Reduce(myMin, cub::Min{});
            myMax = BlockReduce(temp2).Reduce(myMax, cub::Max{});

            if(tid == 0){
                *(minmax + 0) = myMin;
                *(minmax + 1) = myMax;
            }
        }
    }


    //extender kernels


    template<int blocksize, int groupsize, class MapIter>
    __global__
    void taskGatherKernel1(
        MapIter d_mapBegin,
        MapIter d_mapEnd,
        int gathersize,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        std::size_t encodedSequencePitchInInts,
        std::size_t extendedSequencePitchInBytes,
        char* __restrict__ selection_extendedSequences,
        const char* __restrict__ extendedSequences,
        char* __restrict__ selection_qualitiesOfExtendedSequences,
        const char* __restrict__ qualitiesOfExtendedSequences,
        unsigned int* __restrict__ selection_inputEncodedMate,
        const unsigned int* __restrict__ inputEncodedMate,
        unsigned int* __restrict__ selection_inputAnchorsEncoded,
        const unsigned int* __restrict__ inputAnchorsEncoded,
        char* __restrict__ selection_inputAnchorQualities,
        const char* __restrict__ inputAnchorQualities,
        char* __restrict__ selection_inputMateQualities,
        const char* __restrict__ inputMateQualities
    ){

        for(int i = blockIdx.x; i < gathersize; i += gridDim.x){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;

            for(int k = threadIdx.x; k < encodedSequencePitchInInts; k += blockDim.x){
                selection_inputEncodedMate[destindex * encodedSequencePitchInInts + k]
                    = inputEncodedMate[srcindex * encodedSequencePitchInInts + k];

                selection_inputAnchorsEncoded[destindex * encodedSequencePitchInInts + k]
                    = inputAnchorsEncoded[srcindex * encodedSequencePitchInInts + k];
            }

            for(int k = threadIdx.x; k < extendedSequencePitchInBytes; k += blockDim.x){
                selection_extendedSequences[destindex * extendedSequencePitchInBytes + k] 
                    = extendedSequences[srcindex * extendedSequencePitchInBytes + k];
            }
            for(int k = threadIdx.x; k < extendedSequencePitchInBytes; k += blockDim.x){
                selection_qualitiesOfExtendedSequences[destindex * extendedSequencePitchInBytes + k] 
                    = qualitiesOfExtendedSequences[srcindex * extendedSequencePitchInBytes + k];
            }

            for(int k = threadIdx.x; k < qualityPitchInBytes; k += blockDim.x){
                selection_inputAnchorQualities[destindex * qualityPitchInBytes + k] 
                    = inputAnchorQualities[srcindex * qualityPitchInBytes + k];
            }
            for(int k = threadIdx.x; k < qualityPitchInBytes; k += blockDim.x){
                selection_inputMateQualities[destindex * qualityPitchInBytes + k] 
                    = inputMateQualities[srcindex * qualityPitchInBytes + k];
            }
        }
    }

    template<int blocksize, int groupsize, class MapIter>
    __global__
    void taskGatherKernel2(
        MapIter d_mapBegin,
        MapIter d_mapEnd,
        int gathersize,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        const int* __restrict__ selection_d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        read_number* __restrict__ selection_d_usedReadIds,
        const read_number* __restrict__ d_usedReadIds,
        const int* __restrict__  selection_d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__  d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numFullyUsedReadIdsPerTask,
        read_number* __restrict__ selection_d_fullyUsedReadIds,
        const read_number* __restrict__ d_fullyUsedReadIds
    ){
        for(int i = blockIdx.x; i < gathersize; i += gridDim.x){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;

            //used ids
            {
                int destoffset = selection_d_numUsedReadIdsPerTaskPrefixSum[destindex];
                int srcoffset = d_numUsedReadIdsPerTaskPrefixSum[srcindex];
                int num = d_numUsedReadIdsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_d_usedReadIds[destoffset + k] 
                        = d_usedReadIds[srcoffset + k];
                }
            }

            //fully used ids
            {
                int destoffset = selection_d_numFullyUsedReadIdsPerTaskPrefixSum[destindex];
                int srcoffset = d_numFullyUsedReadIdsPerTaskPrefixSum[srcindex];
                int num = d_numFullyUsedReadIdsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_d_fullyUsedReadIds[destoffset + k] 
                        = d_fullyUsedReadIds[srcoffset + k];
                }
            }
        }
    }

    template<int blocksize, int groupsize, class MapIter>
    __global__
    void taskGatherKernel2(
        MapIter d_mapBegin,
        MapIter d_mapEnd,
        int gathersize,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        const int* __restrict__ selection_d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        read_number* __restrict__ selection_d_usedReadIds,
        const read_number* __restrict__ d_usedReadIds
    ){
        for(int i = blockIdx.x; i < gathersize; i += gridDim.x){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;

           
            //used ids
            {
                int destoffset = selection_d_numUsedReadIdsPerTaskPrefixSum[destindex];
                int srcoffset = d_numUsedReadIdsPerTaskPrefixSum[srcindex];
                int num = d_numUsedReadIdsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_d_usedReadIds[destoffset + k] 
                        = d_usedReadIds[srcoffset + k];
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void taskFixAppendedPrefixSumsKernel(
        int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        int* __restrict__ d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        const int* __restrict__ d_numFullyUsedReadIdsPerTask,
        int size,
        int rhssize
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        if(size == 0){
            if(tid == 0){
                d_numUsedReadIdsPerTaskPrefixSum[rhssize] 
                    = d_numUsedReadIdsPerTaskPrefixSum[rhssize-1] + d_numUsedReadIdsPerTask[rhssize-1];
                d_numFullyUsedReadIdsPerTaskPrefixSum[rhssize] 
                    = d_numFullyUsedReadIdsPerTaskPrefixSum[rhssize-1] + d_numFullyUsedReadIdsPerTask[rhssize-1];
            }
        }else{
            for(int i = tid; i < rhssize+1; i += stride){
                d_numUsedReadIdsPerTaskPrefixSum[size + i] 
                    += d_numUsedReadIdsPerTaskPrefixSum[size-1] + d_numUsedReadIdsPerTask[size - 1];
                d_numFullyUsedReadIdsPerTaskPrefixSum[size + i] 
                    += d_numFullyUsedReadIdsPerTaskPrefixSum[size-1] + d_numFullyUsedReadIdsPerTask[size - 1];
            }
        }
    }

    template<int blocksize>
    __global__
    void taskFixAppendedPrefixSumsKernel(
        int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        int size,
        int rhssize
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        if(size == 0){
            if(tid == 0){
                d_numUsedReadIdsPerTaskPrefixSum[rhssize] 
                    = d_numUsedReadIdsPerTaskPrefixSum[rhssize-1] + d_numUsedReadIdsPerTask[rhssize-1];
            }
        }else{
            for(int i = tid; i < rhssize+1; i += stride){
                d_numUsedReadIdsPerTaskPrefixSum[size + i] 
                    += d_numUsedReadIdsPerTaskPrefixSum[size-1] + d_numUsedReadIdsPerTask[size - 1];
            }
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void taskAddIterationResultsKernel(
        int numTasks,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        std::size_t addSequencesPitchInBytes,
        std::size_t addQualityPitchInBytes,
        const int* __restrict__ newNumEntriesPerTaskPrefixSum,
        char* __restrict__ newsoatotalDecodedAnchorsFlat,
        char* __restrict__ newsoatotalAnchorQualityScoresFlat,
        int* __restrict__ newsoatotalDecodedAnchorsLengths,
        int* __restrict__ newsoatotalAnchorBeginInExtendedRead,
        const int* __restrict__ soaNumIterationResultsPerTask,
        const int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        const int* __restrict__ soatotalDecodedAnchorsLengths,
        const int* __restrict__ soatotalAnchorBeginInExtendedRead,
        const char* __restrict__ soatotalDecodedAnchorsFlat,
        const char* __restrict__ soatotalAnchorQualityScoresFlat,
        const int* __restrict__ addNumEntriesPerTask,
        const int* __restrict__ addNumEntriesPerTaskPrefixSum,
        const int* __restrict__ addTotalDecodedAnchorsLengths,
        const int* __restrict__ addTotalAnchorBeginInExtendedRead,
        const char* __restrict__ addTotalDecodedAnchorsFlat,
        const char* __restrict__ addTotalAnchorQualityScoresFlat
    ){
        for(int i = blockIdx.x; i < numTasks; i += gridDim.x){
            //copy current data to new buffer
            const int currentnum = soaNumIterationResultsPerTask[i];
            const int currentoffset = soaNumIterationResultsPerTaskPrefixSum[i];

            const int newoffset = newNumEntriesPerTaskPrefixSum[i];

            for(int k = threadIdx.x; k < currentnum; k += blockDim.x){
                newsoatotalDecodedAnchorsLengths[newoffset + k] = soatotalDecodedAnchorsLengths[currentoffset + k];
                newsoatotalAnchorBeginInExtendedRead[newoffset + k] = soatotalAnchorBeginInExtendedRead[currentoffset + k];
            }

            for(int k = threadIdx.x; k < decodedSequencePitchInBytes * currentnum; k += blockDim.x){
                newsoatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * newoffset + k]
                    = soatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * currentoffset + k];
            }

            for(int k = threadIdx.x; k < qualityPitchInBytes * currentnum; k += blockDim.x){
                newsoatotalAnchorQualityScoresFlat[qualityPitchInBytes * newoffset + k]
                    = soatotalAnchorQualityScoresFlat[qualityPitchInBytes * currentoffset + k];
            }

            //copy add data to new buffer
            const int addnum = addNumEntriesPerTask[i];
            if(addnum > 0){
                const int addoffset = addNumEntriesPerTaskPrefixSum[i];

                for(int k = threadIdx.x; k < addnum; k += blockDim.x){
                    newsoatotalDecodedAnchorsLengths[(newoffset + currentnum) + k] = addTotalDecodedAnchorsLengths[addoffset + k];
                    newsoatotalAnchorBeginInExtendedRead[(newoffset + currentnum) + k] = addTotalAnchorBeginInExtendedRead[addoffset + k];
                }

                for(int k = 0; k < addnum; k++){
                    for(int l = threadIdx.x; l < addSequencesPitchInBytes; l += blockDim.x){
                        newsoatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * (newoffset + currentnum + k) + l]
                            = addTotalDecodedAnchorsFlat[addSequencesPitchInBytes * (addoffset + k) + l];
                    }  

                    for(int l = threadIdx.x; l < addQualityPitchInBytes; l += blockDim.x){
                        newsoatotalAnchorQualityScoresFlat[qualityPitchInBytes * (newoffset + currentnum + k) + l]
                            = addTotalAnchorQualityScoresFlat[addQualityPitchInBytes * (addoffset + k) + l];
                    }                       
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void taskComputeActiveFlagsKernel(
        int numTasks,
        int minFragmentSize,
        int maxFragmentSize,
        bool* __restrict__ d_flags,
        const int* __restrict__ iteration,
        const AbortReason* __restrict__ abortReason,
        const bool* __restrict__ mateHasBeenFound,
        const int* __restrict__ extendedSequenceLengths        
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numTasks; i+= stride){
            const int currentExtendedLength = extendedSequenceLengths[i];

            d_flags[i] = (iteration[i] < minFragmentSize 
                && currentExtendedLength < maxFragmentSize
                && (abortReason[i] == AbortReason::None) 
                && !mateHasBeenFound[i]
            );
        }
    }

    template<int blocksize>
    __global__
    void taskUpdateScalarIterationResultsKernel(
        int numTasks,
        float* __restrict__ task_goodscore,
        AbortReason* __restrict__ task_abortReason,
        bool* __restrict__ task_mateHasBeenFound,
        const float* __restrict__ d_goodscores,
        const AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_mateHasBeenFound
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        for(int i = tid; i < numTasks; i += stride){
            task_goodscore[i] += d_goodscores[i];
            task_abortReason[i] = d_abortReasons[i];
            task_mateHasBeenFound[i] = d_mateHasBeenFound[i];
        }
    }
        
    template<int blocksize>
    __global__
    void taskIncrementIterationKernel(
        int numTasks,
        const ExtensionDirection* __restrict__ task_direction,
        const bool* __restrict__ task_pairedEnd,
        const bool* __restrict__ task_mateHasBeenFound,
        const int* __restrict__ task_pairId,
        const int* __restrict__ task_id,
        AbortReason* __restrict__ task_abortReason,
        int* __restrict__ task_iteration
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        constexpr bool disableOtherStrand = false;
        constexpr bool disableSecondaryIfPrimaryFindMate = true;
        constexpr bool debugprint = false;

        for(int i = tid; i < numTasks; i += stride){
            task_iteration[i]++;
            
            const int whichtype = task_id[i] % 4;

            if(whichtype == 0){
                assert(task_direction[i] == ExtensionDirection::LR);
                assert(task_pairedEnd[i] == true);

                if(task_mateHasBeenFound[i]){
                    for(int k = 1; k <= 4; k++){
                        if(i+k < numTasks){
                            if(task_pairId[i + k] == task_pairId[i]){
                                if(task_id[i + k] == task_id[i] + 1){
                                    //disable LR partner task
                                    if constexpr (disableSecondaryIfPrimaryFindMate){
                                        if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                        task_abortReason[i + k] = AbortReason::PairedAnchorFinished;
                                    }
                                }else if(task_id[i+k] == task_id[i] + 2){
                                    //disable RL search task
                                    if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                    if constexpr (disableOtherStrand){
                                        task_abortReason[i + k] = AbortReason::OtherStrandFoundMate;
                                    }
                                }
                            }else{
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }else if(task_abortReason[i] != AbortReason::None){
                    for(int k = 1; k <= 4; k++){
                        if(i+k < numTasks){
                            if(task_pairId[i + k] == task_pairId[i]){
                                if(task_id[i + k] == task_id[i] + 1){
                                    //disable LR partner task  
                                    if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                    task_abortReason[i + k] = AbortReason::PairedAnchorFinished;
                                    break;
                                }
                            }else{
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }
            }else if(whichtype == 2){
                assert(task_direction[i] == ExtensionDirection::RL);
                assert(task_pairedEnd[i] == true);

                if(task_mateHasBeenFound[i]){
                    if(i+1 < numTasks){
                        if(task_pairId[i + 1] == task_pairId[i]){
                            if(task_id[i + 1] == task_id[i] + 1){
                                //disable RL partner task
                                if constexpr (disableSecondaryIfPrimaryFindMate){
                                    if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                    task_abortReason[i + 1] = AbortReason::PairedAnchorFinished;
                                }
                            }
                        }

                        for(int k = 1; k <= 2; k++){
                            if(i - k >= 0){
                                if(task_pairId[i - k] == task_pairId[i]){
                                    if(task_id[i - k] == task_id[i] - 2){
                                        //disable LR search task
                                        if constexpr (disableOtherStrand){
                                            if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                            task_abortReason[i - k] = AbortReason::OtherStrandFoundMate;
                                        }
                                    }
                                }else{
                                    break;
                                }
                            }else{
                                break;
                            }
                        }
                    }
                    
                }else if(task_abortReason[i] != AbortReason::None){
                    if(i+1 < numTasks){

                        if(task_pairId[i + 1] == task_pairId[i]){
                            if(task_id[i + 1] == task_id[i] + 1){
                                //disable RL partner task
                                if constexpr (debugprint) printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task_mateHasBeenFound[i], task_abortReason[i], disableOtherStrand);
                                task_abortReason[i + 1] = AbortReason::PairedAnchorFinished;
                            }
                        }

                    }
                }
            }
        }
    }


    /*
        For each string with flag == true, copy its last N characters. 
        If string length is less than N the full string is copied.
        If flag == false, outputLength is set to 0
    */
    template<int blocksize, class Flags>
    __global__
    void copyLastNCharactersOfStrings(
        int numStrings,
        const char* __restrict__ inputStrings,
        const int* __restrict__ inputLengths,
        std::size_t inputPitchInBytes,
        char* __restrict__ outputStrings,
        int* __restrict__ outputLengths,
        std::size_t outputPitchInBytes,
        Flags flags,
        int N
    ){
        static_assert(blocksize >= 32, "");

        auto tile = cg::tiled_partition<32>(cg::this_thread_block());
        const int tileId = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
        const int numTiles = (blockDim.x * gridDim.x) / 32;

        for(int s = tileId; s < numStrings; s += numTiles){
            if(flags[s]){
                const int inputLength = inputLengths[s];
                const char* const inputString = inputStrings + s * inputPitchInBytes;
                char* const outputString = outputStrings + s * outputPitchInBytes;

                const int end = inputLength;
                const int begin = max(0, inputLength - N);

                outputLengths[s] = end - begin;
                care::gpu::memcpy<int>(tile, outputString, inputString + begin, sizeof(char) * (end - begin));
            }else{
                outputLengths[s] = 0;
            }
        }

    }


    template<int blocksize>
    __global__
    void flagFullyUsedCandidatesKernel(
        const int* ids,
        const int* iterations,
        int numTasks,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ d_candidateSequencesLengths,
        const int* __restrict__ d_alignment_shifts,
        const int* __restrict__ d_anchorSequencesLength,
        const int* __restrict__ d_oldaccumExtensionsLengths,
        const int* __restrict__ d_newaccumExtensionsLengths,
        const AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_outputMateHasBeenFound,
        bool* __restrict__ d_isFullyUsedCandidate
    ){
        // d_isFullyUsedCandidate must be initialized with 0

        for(int task = blockIdx.x; task < numTasks; task += gridDim.x){
            const int numCandidates = d_numCandidatesPerAnchor[task];
            const auto abortReason = d_abortReasons[task];

            if(numCandidates > 0 && abortReason == AbortReason::None){
                const int anchorLength = d_anchorSequencesLength[task];
                const int offset = d_numCandidatesPerAnchorPrefixSum[task];
                const int oldAccumExtensionsLength = d_oldaccumExtensionsLengths[task];
                const int newAccumExtensionsLength = d_newaccumExtensionsLengths[task];
                const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

                for(int c = threadIdx.x; c < numCandidates; c += blockDim.x){
                    const int candidateLength = d_candidateSequencesLengths[offset + c];
                    const int shift = d_alignment_shifts[offset + c];
                    // if(ids[task] == 1 && iterations[task] == 0){
                    //     printf("%d %d, %d", c , shift, lengthOfExtension);
                    // }

                    if(candidateLength + shift <= anchorLength + lengthOfExtension){
                        d_isFullyUsedCandidate[offset + c] = true;
                        // if(ids[task] == 1 && iterations[task] == 0){
                        //     printf(", fully used\n");
                        // }
                    }else{
                        // if(ids[task] == 1 && iterations[task] == 0){
                        //     printf("\n");
                        // }
                    }
                }
            }
        }
    }


    template<int blocksize, int groupsize>
    __global__
    void updateWorkingSetFromTasksKernel(
        int numTasks,
        std::size_t qualityPitchInBytes,
        std::size_t decodedSequencePitchInBytes,
        int* __restrict__ d_anchorSequencesLength,
        char* __restrict__ d_anchorQualityScores,
        char* __restrict__ d_anchorSequencesDataDecoded,
        const int* __restrict__ originalReadLengths,
        const char* __restrict__ extendedSequences,
        const char* __restrict__ qualitiesOfExtendedSequences,
        const int* __restrict__ extendedSequenceLengths,
        int extendedSequencePitchInBytes
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = tid / groupsize;
        const int numGroups = stride / groupsize;

        for(int i = groupId; i < numTasks; i += numGroups){
            const int extendedLength = extendedSequenceLengths[i];
            const int origLength = originalReadLengths[i];
            const int copyBegin = std::max(0, extendedLength - origLength);
            const int copyLength = extendedLength - copyBegin;

            for(int k = group.thread_rank(); k < copyLength; k += group.size()){
                d_anchorSequencesDataDecoded[decodedSequencePitchInBytes * i + k]
                    = extendedSequences[extendedSequencePitchInBytes * i + copyBegin + k];
                d_anchorQualityScores[qualityPitchInBytes * i + k]
                    = qualitiesOfExtendedSequences[extendedSequencePitchInBytes * i + copyBegin + k];
            }

            if(group.thread_rank() == 0){
                d_anchorSequencesLength[i] = copyLength;
            }
        }
    }

    template<int blocksize>
    __global__
    void computeNumberOfSoaIterationResultsPerTaskKernel(
        int numTasks,
        int* __restrict__ d_addNumEntriesPerTask,
        int* __restrict__ d_addNumEntriesPerTaskPrefixSum,
        const AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_mateHasBeenFound,
        const int* __restrict__ d_sizeOfGapToMate
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        if(tid == 0){
            d_addNumEntriesPerTaskPrefixSum[0] = 0;
        }

        for(int i = tid; i < numTasks; i += stride){
            int num = 0;

            if(d_abortReasons[i] == AbortReason::None){
                num = 1;

                if(d_mateHasBeenFound[i]){
                    if(d_sizeOfGapToMate[i] != 0){
                        num = 2;
                    }
                }
            }

            d_addNumEntriesPerTask[i] = num;
        }
    }

    template<int blocksize>
    __global__
    void makeSoAIterationResultsKernel(
        const int* ids,
        const int* iterations,
        int numTasks,
        std::size_t outputAnchorPitchInBytes,
        std::size_t outputAnchorQualityPitchInBytes,
        const int* __restrict__ d_addNumEntriesPerTask,
        const int* __restrict__ d_addNumEntriesPerTaskPrefixSum,
        char* __restrict__ d_addTotalDecodedAnchorsFlat,
        char* __restrict__ d_addTotalAnchorQualityScoresFlat,
        int* __restrict__ d_addAnchorLengths,
        int* __restrict__ d_addAnchorBeginsInExtendedRead,
        std::size_t task_decodedSequencePitchInBytes,
        std::size_t task_qualityPitchInBytes,
        const AbortReason* __restrict__ task_abortReason,
        const bool* __restrict__ task_mateHasBeenFound,
        const char* __restrict__ task_materevc,
        const char* __restrict__ task_materevcqual,
        const int* __restrict__ task_matelength,
        const int* __restrict__ d_sizeOfGapToMate,
        const int* __restrict__ d_outputAnchorLengths,
        const char* __restrict__ d_outputAnchors,
        const char* __restrict__ d_outputAnchorQualities,
        const int* __restrict__ d_accumExtensionsLengths
    ){
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     for(int i = 0; i < numTasks; i += 1){
        //         printf("id: %d, iter: %d, mateHasBeenFound: %d, abort: %4d, newaccum: %d, newanchor: ", ids[i], iterations[i], 
        //             task_mateHasBeenFound[i], int(task_abortReason[i]), d_accumExtensionsLengths[i]);
                    
        //         for(int k = 0; k < d_outputAnchorLengths[i]; k++){
        //             printf("%c", d_outputAnchors[i * outputAnchorPitchInBytes + k]);
        //         }
        //         printf("\n");
        //     }
        // }

        for(int i = blockIdx.x; i < numTasks; i += gridDim.x){
            if(task_abortReason[i] == AbortReason::None){                
                const int offset = d_addNumEntriesPerTaskPrefixSum[i];

                if(!task_mateHasBeenFound[i]){
                    const int length = d_outputAnchorLengths[i];
                    d_addAnchorLengths[offset] = length;

                    //copy result
                    for(int k = threadIdx.x; k < length; k += blockDim.x){
                        d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = d_outputAnchors[i * outputAnchorPitchInBytes + k];
                        d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = d_outputAnchorQualities[i * outputAnchorQualityPitchInBytes + k];
                    }
                    d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];

                }else{
                    const int sizeofGap = d_sizeOfGapToMate[i];
                    if(sizeofGap == 0){                                
                        //copy mate revc

                        const int mateLength = task_matelength[i];
                        d_addAnchorLengths[offset] = mateLength;

                        for(int k = threadIdx.x; k < mateLength; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = task_materevc[i * task_decodedSequencePitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = task_materevcqual[i * task_qualityPitchInBytes + k];
                        }
                        d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];          
                    }else{
                        //copy until mate
                        const int length = d_outputAnchorLengths[i];
                        d_addAnchorLengths[offset] = length;
                        
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = d_outputAnchors[i * outputAnchorPitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = d_outputAnchorQualities[i * outputAnchorQualityPitchInBytes + k];
                        }
                        d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];
                        

                        //copy mate revc
                        const int mateLength = task_matelength[i];
                        d_addAnchorLengths[(offset+1)] = mateLength;

                        for(int k = threadIdx.x; k < mateLength; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[(offset+1) * outputAnchorPitchInBytes + k] = task_materevc[i * task_decodedSequencePitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[(offset+1) * outputAnchorQualityPitchInBytes + k] = task_materevcqual[i * task_qualityPitchInBytes + k];
                        }

                        //the gap between current anchor and mate begins at d_accumExtensionsLengths[i], and ends at d_accumExtensionsLengths[i] + length
                        //mate follows after the gap
                        d_addAnchorBeginsInExtendedRead[(offset+1)] = d_accumExtensionsLengths[i] + length;
                    }
                }
            }
        }
    };

    //replace positions which are covered by anchor and mate with the original data
    template<int blocksize, int groupsize>
    __global__
    void applyOriginalReadsToExtendedReads(
        std::size_t resultMSAColumnPitchInElements,
        int numFinishedTasks,
        char* __restrict__ d_decodedConsensus,
        char* __restrict__ d_consensusQuality,
        const int* __restrict__ d_resultLengths,
        const unsigned int* __restrict__ d_inputAnchorsEncoded,
        const int* __restrict__ d_inputAnchorLengths,
        const char* __restrict__ d_inputAnchorQualities,
        const bool* __restrict__ d_mateHasBeenFound,
        std::size_t  encodedSequencePitchInInts,
        std::size_t  qualityPitchInBytes
    ){
        const int numPairs = numFinishedTasks / 4;

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupIdInBlock = threadIdx.x / groupsize;

        for(int pair = blockIdx.x; pair < numPairs; pair += gridDim.x){
            const int resultLength = d_resultLengths[4 * pair + groupIdInBlock];
            const int anchorLength = d_inputAnchorLengths[4 * pair + groupIdInBlock];
            const unsigned int* const inputAnchor = &d_inputAnchorsEncoded[(4 * pair + groupIdInBlock) * encodedSequencePitchInInts];
            char* const resultSequence = &d_decodedConsensus[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];
            const char* const inputQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock) * qualityPitchInBytes];
            char* const resultQuality = &d_consensusQuality[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];

            SequenceHelpers::decodeSequence2Bit<int4>(group, inputAnchor, anchorLength, resultSequence);

            //copy anchor quality
            {
                const int numIters = anchorLength / sizeof(int);
                for(int i = group.thread_rank(); i < numIters; i += group.size()){
                    ((int*)resultQuality)[i] = ((const int*)inputQuality)[i];
                }
                const int remaining = anchorLength - sizeof(int) * numIters;
                if(remaining > 0){
                    for(int i = group.thread_rank(); i < remaining; i += group.size()){
                        resultQuality[sizeof(int) * numIters + i] = inputQuality[sizeof(int) * numIters + i];
                    }
                }
            }

            if(d_mateHasBeenFound[4 * pair + groupIdInBlock]){
                const int mateLength = d_inputAnchorLengths[4 * pair + groupIdInBlock + 1];
                const unsigned int* const anchorMate = &d_inputAnchorsEncoded[(4 * pair + groupIdInBlock + 1) * encodedSequencePitchInInts];
                const char* const anchorMateQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock + 1) * qualityPitchInBytes];
                SequenceHelpers::decodeSequence2Bit<char>(group, anchorMate, mateLength, resultSequence + resultLength - mateLength);

                //reverse copy mate quality scores
                for(int i = group.thread_rank(); i < mateLength; i += group.size()){
                    resultQuality[resultLength - mateLength + i] = anchorMateQuality[i];
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void flagFirstTasksOfConsecutivePairedTasks(
        int numTasks,
        bool* __restrict__ d_flags,
        const int* __restrict__ ids
    ){
        //d_flags must be zero'd

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numTasks - 1; i += stride){
            const bool areConsecutiveTasks = ids[i] + 1 == ids[i+1];
            const bool arePairedTasks = (ids[i] % 2) + 1 == (ids[i+1] % 2);

            if(areConsecutiveTasks && arePairedTasks){
                d_flags[i] = true;
            }
        }
    }

    template<int blocksize, int smemSizeBytes>
    __global__
    void flagPairedCandidatesKernel(
        const int* __restrict__ d_numChecks,
        const int* __restrict__ d_firstTasksOfPairsToCheck,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const read_number* __restrict__ d_candidateReadIds,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ d_numUsedReadIdsPerAnchorPrefixSum,
        const read_number* __restrict__ d_usedReadIds,
        bool* __restrict__ d_isPairedCandidate
    ){

        constexpr int numSharedElements = SDIV(smemSizeBytes, sizeof(int));

        __shared__ read_number sharedElements[numSharedElements];

        //search elements of array1 in array2. if found, set output element to true
        //array1 and array2 must be sorted
        auto process = [&](
            const read_number* array1,
            int numElements1,
            const read_number* array2,
            int numElements2,
            bool* output
        ){
            const int numIterations = SDIV(numElements2, numSharedElements);

            for(int iteration = 0; iteration < numIterations; iteration++){

                const int begin = iteration * numSharedElements;
                const int end = min((iteration+1) * numSharedElements, numElements2);
                const int num = end - begin;

                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    sharedElements[i] = array2[begin + i];
                }

                __syncthreads();

                //TODO in iteration > 0, we may skip elements at the beginning of first range

                for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                    if(!output[i]){
                        const read_number readId = array1[i];
                        const read_number readIdToFind = readId % 2 == 0 ? readId + 1 : readId - 1;

                        const bool found = thrust::binary_search(thrust::seq, sharedElements, sharedElements + num, readIdToFind);
                        if(found){
                            output[i] = true;
                        }
                    }
                }

                __syncthreads();
            }
        };

        const int numChecks = *d_numChecks;

        for(int a = blockIdx.x; a < numChecks; a += gridDim.x){
            const int firstTask = d_firstTasksOfPairsToCheck[a];
            //const int secondTask = firstTask + 1;

            //check for pairs in current candidates
            const int rangeBegin = d_numCandidatesPerAnchorPrefixSum[firstTask];                        
            const int rangeMid = d_numCandidatesPerAnchorPrefixSum[firstTask + 1];
            const int rangeEnd = rangeMid + d_numCandidatesPerAnchor[firstTask + 1];

            process(
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_isPairedCandidate + rangeBegin
            );

            process(
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_isPairedCandidate + rangeMid
            );

            //check for pairs in candidates of previous extension iterations

            const int usedRangeBegin = d_numUsedReadIdsPerAnchorPrefixSum[firstTask];                        
            const int usedRangeMid = d_numUsedReadIdsPerAnchorPrefixSum[firstTask + 1];
            const int usedRangeEnd = usedRangeMid + d_numUsedReadIdsPerAnchor[firstTask + 1];

            process(
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_usedReadIds + usedRangeMid,
                usedRangeEnd - usedRangeMid,
                d_isPairedCandidate + rangeBegin
            );

            process(
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_usedReadIds + usedRangeBegin,
                usedRangeMid - usedRangeBegin,
                d_isPairedCandidate + rangeMid
            );
        }
    }

    template<int blocksize>
    __global__
    void flagGoodAlignmentsKernel(
        const int* ids,
        const int* iterations,
        const AlignmentOrientation* __restrict__ d_alignment_best_alignment_flags,
        const int* __restrict__ d_alignment_shifts,
        const int* __restrict__ d_alignment_overlaps,
        const int* __restrict__ d_anchorSequencesLength,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const bool* __restrict__ d_isPairedCandidate,
        bool* __restrict__ d_keepflags,
        float min_overlap_ratio,
        int numAnchors,
        const int* __restrict__ currentNumCandidatesPtr,
        int initialNumCandidates
    ){
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;
        using BlockReduceInt = cub::BlockReduce<int, blocksize>;

        __shared__ union {
            typename BlockReduceFloat::TempStorage floatreduce;
            typename BlockReduceInt::TempStorage intreduce;
        } cubtemp;

        __shared__ int intbroadcast;
        __shared__ float floatbroadcast;

        for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
            const int num = d_numCandidatesPerAnchor[a];
            const int offset = d_numCandidatesPerAnchorPrefixSum[a];
            const float anchorLength = d_anchorSequencesLength[a];

            int threadReducedGoodAlignmentExists = 0;
            float threadReducedRelativeOverlapThreshold = 0.0f;

            for(int c = threadIdx.x; c < num; c += blockDim.x){
                d_keepflags[offset + c] = true;
            }

            //loop over candidates to compute relative overlap threshold

            for(int c = threadIdx.x; c < num; c += blockDim.x){
                const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                const int shift = d_alignment_shifts[offset + c];

                if(alignmentflag != AlignmentOrientation::None && shift >= 0){
                    if(!d_isPairedCandidate[offset+c]){
                        const float overlap = d_alignment_overlaps[offset + c];                            
                        const float relativeOverlap = overlap / anchorLength;
                        
                        if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                            threadReducedGoodAlignmentExists = 1;
                            const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                            threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                            // if(ids[a] == 1 && iterations[a] == 14){
                            //     printf("%d %f %f %f\n", c, relativeOverlap, tmp, threadReducedRelativeOverlapThreshold);
                            // }
                        }
                    }
                }else{
                    //remove alignment with negative shift or bad alignments
                    d_keepflags[offset + c] = false;
                }                       
            }

            int blockreducedGoodAlignmentExists = BlockReduceInt(cubtemp.intreduce)
                .Sum(threadReducedGoodAlignmentExists);
            if(threadIdx.x == 0){
                intbroadcast = blockreducedGoodAlignmentExists;
                //printf("task %d good: %d\n", a, blockreducedGoodAlignmentExists);
            }
            __syncthreads();

            blockreducedGoodAlignmentExists = intbroadcast;

            if(blockreducedGoodAlignmentExists > 0){
                float blockreducedRelativeOverlapThreshold = BlockReduceFloat(cubtemp.floatreduce)
                    .Reduce(threadReducedRelativeOverlapThreshold, cub::Max());
                if(threadIdx.x == 0){
                    floatbroadcast = blockreducedRelativeOverlapThreshold;
                    //printf("task %d thresh: %f\n", a, blockreducedRelativeOverlapThreshold);
                }
                __syncthreads();

                blockreducedRelativeOverlapThreshold = floatbroadcast;

                // if(ids[a] == 1 && iterations[a] == 14){
                //     printf("blockreducedRelativeOverlapThreshold %f\n", blockreducedRelativeOverlapThreshold);
                //     printf("ispaired %d, keep %d, overlap %f, relative overlap %f\n", 
                //         d_isPairedCandidate[offset+35], d_keepflags[offset + 35], float(d_alignment_overlaps[offset + 35]), float(d_alignment_overlaps[offset + 35]) / anchorLength);
                // }

                // loop over candidates and remove those with an alignment overlap threshold smaller than the computed threshold
                for(int c = threadIdx.x; c < num; c += blockDim.x){
                    if(!d_isPairedCandidate[offset+c]){
                        if(d_keepflags[offset + c]){
                            const float overlap = d_alignment_overlaps[offset + c];                            
                            const float relativeOverlap = overlap / anchorLength;                 

                            if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                d_keepflags[offset + c] = false;
                            }
                        }
                    }
                }
            }else{
                //NOOP.
                //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                // if(threadIdx.x == 0){
                //     printf("no good alignment,nc %d\n", num);
                // }
            }

            __syncthreads();

            // if(ids[a] == 1 && iterations[a] == 14){
            //     if(threadIdx.x == 0){
            //         printf("keepflags\n");
            //         for(int c = 0; c < num; c += 1){
            //             printf("%d, ", d_keepflags[offset + c]);
            //         }
            //         printf("\n");
            //     }
            // }
        }
    
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        for(int i = *currentNumCandidatesPtr + tid; i < initialNumCandidates; i += stride){
            d_keepflags[i] = false;
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void convertLocalIndicesInSegmentsToGlobalFlags(
        bool* __restrict__ d_flags,
        const int* __restrict__ indices,
        const int* __restrict__ segmentSizes,
        const int* __restrict__ segmentOffsets,
        int numSegments
    ){
        /*
            Input:
            indices: 0,1,2,0,0,0,0,3,5,0
            segmentSizes: 6,4,1
            segmentOffsets: 0,6,10,11

            Output:
            d_flags: 1,1,1,0,0,0,1,0,0,1,0,1

            d_flags must be initialized with 0
        */
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        
        const int numGroups = stride / groupsize;
        const int groupId = tid / groupsize;


        for(int s = groupId; s < numSegments; s += numGroups){        
            const int num = segmentSizes[s];
            const int offset = segmentOffsets[s];

            for(int i = group.thread_rank(); i < num; i += group.size()){
                const int globalIndex = indices[offset + i] + offset;
                d_flags[globalIndex] = true;
            }
        }
    }


    template<int blocksize>
    __global__
    void computeTaskSplitGatherIndicesDefaultKernel(
        int numTasks,
        int* __restrict__ d_positions4,
        int* __restrict__ d_positionsNot4,
        int* __restrict__ d_numPositions4_out,
        int* __restrict__ d_numPositionsNot4_out,
        const int* __restrict__ d_run_endoffsets,
        const int* __restrict__ d_num_runs,
        const int* __restrict__ d_sortedindices,
        const int* __restrict__ task_ids,
        const int* __restrict__ d_outputoffsetsPos4,
        const int* __restrict__ d_outputoffsetsNotPos4
    ){
        __shared__ int count4;
        __shared__ int countNot4;

        if(threadIdx.x == 0){
            count4 = 0;
            countNot4 = 0;
        }
        __syncthreads();

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        const int numRuns = *d_num_runs;

        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        const int numGroups = stride / 4;
        const int groupId = tid / 4;

        for(int t = groupId; t < numRuns; t += numGroups){
            const int runBegin = (t == 0 ? 0 : d_run_endoffsets[t-1]);
            const int runEnd = d_run_endoffsets[t];

            const int size = runEnd - runBegin;
            if(size < 4){
                if(group.thread_rank() == 0){
                    atomicAdd(&countNot4, size);
                }

                if(group.thread_rank() < size){
                    d_positionsNot4[d_outputoffsetsNotPos4[t] + group.thread_rank()]
                        = d_sortedindices[runBegin + group.thread_rank()];
                }
            }else{
                if(size != 4){
                    if(group.thread_rank() == 0){
                        printf("error size %d\n", size);
                    }
                    group.sync(); //DEBUG
                }
                assert(size == 4);

                if(group.thread_rank() == 0){
                    atomicAdd(&count4, 4);
                }

                //sort 4 elements of same pairId by id. id is either 0,1,2, or 3
                const int position = d_sortedindices[runBegin + group.thread_rank()];
                const int id = task_ids[position];
                assert(0 <= id && id < 4);

                for(int x = 0; x < 4; x++){
                    if(id == x){
                        //d_positions4[groupoutputbegin + x] = position;
                        d_positions4[d_outputoffsetsPos4[t] + x] = position;
                    }
                }
            }
        }
    
        __syncthreads();
        if(threadIdx.x == 0){
            atomicAdd(d_numPositions4_out + 0, count4);
            atomicAdd(d_numPositionsNot4_out + 0, countNot4);
        }
    }

    
    //requires external shared memory of size (sizeof(int) * numTasks * 2) bytes;
    template<int blocksize, int elementsPerThread>
    __global__
    void computeTaskSplitGatherIndicesSmallInputGetStaticSmemSizeKernel(
        std::size_t* output
    ){
        using BlockLoad = cub::BlockLoad<int, blocksize, elementsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockRadixSort = cub::BlockRadixSort<int, blocksize, elementsPerThread, int>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<int, blocksize>;
        using BlockStore = cub::BlockStore<int, blocksize, elementsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
        using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockExchange = cub::BlockExchange<int, blocksize, elementsPerThread>;

        using TempType = union{
            typename BlockLoad::TempStorage load;
            typename BlockRadixSort::TempStorage sort;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockStore::TempStorage store;
            typename BlockScan::TempStorage scan;
            typename BlockExchange::TempStorage exchange;
        };

        *output = sizeof(TempType);
    }

    template<int blocksize, int elementsPerThread>
    __global__
    void computeTaskSplitGatherIndicesSmallInputKernel(
        int numTasks,
        int* __restrict__ d_positions4,
        int* __restrict__ d_positionsNot4,
        int* __restrict__ d_numPositions4_out,
        int* __restrict__ d_numPositionsNot4_out,
        const int* __restrict__ task_pairIds,
        const int* __restrict__ task_ids,
        const int* __restrict__ d_minmax_pairId
    ){
        #ifndef NDEBUG
        constexpr int maxInputSize = blocksize * elementsPerThread;
        assert(numTasks <= maxInputSize);
        #endif

        assert(blockDim.x == blocksize);
        assert(gridDim.x == 1);

        using BlockLoad = cub::BlockLoad<int, blocksize, elementsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockRadixSort = cub::BlockRadixSort<int, blocksize, elementsPerThread, int>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<int, blocksize>;
        using BlockStore = cub::BlockStore<int, blocksize, elementsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
        using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockExchange = cub::BlockExchange<int, blocksize, elementsPerThread>;

        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockRadixSort::TempStorage sort;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockStore::TempStorage store;
            typename BlockScan::TempStorage scan;
            typename BlockExchange::TempStorage exchange;
        } temp;

        extern __shared__ int extsmemTaskSplit[]; // (sizeof(int) * numTasks * 2) bytes
        int* const sharedCounts = &extsmemTaskSplit[0];
        int* const sharedIndices = sharedCounts + numTasks;

        int numRuns = 0;
        int prefixsum[elementsPerThread];

        {

            int myPairIds[elementsPerThread];
            int myIndices[elementsPerThread];
            int headFlags[elementsPerThread];
            int maxScan[elementsPerThread];
        
            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                myIndices[i] = elementsPerThread * threadIdx.x + i;
            }

            BlockLoad(temp.load).Load(task_pairIds, myPairIds, numTasks, std::numeric_limits<int>::max());
            __syncthreads();

            BlockRadixSort(temp.sort).Sort(myPairIds, myIndices);
            __syncthreads();

            BlockStore(temp.store).Store(sharedIndices, myIndices, numTasks);
            __syncthreads();
        
            BlockDiscontinuity(temp.discontinuity).FlagHeads(headFlags, myPairIds, cub::Inequality());
            __syncthreads();                    
            
            BlockScan(temp.scan).ExclusiveSum(headFlags, prefixsum, numRuns);
            __syncthreads();
            
            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                if(headFlags[i] > 0){
                    maxScan[i] = prefixsum[i];
                }else{
                    maxScan[i] = 0;
                }
            }                

            BlockScan(temp.scan).InclusiveScan(maxScan, maxScan, cub::Max{});
            __syncthreads();

            //compute counts of unique pair ids
            for(int i = threadIdx.x; i < numRuns; i += blockDim.x){
                sharedCounts[i] = 0;
            }

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                if(threadIdx.x * elementsPerThread + i < numTasks){
                    atomicAdd(sharedCounts + maxScan[i], 1);
                }
            }

            __syncthreads();
        }

        //compute output offsets and perform split
        
        int myCounts[elementsPerThread];
        int outputoffsets4[elementsPerThread];
        int outputoffsetsNot4[elementsPerThread];

        BlockLoad(temp.load).Load(sharedCounts, myCounts, numRuns, 0);
        __syncthreads();

        int myModifiedCounts[elementsPerThread];
        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            myModifiedCounts[i] = (myCounts[i] == 4) ? 4 : 0;
        }

        int numPos4 = 0;
        BlockScan(temp.scan).ExclusiveSum(myModifiedCounts, outputoffsets4, numPos4);
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            myModifiedCounts[i] = (myCounts[i] < 4) ? myCounts[i] : 0;
        }

        int numPosNot4 = 0;
        BlockScan(temp.scan).ExclusiveSum(myModifiedCounts, outputoffsetsNot4, numPosNot4);
        __syncthreads();

        BlockScan(temp.scan).ExclusiveSum(myCounts, prefixsum, numRuns);
        __syncthreads();

        BlockExchange(temp.exchange).BlockedToStriped(myCounts);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(outputoffsets4);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(outputoffsetsNot4);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(prefixsum);
        __syncthreads();

        //compact indices
        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            if(i * blocksize + threadIdx.x < numRuns){
                const int runBegin = prefixsum[i];
                const int runEnd = runBegin + myCounts[i];
                const int size = runEnd - runBegin;

                if(size < 4){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        if(k < size){
                            const int outputoffset = outputoffsetsNot4[i] + k;
                            d_positionsNot4[outputoffset] = sharedIndices[runBegin + k];
                        }
                    }
                }else{
                    int positions[4];
                    int ids[4];

                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        positions[k] = sharedIndices[runBegin + k];
                    }

                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        ids[k] = task_ids[positions[k]];
                        assert(0 <= ids[k] && ids[k] < 4);
                    }

                    // printf("thread %d. positions %d %d %d %d, runBegin %d\n", threadIdx.x, 
                    //     positions[0], positions[1], positions[2], positions[3],runBegin
                    // );

                    //sort 4 elements of same pairId by id and store them. id is either 0,1,2, or 3
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        #pragma unroll
                        for(int l = 0; l < 4; l++){
                            if(ids[l] == k){
                                const int outputoffset = outputoffsets4[i] + k;
                                d_positions4[outputoffset] = positions[l];
                            }
                        }
                    }
                }

            }
        }

        __syncthreads();
        if(threadIdx.x == 0){
            atomicAdd(d_numPositions4_out, numPos4);
            atomicAdd(d_numPositionsNot4_out, numPosNot4);
        }

    }





    template<int blocksize>
    __global__
    void computeExtensionStepFromMsaKernel_new(
        int minFragmentSize,
        int maxFragmentSize,
        const gpu::GPUMultiMSA multiMSA,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ d_anchorSequencesLength,
        const int* __restrict__ d_inputMateLengths,
        AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_isPairedTask,
        const unsigned int* __restrict__ d_inputanchormatedata,
        const char* __restrict__ mateQualities,
        int encodedSequencePitchInInts,
        int qualityPitchInBytes,
        bool* __restrict__ d_outputMateHasBeenFound,
        int minCoverageForExtension,
        int fixedStepsize,
        char* __restrict__ extendedSequences,
        int* __restrict__ extendedSequenceLengths,
        char* __restrict__ qualitiesOfExtendedSequences,
        int extendedSequencePitchInBytes//,
        //int debugindex = -1
    ){

        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
        } temp;

        constexpr int smemEncodedMateInts = 32;
        __shared__ unsigned int smemEncodedMate[smemEncodedMateInts];

        __shared__ int broadcastsmem_int;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            const int numCandidates = d_numCandidatesPerAnchor[t];

            // if(threadIdx.x == 0){
            //     if(debugindex != -1 && debugindex == t) printf("numCandidates %d\n", numCandidates);
            // }

            if(numCandidates > 0){
                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int anchorLength = d_anchorSequencesLength[t];
                const int mateLength = d_inputMateLengths[t];
                const bool isPaired = d_isPairedTask[t];
                const char* myMateQualities = mateQualities + qualityPitchInBytes * t;
                const int consensusLength = msa.computeSize();

                auto consensusDecoded = msa.getDecodedConsensusIterator();
                auto consensusQuality = msa.getConsensusQualityIterator();


                AbortReason* const abortReasonPtr = d_abortReasons + t;
                bool* const mateHasBeenFoundPtr = d_outputMateHasBeenFound + t;

                char* const outputExtendedSequence = extendedSequences + t * extendedSequencePitchInBytes;
                char* const outputExtendedQuality = qualitiesOfExtendedSequences + t * extendedSequencePitchInBytes;

                const int currentExtensionLength = extendedSequenceLengths[t];
                const int accumExtensionsLength = currentExtensionLength - anchorLength;



                int extendBy = std::min(
                    consensusLength - anchorLength, 
                    std::max(0, fixedStepsize)
                );

                //cannot extend over fragment
                const int maxExtendBy_forFragmentSize = maxFragmentSize - currentExtensionLength;
                const int maxExtendBy_toReachMate = (maxFragmentSize - mateLength) - accumExtensionsLength;
                extendBy = std::min(extendBy, std::min(maxExtendBy_forFragmentSize, maxExtendBy_toReachMate));



                //auto firstLowCoverageIter = std::find_if(coverage + anchorLength, coverage + consensusLength, [&](int cov){ return cov < minCoverageForExtension; });
                //coverage is monotonically decreasing. convert coverages to 1 if >= minCoverageForExtension, else 0. Find position of first 0
                int myPos = consensusLength;
                for(int i = anchorLength + threadIdx.x; i < consensusLength; i += blockDim.x){
                    int flag = msa.coverages[i] < minCoverageForExtension ? 0 : 1;
                    if(flag == 0 && i < myPos){
                        myPos = i;
                    }
                }

                myPos = BlockReduce(temp.reduce).Reduce(myPos, cub::Min{});

                if(threadIdx.x == 0){
                    broadcastsmem_int = myPos;
                }
                __syncthreads();
                myPos = broadcastsmem_int;
                __syncthreads();

                if(fixedStepsize <= 0){
                    extendBy = myPos - anchorLength;
                    extendBy = std::min(extendBy, std::min(maxExtendBy_forFragmentSize, maxExtendBy_toReachMate));
                }

                //if(threadIdx.x == 0) printf("t %d, extendBy %d, currentExtensionLength %d\n", t, extendBy, currentExtensionLength);

                auto makeAnchorForNextIteration = [&](){
                    if(extendBy == 0){
                        if(threadIdx.x == 0){
                            *abortReasonPtr = AbortReason::MsaNotExtended;
                            //if(debugindex != -1 && debugindex == t) printf("makeAnchorForNextIteration abort\n");
                        }
                    }else{
          

                        {
                            // for(int i = threadIdx.x; i < extendBy; i += blockDim.x){
                            //     outputExtendedSequence[currentExtensionLength + i] = consensusDecoded[anchorLength + i];
                            // }
                            for(int i = threadIdx.x; i < anchorLength; i += blockDim.x){
                                outputExtendedSequence[currentExtensionLength - anchorLength + extendBy + i] = consensusDecoded[extendBy + i];
                                outputExtendedQuality[currentExtensionLength - anchorLength + extendBy + i] = consensusQuality[extendBy + i];
                            }
                            if(threadIdx.x == 0){
                                extendedSequenceLengths[t] = currentExtensionLength + extendBy;
                                //printf("extendedSequenceLengths[t] %d\n", extendedSequenceLengths[t]);
                               // if(debugindex != -1 && debugindex == t) printf("makeAnchorForNextIteration extendBy %d\n", extendBy);
                            }
                            
                        }
                    }
                };

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr float maxRelativeMismatchesInOverlap = 0.06f;
                constexpr int maxAbsoluteMismatchesInOverlap = 10;

                const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);

                
                //could we have reached the mate ? 
                if(isPaired && accumExtensionsLength + consensusLength - requiredOverlapMate + mateLength >= minFragmentSize){
                    //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [minFragmentSize, maxFragmentSize]

                    const int firstStartpos = std::max(0, minFragmentSize - accumExtensionsLength - mateLength);
                    const int lastStartposExcl = std::min(
                        std::max(0, maxFragmentSize - accumExtensionsLength - mateLength) + 1,
                        consensusLength - requiredOverlapMate
                    );

                    int bestOverlapMismatches = std::numeric_limits<int>::max();
                    int bestOverlapStartpos = -1;

                    const unsigned int* encodedMate = nullptr;
                    {
                        const unsigned int* const gmemEncodedMate = d_inputanchormatedata + t * encodedSequencePitchInInts;
                        const int requirednumints = SequenceHelpers::getEncodedNumInts2Bit(mateLength);
                        if(smemEncodedMateInts >= requirednumints){
                            for(int i = threadIdx.x; i < requirednumints; i += blockDim.x){
                                smemEncodedMate[i] = gmemEncodedMate[i];
                            }
                            encodedMate = &smemEncodedMate[0];
                            __syncthreads();
                        }else{
                            encodedMate = &gmemEncodedMate[0];
                        }
                    }

                    for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                        //compute metrics of overlap

                        //Hamming distance. positions which do not overlap are not accounted for
                        int ham = 0;
                        for(int i = threadIdx.x; i < min(consensusLength - startpos, mateLength); i += blockDim.x){
                            std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                            std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                            char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);

                            //TODO store consensusDecoded in smem ?
                            ham += (consensusDecoded[startpos + i] != decbasematecomp) ? 1 : 0;
                        }

                        ham = BlockReduce(temp.reduce).Sum(ham);

                        if(threadIdx.x == 0){
                            broadcastsmem_int = ham;
                        }
                        __syncthreads();
                        ham = broadcastsmem_int;
                        __syncthreads();

                        if(bestOverlapMismatches > ham){
                            bestOverlapMismatches = ham;
                            bestOverlapStartpos = startpos;
                        }

                        if(bestOverlapMismatches == 0){
                            break;
                        }
                    }

                    // if(threadIdx.x == 0){
                    //     printf("gpu: bestOverlapMismatches %d,bestOverlapStartpos %d\n", bestOverlapMismatches, bestOverlapStartpos);
                    // }

                    if(bestOverlapMismatches <= maxNumMismatches){
                        const int mateStartposInConsensus = bestOverlapStartpos;
                        const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - anchorLength);
                        if(threadIdx.x == 0){
                            //printf("missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                        }

                        if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                            //bridge the gap between current anchor and mate

                            if(threadIdx.x == 0){
                                *mateHasBeenFoundPtr = true;
                                //printf("d_accumExtensionsLengthsOUT[t] %d, *outputAnchorLengthPtr %d\n", d_accumExtensionsLengthsOUT[t], *outputAnchorLengthPtr);
                            }


                            for(int i = threadIdx.x; i < missingPositionsBetweenAnchorEndAndMateBegin; i += blockDim.x){
                                outputExtendedSequence[currentExtensionLength + i] = consensusDecoded[anchorLength + i];
                                outputExtendedQuality[currentExtensionLength + i] = consensusQuality[anchorLength + i];
                            }
                            for(int i = threadIdx.x; i < mateLength; i += blockDim.x){
                                std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                                std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                                char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);
                                outputExtendedSequence[currentExtensionLength + missingPositionsBetweenAnchorEndAndMateBegin + i] = decbasematecomp;
                                outputExtendedQuality[currentExtensionLength + missingPositionsBetweenAnchorEndAndMateBegin + i] = myMateQualities[mateLength - 1 - i];
                            }
                            if(threadIdx.x == 0){
                                extendedSequenceLengths[t] = currentExtensionLength + missingPositionsBetweenAnchorEndAndMateBegin + mateLength;
                                //printf("extendedSequenceLengths[t] %d\n", extendedSequenceLengths[t]);

                                //if(debugindex != -1 && debugindex == t) printf("finished missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                            }
                        }else{

                            if(threadIdx.x == 0){
                                *mateHasBeenFoundPtr = true;
                                //printf("d_accumExtensionsLengthsOUT[t] %d, *outputAnchorLengthPtr %d, mateStartposInConsensus %d\n", d_accumExtensionsLengthsOUT[t], *outputAnchorLengthPtr, mateStartposInConsensus);
                            
                            }

                            for(int i = threadIdx.x; i < mateLength; i += blockDim.x){
                                std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                                std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                                char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);
                                outputExtendedSequence[currentExtensionLength - anchorLength + mateStartposInConsensus + i] = decbasematecomp;
                                outputExtendedQuality[currentExtensionLength - anchorLength + mateStartposInConsensus + i] = myMateQualities[mateLength - 1 - i];
                            }
                            if(threadIdx.x == 0){
                                extendedSequenceLengths[t] = currentExtensionLength - anchorLength + mateStartposInConsensus + mateLength;
                                //printf("extendedSequenceLengths[t] %d\n", extendedSequenceLengths[t]);

                                //if(debugindex != -1 && debugindex == t) printf("finished missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                            }
                        }

                        
                    }else{
                        makeAnchorForNextIteration();
                    }
                }else{
                    makeAnchorForNextIteration();
                }

            }else{ //numCandidates == 0
                if(threadIdx.x == 0){
                    d_outputMateHasBeenFound[t] = false;
                    d_abortReasons[t] = AbortReason::NoPairedCandidatesAfterAlignment;
                }
            }
        }
    }







    template<int blocksize>
    __global__
    void computeExtensionStepQualityKernel(
        //const int* d_iterations,
        float* d_goodscores,
        const gpu::GPUMultiMSA multiMSA,
        const AbortReason* d_abortReasons,
        const bool* d_mateHasBeenFound,
        const int* accumExtensionLengthsBefore,
        const int* accumExtensionLengthsAfter,
        const int* anchorLengths,
        const int* d_numCandidatesPerAnchor,
        const int* d_numCandidatesPerAnchorPrefixSum,
        const int* d_candidateSequencesLengths,
        const int* d_alignment_shifts,
        const AlignmentOrientation* d_alignment_best_alignment_flags,
        const unsigned int* d_candidateSequencesData,
        const gpu::MSAColumnProperties* d_msa_column_properties,
        int encodedSequencePitchInInts
    ){
        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;
        using AmbiguousColumnsChecker = CheckAmbiguousColumnsGpu<blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
            typename AmbiguousColumnsChecker::TempStorage columnschecker;
        } temp;

        __shared__ typename AmbiguousColumnsChecker::SplitInfos smemSplitInfos;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            if(d_abortReasons[t] == AbortReason::None && !d_mateHasBeenFound[t]){

                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int extendedBy = accumExtensionLengthsAfter[t] - accumExtensionLengthsBefore[t];
                const int anchorLength = anchorLengths[t];

                //const float* const mySupport = msa.support;
                const int* const myCounts = msa.counts;
                const int* const myCoverage = msa.coverages;

                const int* const myCandidateLengths = d_candidateSequencesLengths + d_numCandidatesPerAnchorPrefixSum[t];
                const int* const myCandidateShifts = d_alignment_shifts + d_numCandidatesPerAnchorPrefixSum[t];
                const AlignmentOrientation* const myCandidateBestAlignmentFlags = d_alignment_best_alignment_flags + d_numCandidatesPerAnchorPrefixSum[t];
                const unsigned int* const myCandidateSequencesData = d_candidateSequencesData + encodedSequencePitchInInts * d_numCandidatesPerAnchorPrefixSum[t];

                // float supportSum = 0.0f;

                // for(int i = threadIdx.x; i < extendedBy; i += blockDim.x){
                //     supportSum += 1.0f - mySupport[anchorLength + i];
                // }

                // float reducedSupport = BlockReduceFloat(temp.reduceFloat).Sum(supportSum);

                // //printf("reducedSupport %f\n", reducedSupport);
                // if(threadIdx.x == 0){
                //     d_goodscores[t] = reducedSupport;
                // }

                AmbiguousColumnsChecker checker(
                    myCounts + 0 * msa.columnPitchInElements,
                    myCounts + 1 * msa.columnPitchInElements,
                    myCounts + 2 * msa.columnPitchInElements,
                    myCounts + 3 * msa.columnPitchInElements,
                    myCoverage
                );

                //int count = checker.getAmbiguousColumnCount(anchorLength, anchorLength + extendedBy, temp.reduce);

                //auto a = clock();

                checker.getSplitInfos(anchorLength, anchorLength + extendedBy, 0.4f, 0.6f, smemSplitInfos);

                //auto b = clock();

                int count = checker.getNumberOfSplits(
                    smemSplitInfos, 
                    d_msa_column_properties[t],
                    d_numCandidatesPerAnchor[t], 
                    myCandidateShifts,
                    myCandidateBestAlignmentFlags,
                    myCandidateLengths,
                    myCandidateSequencesData, 
                    encodedSequencePitchInInts, 
                    temp.columnschecker
                );

                //auto c = clock();

                if(threadIdx.x == 0){
                    //printf("t %d, iteration %d, numSplits %d\n", t, d_iterations[t], count);
                    d_goodscores[t] = count;
                    //printf("cand %d extendedBy %d, %lu %lu, infos %d, count %d\n", d_numCandidatesPerAnchor[t], extendedBy, b-a, c-b, smemSplitInfos.numSplitInfos, count);
                }
                
                __syncthreads();
            }
        }
    }

    template<int blocksize>
    __global__
    void makePairResultsFromFinishedTasksDryRunKernel(
        int numResults,
        int* __restrict__ outputLengthUpperBounds,
        const int* __restrict__ originalReadLengths,
        const int* __restrict__ dataExtendedReadLengths,
        const char* __restrict__ dataExtendedReadSequences,
        const char* __restrict__ dataExtendedReadQualities,
        const bool* __restrict__ dataMateHasBeenFound,
        const float* __restrict__ dataGoodScores,
        int inputPitch,
        int minFragmentSize,
        int maxFragmentSize
    ){
        auto group = cg::this_thread_block();
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / group.size();
        const int groupIdInGrid = (threadIdx.x + blockDim.x * blockIdx.x) / group.size();

        auto computeRead2Begin = [&](int i){
            if(dataMateHasBeenFound[i]){
                return dataExtendedReadLengths[i] - originalReadLengths[i+1];
            }else{
                return -1;
            }
        };

        auto mergelength = [&](int l, int r){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthR = dataExtendedReadLengths[r];

            const int read2begin = computeRead2Begin(l);
   
            auto overlapstart = read2begin;

            const int resultsize = overlapstart + lengthR;
            
            return resultsize;
        };
    
        //process pair at position pairIdsToProcess[posInList] and store to result position posInList
        for(int posInList = groupIdInGrid; posInList < numResults; posInList += numGroupsInGrid){
            group.sync(); //reuse smem

            const int p = posInList;

            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            int* const myResultLengths = outputLengthUpperBounds + posInList;            

            auto LRmatefoundfunc = [&](){
                const int extendedReadLength3 = dataExtendedReadLengths[i3];
                const int originalLength3 = originalReadLengths[i3];

                int resultsize = mergelength(i0, i1);
                if(extendedReadLength3 > originalLength3){
                    resultsize += extendedReadLength3 - originalLength3;
                }                

                if(group.thread_rank() == 0){
                    *myResultLengths = resultsize;
                }
            };

            auto RLmatefoundfunc = [&](){
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int originalLength1 = originalReadLengths[i1];

                const int mergedLength = mergelength(i2, i3);
                int resultsize = mergedLength;
                if(extendedReadLength1 > originalLength1){
                    resultsize += extendedReadLength1 - originalLength1;
                }
                
                if(group.thread_rank() == 0){
                    *myResultLengths = resultsize;
                }
            };

            if(dataMateHasBeenFound[i0] && dataMateHasBeenFound[i2]){
                if(dataGoodScores[i0] < dataGoodScores[i2]){
                    LRmatefoundfunc();
                }else{
                    RLmatefoundfunc();
                }                
            }else 
            if(dataMateHasBeenFound[i0]){
                LRmatefoundfunc();                
            }else if(dataMateHasBeenFound[i2]){
                RLmatefoundfunc();                
            }else{
                constexpr int minimumOverlap = 40;

                int currentsize = 0;

                const int extendedReadLength0 = dataExtendedReadLengths[i0];
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int extendedReadLength2 = dataExtendedReadLengths[i2];
                const int extendedReadLength3 = dataExtendedReadLengths[i3];

                const int originalLength1 = originalReadLengths[i1];
                const int originalLength3 = originalReadLengths[i3];

                //insert extensions of reverse complement of d3 at beginning
                if(extendedReadLength3 > originalLength3){
                    currentsize = (extendedReadLength3 - originalLength3);
                }

                //try to find overlap of d0 and revc(d2)
                bool didMergeDifferentStrands = false;

                if(extendedReadLength0 + extendedReadLength2 >= minFragmentSize + minimumOverlap){
                    const int maxNumberOfPossibilities = (maxFragmentSize - minFragmentSize) + 1;
                    const int resultLengthUpperBound = minFragmentSize + maxNumberOfPossibilities;
                    currentsize += resultLengthUpperBound;
                    didMergeDifferentStrands = true;
                }

                if(didMergeDifferentStrands){

                }else{
                    currentsize += extendedReadLength0;
                }

                if(didMergeDifferentStrands && extendedReadLength1 > originalLength1){

                    //insert extensions of d1 at end
                    currentsize += (extendedReadLength1 - originalLength1);
                }

                if(group.thread_rank() == 0){
                    *myResultLengths = currentsize;
                }
            }

        }
    }


    //requires external shared memory of 3*outputPitch per group in block
    template<int blocksize>
    __global__
    void makePairResultsFromFinishedTasksKernel(
        int numResults,
        bool* __restrict__ outputAnchorIsLR,
        char* __restrict__ outputSequences,
        char* __restrict__ outputQualities,
        int* __restrict__ outputLengths,
        int* __restrict__ outRead1Begins,
        int* __restrict__ outRead2Begins,
        bool* __restrict__ outMateHasBeenFound,
        bool* __restrict__ outMergedDifferentStrands,
        int outputPitch,
        const int* __restrict__ originalReadLengths,
        const int* __restrict__ dataExtendedReadLengths,
        const char* __restrict__ dataExtendedReadSequences,
        const char* __restrict__ dataExtendedReadQualities,
        const bool* __restrict__ dataMateHasBeenFound,
        const float* __restrict__ dataGoodScores,
        int inputPitch,
        int minFragmentSize,
        int maxFragmentSize
    ){
        auto group = cg::this_thread_block();
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / group.size();
        const int groupIdInGrid = (threadIdx.x + blockDim.x * blockIdx.x) / group.size();

        extern __shared__ int smemForResults[];
        char* const smemChars = (char*)&smemForResults[0];
        char* const smemSequence = smemChars;
        char* const smemSequence2 = smemSequence + outputPitch;
        char* const smemQualities = smemSequence2 + outputPitch;

        __shared__ typename gpu::MismatchRatioGlueDecider<blocksize>::TempStorage smemDecider;

        auto checkIndex = [&](int index){
            // assert(0 <= index);
            // assert(index < outputPitch);
        };

        auto computeRead2Begin = [&](int i){
            if(dataMateHasBeenFound[i]){
                //extendedRead.length() - task.decodedMateRevC.length();
                return dataExtendedReadLengths[i] - originalReadLengths[i+1];
            }else{
                return -1;
            }
        };

        auto mergelength = [&](int l, int r){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthR = dataExtendedReadLengths[r];

            const int read2begin = computeRead2Begin(l);
   
            auto overlapstart = read2begin;

            const int resultsize = overlapstart + lengthR;
            
            return resultsize;
        };

        //merge extensions of the same pair and same strand and append to result
        auto merge = [&](auto& group, int l, int r, char* sequenceOutput, char* qualityOutput){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthL = dataExtendedReadLengths[l];
            const int lengthR = dataExtendedReadLengths[r];
            const int originalLengthR = originalReadLengths[r];
   
            auto overlapstart = computeRead2Begin(l);

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                checkIndex(i);
                sequenceOutput[i] 
                    = dataExtendedReadSequences[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                checkIndex(lengthL + i);
                checkIndex(originalLengthR + i);
                sequenceOutput[lengthL + i] 
                    = dataExtendedReadSequences[r * inputPitch + originalLengthR + i];
            }

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                checkIndex(i);
                qualityOutput[i] 
                    = dataExtendedReadQualities[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                checkIndex(lengthL + i);
                checkIndex(originalLengthR + i);
                qualityOutput[lengthL + i] 
                    = dataExtendedReadQualities[r * inputPitch + originalLengthR + i];
            }
        };
    
        //process pair at position pairIdsToProcess[posInList] and store to result position posInList
        for(int posInList = groupIdInGrid; posInList < numResults; posInList += numGroupsInGrid){
            group.sync(); //reuse smem

            const int p = posInList;

            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            char* const myResultSequence = outputSequences + posInList * outputPitch;
            char* const myResultQualities = outputQualities + posInList * outputPitch;
            int* const myResultRead1Begins = outRead1Begins + posInList;
            int* const myResultRead2Begins = outRead2Begins + posInList;
            int* const myResultLengths = outputLengths + posInList;
            bool* const myResultAnchorIsLR = outputAnchorIsLR + posInList;
            bool* const myResultMateHasBeenFound = outMateHasBeenFound + posInList;
            bool* const myResultMergedDifferentStrands = outMergedDifferentStrands + posInList;
            

            auto LRmatefoundfunc = [&](){
                const int extendedReadLength3 = dataExtendedReadLengths[i3];
                const int originalLength3 = originalReadLengths[i3];

                int resultsize = mergelength(i0, i1);
                if(extendedReadLength3 > originalLength3){
                    resultsize += extendedReadLength3 - originalLength3;
                }                

                int currentsize = 0;

                if(extendedReadLength3 > originalLength3){
                    //insert extensions of reverse complement of d3 at beginning

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k]
                        );
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultQualities[k] = 
                            dataExtendedReadQualities[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k];
                    }

                    currentsize = (extendedReadLength3 - originalLength3);
                }

                merge(group, i0, i1, myResultSequence + currentsize, myResultQualities + currentsize);

                int read1begin = 0;
                int read2begin = computeRead2Begin(i0);

                if(extendedReadLength3 > originalLength3){                    
                    read1begin += (extendedReadLength3 - originalLength3);
                    read2begin += (extendedReadLength3 - originalLength3);
                }

                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            auto RLmatefoundfunc = [&](){
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int originalLength1 = originalReadLengths[i1];

                const int mergedLength = mergelength(i2, i3);
                int resultsize = mergedLength;
                if(extendedReadLength1 > originalLength1){
                    resultsize += extendedReadLength1 - originalLength1;
                }

                merge(group, i2, i3, smemSequence, smemQualities);

                group.sync();

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    checkIndex(k);
                    checkIndex(mergedLength - 1 - k);
                    myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                        smemSequence[mergedLength - 1 - k]
                    );
                }

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    checkIndex(k);
                    checkIndex(mergedLength - 1 - k);
                    myResultQualities[k] = smemQualities[mergedLength - 1 - k];
                }

                group.sync();

                const int sizeOfRightExtension = mergedLength - (computeRead2Begin(i2) + originalReadLengths[i3]);

                int read1begin = 0;
                int newread2begin = mergedLength - (read1begin + originalReadLengths[i2]);
                int newread1begin = sizeOfRightExtension;

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                #ifndef NDEBUG
                int newread2length = originalReadLengths[i2];
                int newread1length = originalReadLengths[i3];
                assert(newread1begin + newread1length <= mergedLength);
                assert(newread2begin + newread2length <= mergedLength);
                #endif

                if(extendedReadLength1 > originalLength1){
                    //insert extensions of d1 at end
                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(mergedLength + k);
                        checkIndex(originalLength1 + k);
                        myResultSequence[mergedLength + k] = 
                            dataExtendedReadSequences[i1 * inputPitch + originalLength1 + k];                        
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(mergedLength + k);
                        checkIndex(originalLength1 + k);
                        myResultQualities[mergedLength + k] = 
                            dataExtendedReadQualities[i1 * inputPitch + originalLength1 + k];                        
                    }
                }

                read1begin = newread1begin;
                const int read2begin = newread2begin;
                
                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = false;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            // if(group.thread_rank() == 0){
            //     printf("posInList %d, %d %d, %f %f\n", posInList, dataMateHasBeenFound[i0], dataMateHasBeenFound[i2], dataGoodScores[i0], dataGoodScores[i2]);
            // }

            if(dataMateHasBeenFound[i0] && dataMateHasBeenFound[i2]){
                if(dataGoodScores[i0] < dataGoodScores[i2]){
                    LRmatefoundfunc();
                }else{
                    RLmatefoundfunc();
                }                
            }else if(dataMateHasBeenFound[i0]){
                LRmatefoundfunc();                
            }else if(dataMateHasBeenFound[i2]){
                RLmatefoundfunc();                
            }else{
                
                constexpr int minimumOverlap = 40;
                constexpr float maxRelativeErrorInOverlap = 0.05;

                int read1begin = 0;
                int read2begin = computeRead2Begin(i0);
                int currentsize = 0;

                const int extendedReadLength0 = dataExtendedReadLengths[i0];
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int extendedReadLength2 = dataExtendedReadLengths[i2];
                const int extendedReadLength3 = dataExtendedReadLengths[i3];

                const int originalLength0 = originalReadLengths[i0];
                const int originalLength1 = originalReadLengths[i1];
                const int originalLength2 = originalReadLengths[i2];
                const int originalLength3 = originalReadLengths[i3];

                //insert extensions of reverse complement of d3 at beginning
                if(extendedReadLength3 > originalLength3){

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k]
                        );
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultQualities[k] = 
                            dataExtendedReadQualities[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k];
                    }

                    currentsize = (extendedReadLength3 - originalLength3);
                    read1begin = (extendedReadLength3 - originalLength3);
                }

                //try to find overlap of d0 and revc(d2)                

                bool didMergeDifferentStrands = false;

                //if the longest achievable pseudo read reaches the minimum required pseudo read length
                if(extendedReadLength0 + extendedReadLength2 - minimumOverlap >= minFragmentSize){
                    //copy sequences to smem

                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        checkIndex(k);
                        smemSequence[k] = dataExtendedReadSequences[i0 * inputPitch + k];
                    }

                    for(int k = group.thread_rank(); k < extendedReadLength2; k += group.size()){
                        checkIndex(k);
                        checkIndex(extendedReadLength2 - 1 - k);
                        smemSequence2[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i2 * inputPitch + extendedReadLength2 - 1 - k]
                        );
                    }
                    for(int k = group.thread_rank(); k < extendedReadLength2; k += group.size()){
                        checkIndex(k);
                        checkIndex(extendedReadLength2 - 1 - k);
                        smemQualities[k] = dataExtendedReadQualities[i2 * inputPitch + extendedReadLength2 - 1 - k];
                    }

                    group.sync();

                    const int minimumResultLength = std::max(originalLength0+1, minFragmentSize);
                    const int maximumResultLength = std::min(extendedReadLength0 + extendedReadLength2 - minimumOverlap, maxFragmentSize);

                    gpu::MismatchRatioGlueDecider<blocksize> decider(smemDecider, minimumOverlap, maxRelativeErrorInOverlap);
                    gpu::QualityWeightedGapGluer gluer(originalLength0, originalLength2);

                    for(int resultLength = minimumResultLength; resultLength <= maximumResultLength; resultLength++){                       

                        auto decision = decider(
                            MyStringView(smemSequence, extendedReadLength0), 
                            MyStringView(smemSequence2, extendedReadLength2), 
                            resultLength,
                            MyStringView(&dataExtendedReadQualities[i0 * inputPitch], extendedReadLength0),
                            MyStringView(&smemQualities[0], extendedReadLength2)
                        );

                        if(decision.valid){
                            // if(threadIdx.x == 0){
                            //     printf("decision s1f %d, s2f %d, rl %d\n",
                            //         decision.s1FirstResultIndex,decision.s2FirstResultIndex,decision.resultlength);
                            //     for(int x = 0; x < decision.s1.size(); x++){
                            //         printf("%c", decision.s1[x]);
                            //     }
                            //     printf("\n");
                            //     for(int x = 0; x < decision.q1.size(); x++){
                            //         printf("%c", decision.q1[x]);
                            //     }
                            //     printf("\n");
                            //     for(int x = 0; x < decision.s2.size(); x++){
                            //         printf("%c", decision.s2[x]);
                            //     }
                            //     printf("\n");
                            //     for(int x = 0; x < decision.q2.size(); x++){
                            //         printf("%c", decision.q2[x]);
                            //     }
                            //     printf("\n");
                            // }
                            gluer(group, decision, myResultSequence + currentsize, myResultQualities + currentsize);
                            currentsize += resultLength;
                            
                            didMergeDifferentStrands = true;
                            break;
                        }
                    }
                }

                if(didMergeDifferentStrands){
                    read2begin = currentsize - originalLength2;
                }else{
                    //initialize result with d0
                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        checkIndex(currentsize + k);
                        checkIndex(k);
                        myResultSequence[currentsize + k] = dataExtendedReadSequences[i0 * inputPitch + k];
                    }

                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        checkIndex(currentsize + k);
                        checkIndex(k);
                        myResultQualities[currentsize + k] = dataExtendedReadQualities[i0 * inputPitch + k];
                    }

                    currentsize += extendedReadLength0;
                }

                if(didMergeDifferentStrands && extendedReadLength1 > originalLength1){

                    //insert extensions of d1 at end

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(currentsize + k);
                        checkIndex(originalLength1 + k);
                        myResultSequence[currentsize + k] = dataExtendedReadSequences[i1 * inputPitch + originalLength1 + k];
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(currentsize + k);
                        checkIndex(originalLength1 + k);
                        myResultQualities[currentsize + k] = dataExtendedReadQualities[i1 * inputPitch + originalLength1 + k];
                    }

                    currentsize += (extendedReadLength1 - originalLength1);
                }

                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = currentsize;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = didMergeDifferentStrands;
                    *myResultMergedDifferentStrands = didMergeDifferentStrands;
                }
            }

        }
    }

    template<int blocksize>
    __global__
    void makePairResultsFromFinishedTasksKernel_strict(
        int numResults,
        bool* __restrict__ outputAnchorIsLR,
        char* __restrict__ outputSequences,
        char* __restrict__ outputQualities,
        int* __restrict__ outputLengths,
        int* __restrict__ outRead1Begins,
        int* __restrict__ outRead2Begins,
        bool* __restrict__ outMateHasBeenFound,
        bool* __restrict__ outMergedDifferentStrands,
        int outputPitch,
        const int* __restrict__ originalReadLengths,
        const int* __restrict__ dataExtendedReadLengths,
        const char* __restrict__ dataExtendedReadSequences,
        const char* __restrict__ dataExtendedReadQualities,
        const bool* __restrict__ dataMateHasBeenFound,
        const float* __restrict__ dataGoodScores,
        int inputPitch,
        int minFragmentSize,
        int maxFragmentSize,
        MakePairResultsStrictConfig config
    ){
        auto group = cg::this_thread_block();
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / group.size();
        const int groupIdInGrid = (threadIdx.x + blockDim.x * blockIdx.x) / group.size();

        extern __shared__ int smemForResults[];
        char* const smemChars = (char*)&smemForResults[0];
        char* const smemSequence = smemChars;
        char* const smemSequence2 = smemSequence + outputPitch;
        char* const smemQualities = smemSequence2 + outputPitch;

        using BlockReduce = cub::BlockReduce<int, blocksize>;

        __shared__ typename BlockReduce::TempStorage intreduce1;
        __shared__ int smembcast;

        auto checkIndex = [&](int index){
            assert(0 <= index);
            assert(index < outputPitch);
        };

        auto computeRead2Begin = [&](int i){
            if(dataMateHasBeenFound[i]){
                //extendedRead.length() - task.decodedMateRevC.length();
                return dataExtendedReadLengths[i] - originalReadLengths[i+1];
            }else{
                return -1;
            }
        };

        auto mergelength = [&](int l, int r){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthR = dataExtendedReadLengths[r];

            const int read2begin = computeRead2Begin(l);
   
            auto overlapstart = read2begin;

            const int resultsize = overlapstart + lengthR;
            
            return resultsize;
        };

        //merge extensions of the same pair and same strand and append to result
        auto merge = [&](auto& group, int l, int r, char* sequenceOutput, char* qualityOutput){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthL = dataExtendedReadLengths[l];
            const int lengthR = dataExtendedReadLengths[r];
            const int originalLengthR = originalReadLengths[r];
   
            auto overlapstart = computeRead2Begin(l);

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                checkIndex(i);
                sequenceOutput[i] 
                    = dataExtendedReadSequences[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                checkIndex(lengthL + i);
                checkIndex(originalLengthR + i);
                sequenceOutput[lengthL + i] 
                    = dataExtendedReadSequences[r * inputPitch + originalLengthR + i];
            }

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                checkIndex(i);
                qualityOutput[i] 
                    = dataExtendedReadQualities[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                checkIndex(lengthL + i);
                checkIndex(originalLengthR + i);
                qualityOutput[lengthL + i] 
                    = dataExtendedReadQualities[r * inputPitch + originalLengthR + i];
            }
        };
    
        //process pair at position pairIdsToProcess[posInList] and store to result position posInList
        for(int posInList = groupIdInGrid; posInList < numResults; posInList += numGroupsInGrid){
            group.sync(); //reuse smem

            const int p = posInList;

            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            char* const myResultSequence = outputSequences + posInList * outputPitch;
            char* const myResultQualities = outputQualities + posInList * outputPitch;
            int* const myResultRead1Begins = outRead1Begins + posInList;
            int* const myResultRead2Begins = outRead2Begins + posInList;
            int* const myResultLengths = outputLengths + posInList;
            bool* const myResultAnchorIsLR = outputAnchorIsLR + posInList;
            bool* const myResultMateHasBeenFound = outMateHasBeenFound + posInList;
            bool* const myResultMergedDifferentStrands = outMergedDifferentStrands + posInList;
            

            auto LRmatefoundfunc = [&](){
                const int extendedReadLength3 = dataExtendedReadLengths[i3];
                const int originalLength3 = originalReadLengths[i3];

                int resultsize = mergelength(i0, i1);
                if(extendedReadLength3 > originalLength3){
                    resultsize += extendedReadLength3 - originalLength3;
                }                

                int currentsize = 0;

                if(extendedReadLength3 > originalLength3){
                    //insert extensions of reverse complement of d3 at beginning

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k]
                        );
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        checkIndex(k);
                        checkIndex(originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k);
                        myResultQualities[k] = 
                            dataExtendedReadQualities[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k];
                    }

                    currentsize = (extendedReadLength3 - originalLength3);
                }

                merge(group, i0, i1, myResultSequence + currentsize, myResultQualities + currentsize);

                int read1begin = 0;
                int read2begin = computeRead2Begin(i0);

                if(extendedReadLength3 > originalLength3){                    
                    read1begin += (extendedReadLength3 - originalLength3);
                    read2begin += (extendedReadLength3 - originalLength3);
                }

                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            auto RLmatefoundfunc = [&](){
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int originalLength1 = originalReadLengths[i1];

                const int mergedLength = mergelength(i2, i3);
                int resultsize = mergedLength;
                if(extendedReadLength1 > originalLength1){
                    resultsize += extendedReadLength1 - originalLength1;
                }

                merge(group, i2, i3, smemSequence, smemQualities);

                group.sync();

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    checkIndex(k);
                    checkIndex(mergedLength - 1 - k);
                    myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                        smemSequence[mergedLength - 1 - k]
                    );
                }

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    checkIndex(k);
                    checkIndex(mergedLength - 1 - k);
                    myResultQualities[k] = smemQualities[mergedLength - 1 - k];
                }

                group.sync();

                const int sizeOfRightExtension = mergedLength - (computeRead2Begin(i2) + originalReadLengths[i3]);

                int read1begin = 0;
                int newread2begin = mergedLength - (read1begin + originalReadLengths[i2]);
                int newread1begin = sizeOfRightExtension;

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                #ifndef NDEBUG
                int newread2length = originalReadLengths[i2];
                int newread1length = originalReadLengths[i3];
                assert(newread1begin + newread1length <= mergedLength);
                assert(newread2begin + newread2length <= mergedLength);
                #endif

                if(extendedReadLength1 > originalLength1){
                    //insert extensions of d1 at end
                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(mergedLength + k);
                        checkIndex(originalLength1 + k);
                        myResultSequence[mergedLength + k] = 
                            dataExtendedReadSequences[i1 * inputPitch + originalLength1 + k];                        
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        checkIndex(mergedLength + k);
                        checkIndex(originalLength1 + k);
                        myResultQualities[mergedLength + k] = 
                            dataExtendedReadQualities[i1 * inputPitch + originalLength1 + k];                        
                    }
                }

                read1begin = newread1begin;
                const int read2begin = newread2begin;
                
                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = false;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            auto discardExtensionFunc = [&](){
                const int resultlength = originalReadLengths[i0];
                for(int k = group.thread_rank(); k < resultlength; k += group.size()){
                    checkIndex(k);
                    myResultSequence[k] = dataExtendedReadSequences[i0 * inputPitch + k];
                }

                for(int k = group.thread_rank(); k < resultlength; k += group.size()){
                    checkIndex(k);
                    myResultQualities[k] = dataExtendedReadQualities[i0 * inputPitch + k];
                }
                if(group.thread_rank() == 0){
                    *myResultRead1Begins = 0;
                    *myResultRead2Begins = -1;
                    *myResultLengths = resultlength;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = false;
                    *myResultMergedDifferentStrands = false;
                }
            };

            if(dataMateHasBeenFound[i0] && dataMateHasBeenFound[i2]){
                const int lengthDifference = dataExtendedReadLengths[i0] > dataExtendedReadLengths[i2] ? 
                    dataExtendedReadLengths[i0] - dataExtendedReadLengths[i2]
                    : dataExtendedReadLengths[i2] - dataExtendedReadLengths[i0];
                if(lengthDifference > config.maxLengthDifferenceIfBothFoundMate){
                    //do not extend
                    discardExtensionFunc();
                }else{
                    const int gapBegin = originalReadLengths[i0];
                    const int gapEnd = dataExtendedReadLengths[i0] - originalReadLengths[i2];
                    const int gapSize = std::max(0, gapEnd - gapBegin);

                    const int extendedPositionsOtherStrand = dataExtendedReadLengths[i2] - originalReadLengths[i2];
                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);
                    int matchingPositions = 0;
                    if(extendedPositionsOtherStrand > 0){
                        //find hamming distance of overlap betweens filled gaps of both strands
                        const int begin_i2 = originalReadLengths[i2];
                        const int end_i2 = begin_i2 + overlapsize;

                        for(int k = group.thread_rank(); k < overlapsize; k += group.size()){
                            const int pos_i0 = gapEnd - overlapsize + k;
                            const int pos_i2 = end_i2 - 1 - k;
                            checkIndex(pos_i0);
                            checkIndex(pos_i2);
                            const char c0 = dataExtendedReadSequences[i0 * inputPitch + pos_i0];
                            const char c2 = SequenceHelpers::complementBaseDecoded(dataExtendedReadSequences[i2 * inputPitch + pos_i2]);
                            matchingPositions += (c0 == c2);
                        }

                        matchingPositions = BlockReduce(intreduce1).Sum(matchingPositions);

                        if(threadIdx.x == 0){
                            smembcast = matchingPositions;
                        }
                        __syncthreads();
                        matchingPositions = smembcast;
                        __syncthreads();
                    }

                    if(overlapsize == matchingPositions){
                        LRmatefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }

                }               
            }else if(dataMateHasBeenFound[i0] && !dataMateHasBeenFound[i2]){
                if(!config.allowSingleStrand){
                    discardExtensionFunc();
                }else{

                    const int gapBegin = originalReadLengths[i0];
                    const int gapEnd = dataExtendedReadLengths[i0] - originalReadLengths[i2];
                    const int gapSize = std::max(0, gapEnd - gapBegin);

                    const int extendedPositionsOtherStrand = dataExtendedReadLengths[i2] - originalReadLengths[i2];
                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);

                    // if(threadIdx.x == 0){
                    //     printf("LR %d %d %d %d %d\n", gapBegin, gapEnd, gapSize, extendedPositionsOtherStrand, overlapsize);
                    // }
                    // __syncthreads();

                    int matchingPositions = 0;
                    if(extendedPositionsOtherStrand > 0){

                        //find hamming distance of overlap betweens filled gaps of both strands

                        const int begin_i2 = originalReadLengths[i2];
                        const int end_i2 = begin_i2 + overlapsize;
                        for(int k = group.thread_rank(); k < overlapsize; k += group.size()){
                            const int pos_i0 = gapEnd - overlapsize + k;
                            const int pos_i2 = end_i2 - 1 - k;
                            checkIndex(pos_i0);
                            checkIndex(pos_i2);
                            const char c0 = dataExtendedReadSequences[i0 * inputPitch + pos_i0];
                            const char c2 = SequenceHelpers::complementBaseDecoded(dataExtendedReadSequences[i2 * inputPitch + pos_i2]);
                            matchingPositions += (c0 == c2);

                        }

                        matchingPositions = BlockReduce(intreduce1).Sum(matchingPositions);

                        if(threadIdx.x == 0){
                            smembcast = matchingPositions;
                        }
                        __syncthreads();
                        matchingPositions = smembcast;
                        __syncthreads();
                    }

                    if(overlapsize >= gapSize * config.singleStrandMinOverlapWithOtherStrand 
                            && float(matchingPositions) / float(overlapsize) >= config.singleStrandMinMatchRateWithOtherStrand){
                        LRmatefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }
                }
                             
            }else if(dataMateHasBeenFound[i2] && !dataMateHasBeenFound[i0]){
                if(!config.allowSingleStrand){
                    discardExtensionFunc();
                }else{

                    const int gapBegin = originalReadLengths[i2];
                    const int gapEnd = dataExtendedReadLengths[i2] - originalReadLengths[i0];
                    const int gapSize = std::max(0, gapEnd - gapBegin);

                    const int extendedPositionsOtherStrand = dataExtendedReadLengths[i0] - originalReadLengths[i0];

                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);
                    // if(threadIdx.x == 0){
                    //     printf("RL %d %d %d %d %d\n", gapBegin, gapEnd, gapSize, extendedPositionsOtherStrand, overlapsize);
                    // }
                    // __syncthreads();
                    int matchingPositions = 0;
                    if(extendedPositionsOtherStrand > 0){

                        //find hamming distance of overlap betweens filled gaps of both strands

                        const int begin_i0 = originalReadLengths[i0];
                        const int end_i0 = begin_i0 + overlapsize;

                        for(int k = group.thread_rank(); k < overlapsize; k += group.size()){
                            const int pos_i2 = gapEnd - overlapsize + k;
                            const int pos_i0 = end_i0 - 1 - k;
                            checkIndex(pos_i2);
                            checkIndex(pos_i0);
                            const char c2 = dataExtendedReadSequences[i2 * inputPitch + pos_i2];
                            const char c0 = SequenceHelpers::complementBaseDecoded(dataExtendedReadSequences[i0 * inputPitch + pos_i0]);
                            matchingPositions += (c0 == c2);
                        }

                        matchingPositions = BlockReduce(intreduce1).Sum(matchingPositions);

                        if(threadIdx.x == 0){
                            smembcast = matchingPositions;
                        }
                        __syncthreads();
                        matchingPositions = smembcast;
                        __syncthreads();
                    }

                    if(overlapsize >= gapSize * config.singleStrandMinOverlapWithOtherStrand 
                            && float(matchingPositions) / float(overlapsize) >= config.singleStrandMinMatchRateWithOtherStrand){
                        RLmatefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }
                }
            }else{
                discardExtensionFunc();
            }

        }
    }







    template<int blocksize, int itemsPerThread, bool inclusive, class T>
    __global__
    void prefixSumSingleBlockKernel(
        const T* input,
        T* output,
        int N
    ){
        struct BlockPrefixCallbackOp{
            // Running prefix
            int running_total;

            __device__
            BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
            // Callback operator to be entered by the first warp of threads in the block.
            // Thread-0 is responsible for returning a value for seeding the block-wide scan.
            __device__
            int operator()(int block_aggregate){
                int old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        assert(blocksize == blockDim.x);

        using BlockScan = cub::BlockScan<T, blocksize>;
        using BlockLoad = cub::BlockLoad<T, blocksize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockStore = cub::BlockStore<T, blocksize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

        __shared__ typename BlockScan::TempStorage blockscantemp;
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
        } temp;

        T items[itemsPerThread];

        BlockPrefixCallbackOp prefix_op(0);

        const int iterations = SDIV(N, blocksize);

        int remaining = N;

        const T* currentInput = input;
        T* currentOutput = output;

        for(int iteration = 0; iteration < iterations; iteration++){
            const int valid_items = min(itemsPerThread * blocksize, remaining);

            BlockLoad(temp.load).Load(currentInput, items, valid_items, 0);

            if(inclusive){
                BlockScan(blockscantemp).InclusiveSum(
                    items, items, prefix_op
                );
            }else{
                BlockScan(blockscantemp).ExclusiveSum(
                    items, items, prefix_op
                );
            }
            __syncthreads();

            BlockStore(temp.store).Store(currentOutput, items, valid_items);
            __syncthreads();

            remaining -= valid_items;
            currentInput += valid_items;
            currentOutput += valid_items;
        }
    }


    template<class T, class U>
    __global__ 
    void setFirstSegmentIdsKernel(
        const T* __restrict__ segmentSizes,
        int* __restrict__ segmentIds,
        const U* __restrict__ segmentOffsets,
        int N
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            if(segmentSizes[i] > 0){
                segmentIds[segmentOffsets[i]] = i;
            }
        }
    }



    //flag candidates to remove because they are equal to anchor id or equal to mate id
    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    ){

        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ typename BlockReduceInt::TempStorage intreduce1;
        __shared__ typename BlockReduceInt::TempStorage intreduce2;

        for(int a = blockIdx.x; a < numTasks; a += gridDim.x){
            const int size = numCandidatesPerAnchor[a];
            const int offset = numCandidatesPerAnchorPrefixSum[a];
            const read_number anchorId = anchorReadIds[a];
            read_number mateId = 0;
            if(isPairedEnd){
                mateId = mateReadIds[a];
            }

            int mateIsRemoved = 0;
            int numRemoved = 0;

            // if(threadIdx.x == 0){
            //     printf("looking for anchor %u, mate %u\n", anchorId, mateId);
            // }
            __syncthreads();

            for(int i = threadIdx.x; i < size; i+= blockDim.x){
                bool keep = true;

                const read_number candidateId = candidateReadIds[offset + i];
                //printf("tid %d, comp %u at position %d\n", threadIdx.x, candidateId, offset + i);

                if(candidateId == anchorId){
                    keep = false;
                    numRemoved++;
                }

                if(isPairedEnd && candidateId == mateId){
                    if(keep){
                        keep = false;
                        numRemoved++;
                    }
                    mateIsRemoved++;
                    //printf("mate removed. i = %d\n", i);
                }

                keepflags[offset + i] = keep;
            }
            //printf("tid = %d, mateIsRemoved = %d\n", threadIdx.x, mateIsRemoved);
            int numRemovedBlock = BlockReduceInt(intreduce1).Sum(numRemoved);
            int numMateRemovedBlock = BlockReduceInt(intreduce2).Sum(mateIsRemoved);
            if(threadIdx.x == 0){
                numCandidatesPerAnchorOut[a] = size - numRemovedBlock;
                //printf("numMateRemovedBlock %d\n", numMateRemovedBlock);
                if(numMateRemovedBlock > 0){
                    mateRemovedFlags[a] = true;
                }else{
                    mateRemovedFlags[a] = false;
                }
            }
        }
    }
    

    template<int blocksize>
    __global__
    void reverseComplement2bitKernel(
        const int* __restrict__ lengths,
        const unsigned int* __restrict__ forward,
        unsigned int* __restrict__ reverse,
        int num,
        int encodedSequencePitchInInts
    ){

        for(int s = threadIdx.x + blockIdx.x * blockDim.x; s < num; s += blockDim.x * gridDim.x){
            const unsigned int* input = forward + encodedSequencePitchInInts * s;
            unsigned int* output = reverse + encodedSequencePitchInInts * s;
            const int length = lengths[s];

            SequenceHelpers::reverseComplementSequence2Bit(
                output,
                input,
                length,
                [](auto i){return i;},
                [](auto i){return i;}
            );
        }

        // constexpr int smemsizeints = blocksize * 16;
        // __shared__ unsigned int sharedsequences[smemsizeints]; //sequences will be stored transposed

        // const int sequencesPerSmem = std::min(blocksize, smemsizeints / encodedSequencePitchInInts);
        // assert(sequencesPerSmem > 0);

        // const int smemiterations = SDIV(num, sequencesPerSmem);

        // for(int smemiteration = blockIdx.x; smemiteration < smemiterations; smemiteration += gridDim.x){

        //     const int idBegin = smemiteration * sequencesPerSmem;
        //     const int idEnd = std::min((smemiteration+1) * sequencesPerSmem, num);

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             sharedsequences[intindex * sequencesPerSmem + s] = forward[encodedSequencePitchInInts * s + intindex];
        //         }
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         SequenceHelpers::reverseComplementSequenceInplace2Bit(&sharedsequences[s], lengths[s], [&](auto i){return i * sequencesPerSmem;});
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             reverse[encodedSequencePitchInInts * s + intindex] = sharedsequences[intindex * sequencesPerSmem + s];
        //         }
        //     }
        // }
    }

    template<int blocksize, int groupsize>
    __global__
    void filtermatekernel(
        const unsigned int* __restrict__ anchormatedata,
        const unsigned int* __restrict__ candidatefwddata,
        int encodedSequencePitchInInts,
        const int* __restrict__ numCandidatesPerAnchor,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const bool* __restrict__ mateIdHasBeenRemoved,
        int numTasks,
        bool* __restrict__ output_keepflags,
        int initialNumCandidates,
        const int* currentNumCandidatesPtr
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupindexinblock = threadIdx.x / groupsize;

        static_assert(blocksize % groupsize == 0);
        //constexpr int groupsperblock = blocksize / groupsize;

        extern __shared__ unsigned int smem[]; //sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts

        unsigned int* sharedMate = smem + groupindexinblock * encodedSequencePitchInInts;

        const int currentNumCandidates = *currentNumCandidatesPtr;

        for(int task = groupindex; task < numTasks; task += numgroups){
            const int numCandidates = numCandidatesPerAnchor[task];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[task];

            if(mateIdHasBeenRemoved[task]){

                for(int p = group.thread_rank(); p < encodedSequencePitchInInts; p++){
                    sharedMate[p] = anchormatedata[encodedSequencePitchInInts * task + p];
                }
                group.sync();

                //compare mate to candidates. 1 thread per candidate
                for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                    bool doKeep = false;
                    const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                    for(int p = 0; p < encodedSequencePitchInInts; p++){
                        const unsigned int aaa = sharedMate[p];
                        const unsigned int bbb = candidateptr[p];

                        if(aaa != bbb){
                            doKeep = true;
                            break;
                        }
                    }

                    output_keepflags[(candidatesOffset + c)] = doKeep;
                }

                group.sync();

            }
            // else{
            //     for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
            //         output_keepflags[(candidatesOffset + c)] = true;
            //     }
            // }
        }

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        for(int i = currentNumCandidates + tid; i < initialNumCandidates; i += stride){
            output_keepflags[i] = false;
        }
    }



    template<int groupsize>
    __global__
    void compactUsedIdsOfSelectedTasks(
        const int* __restrict__ indices,
        int numIndices,
        const read_number* __restrict__ d_usedReadIdsIn,
        read_number* __restrict__ d_usedReadIdsOut,
        //int* __restrict__ segmentIdsOut,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ inputSegmentOffsets,
        const int* __restrict__ outputSegmentOffsets
    ){
        const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / groupsize;
        const int numwarps = (blockDim.x * gridDim.x) / groupsize;
        const int lane = threadIdx.x % groupsize;

        for(int t = warpid; t < numIndices; t += numwarps){
            const int activeIndex = indices[t];
            const int num = d_numUsedReadIdsPerAnchor[t];
            const int inputOffset = inputSegmentOffsets[activeIndex];
            const int outputOffset = outputSegmentOffsets[t];

            for(int i = lane; i < num; i += groupsize){
                //copy read id
                d_usedReadIdsOut[outputOffset + i] = d_usedReadIdsIn[inputOffset + i];
                //set new segment id
                //segmentIdsOut[outputOffset + i] = t;
            }
        }
    }


    template<int blocksize, int groupsize>
    __global__
    void createGpuTaskData(
        int numReadPairs,
        const read_number* __restrict__ d_readpair_readIds,
        const int* __restrict__ d_readpair_readLengths,
        const unsigned int* __restrict__ d_readpair_sequences,
        const char* __restrict__ d_readpair_qualities,
        bool* __restrict__ pairedEnd, 
        bool* __restrict__ mateHasBeenFound,
        int* __restrict__ ids,
        int* __restrict__ pairIds,
        int* __restrict__ iteration,
        float* __restrict__ goodscore,
        read_number* __restrict__ d_anchorReadIds,
        read_number* __restrict__ d_mateReadIds,
        AbortReason* __restrict__ abortReason,
        ExtensionDirection* __restrict__ direction,
        unsigned int* __restrict__ inputEncodedMate,
        int* __restrict__ inputmateLengths,
        unsigned int* __restrict__ inputAnchorsEncoded,
        int* __restrict__ inputAnchorLengths,
        char* __restrict__ inputAnchorQualities,
        char* __restrict__ inputMateQualities,
        int decodedSequencePitchInBytes,
        int qualityPitchInBytes,
        int encodedSequencePitchInInts,
        char* __restrict__ extendedSequences,
        char* __restrict__ qualitiesOfExtendedSequences,
        int* __restrict__ extendedSequenceLengths,
        int extendedSequencePitchInBytes
    ){
        constexpr int numGroupsInBlock = blocksize / groupsize;

        __shared__ unsigned int sharedEncodedSequence[numGroupsInBlock][32];
        __shared__ unsigned int sharedEncodedSequence2[numGroupsInBlock][32];

        assert(encodedSequencePitchInInts <= 32);

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = groupId % numGroupsInBlock;
        const int numGroups = (blockDim.x * gridDim.x) / groupsize;
        
        const int numTasks = numReadPairs * 4;

        //handle scalars

        for(int t = threadIdx.x + blockIdx.x * blockDim.x; t < numTasks; t += blockDim.x * gridDim.x){

            const int inputPairId = t / 4;
            const int id = t % 4;

            mateHasBeenFound[t] = false;
            ids[t] = id;
            pairIds[t] = d_readpair_readIds[2 * inputPairId + 0] / 2;
            iteration[t] = 0;
            goodscore[t] = 0.0f;
            abortReason[t] = AbortReason::None;

            if(id == 0){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                d_mateReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                inputmateLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                pairedEnd[t] = true;
                direction[t] = ExtensionDirection::LR;
                extendedSequenceLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
            }else if(id == 1){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                inputmateLengths[t] = 0;
                pairedEnd[t] = false;
                direction[t] = ExtensionDirection::LR;
                extendedSequenceLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
            }else if(id == 2){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                d_mateReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                inputmateLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                pairedEnd[t] = true;
                direction[t] = ExtensionDirection::RL;
                extendedSequenceLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
            }else{
                //id == 3
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                inputmateLengths[t] = 0;
                pairedEnd[t] = false;
                direction[t] = ExtensionDirection::RL;
                extendedSequenceLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
            }
        }

        //handle sequences
        for(int t = groupId; t < numTasks; t += numGroups){
            const int inputPairId = t / 4;
            const int id = t % 4;

            const unsigned int* const myReadpairSequences = d_readpair_sequences + 2 * inputPairId * encodedSequencePitchInInts;
            const int* const myReadPairLengths = d_readpair_readLengths + 2 * inputPairId;

            unsigned int* const myAnchorSequence = inputAnchorsEncoded + t * encodedSequencePitchInInts;
            unsigned int* const myMateSequence = inputEncodedMate + t * encodedSequencePitchInInts;

            char* const extendedSequence = extendedSequences + t * extendedSequencePitchInBytes;

            if(id == 0){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k]; //anchor in shared
                    sharedEncodedSequence2[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k]; //mate in shared2
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence[groupIdInBlock][k];
                    myMateSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }

                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[0], extendedSequence);
   
            }else if(id == 1){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        &sharedEncodedSequence[groupIdInBlock][0],
                        myReadPairLengths[1],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence2[groupIdInBlock][0], myReadPairLengths[1], extendedSequence);
            }else if(id == 2){

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k]; //anchor in shared
                    sharedEncodedSequence2[groupIdInBlock][k] = myReadpairSequences[k]; //mate in shared2
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence[groupIdInBlock][k];
                    myMateSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }

                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[0], extendedSequence);
            }else{
                //id == 3
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        &sharedEncodedSequence[groupIdInBlock][0],
                        myReadPairLengths[0],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence2[groupIdInBlock][0], myReadPairLengths[0], extendedSequence);
            }
        }

        //handle qualities
        for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
            const int inputPairId = t / 4;
            const int id = t % 4;

            const int* const myReadPairLengths = d_readpair_readLengths + 2 * inputPairId;
            const char* const myReadpairQualities = d_readpair_qualities + 2 * inputPairId * qualityPitchInBytes;

            char* const myAnchorQualities = inputAnchorQualities + t * qualityPitchInBytes;
            char* const myMateQualityScores = inputMateQualities + t * qualityPitchInBytes;


            char* const extendedSequenceQuality = qualitiesOfExtendedSequences + t * extendedSequencePitchInBytes;

            //const int numInts = qualityPitchInBytes / sizeof(int);
            int l0 = myReadPairLengths[0];
            int l1 = myReadPairLengths[1];

            if(id == 0){
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    extendedSequenceQuality[k] = myReadpairQualities[k];
                    myAnchorQualities[k] = myReadpairQualities[k];
                }
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myMateQualityScores[k] = myReadpairQualities[qualityPitchInBytes + k];
                }
            }else if(id == 1){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    extendedSequenceQuality[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                }
            }else if(id == 2){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    extendedSequenceQuality[k] = myReadpairQualities[qualityPitchInBytes + k];
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + k];
                }
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myMateQualityScores[k] = myReadpairQualities[k];
                }
            }else{
                //id == 3
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    extendedSequenceQuality[k] = myReadpairQualities[l0 - 1 - k];
                    myAnchorQualities[k] = myReadpairQualities[l0 - 1 - k];
                }
            }
        }

    }


    struct ComputeTaskSplitGatherIndicesSmallInput{

        std::size_t staticSharedMemory = 0;
        std::size_t maxDynamicSharedMemory = 0;

        ComputeTaskSplitGatherIndicesSmallInput(){
            std::size_t* d_output;
            CUDACHECK(cudaMalloc(&d_output, sizeof(std::size_t)));

            readextendergpukernels::computeTaskSplitGatherIndicesSmallInputGetStaticSmemSizeKernel<256,16><<<1, 1, 0, cudaStreamPerThread>>>(d_output); CUDACHECKASYNC;

            CUDACHECK(cudaMemcpyAsync(&staticSharedMemory, d_output, sizeof(std::size_t), D2H, cudaStreamPerThread));
            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

            CUDACHECK(cudaFree(d_output));

            int device = 0;
            CUDACHECK(cudaGetDevice(&device));

            int smemoptin = 0;
            CUDACHECK(cudaDeviceGetAttribute(&smemoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            
            CUDACHECK(cudaFuncSetAttribute(
                readextendergpukernels::computeTaskSplitGatherIndicesSmallInputKernel<256,16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 
                smemoptin - staticSharedMemory
            ));

            maxDynamicSharedMemory = smemoptin - staticSharedMemory;
        }

        std::size_t getRequiredDynamicSharedMemory(int numTasks) const{
            return sizeof(int) * numTasks * 2;
        }

        bool computationPossible(int numTasks) const{
            return numTasks <= (256 * 16) && getRequiredDynamicSharedMemory(numTasks) <= maxDynamicSharedMemory;
        }

        void compute(
            int numTasks,
            int* d_positions4,
            int* d_positionsNot4,
            int* d_numPositions4_out,
            int* d_numPositionsNot4_out,
            const int* task_pairIds,
            const int* task_ids,
            const int* d_minmax_pairId,
            cudaStream_t stream
        ) const{

            std::size_t smem = getRequiredDynamicSharedMemory(numTasks);
            readextendergpukernels::computeTaskSplitGatherIndicesSmallInputKernel<256,16><<<1, 256, smem, stream>>>(
                numTasks,
                d_positions4,
                d_positionsNot4,
                d_numPositions4_out,
                d_numPositionsNot4_out,
                task_pairIds,
                task_ids,
                d_minmax_pairId
            ); CUDACHECKASYNC;
        }
    };

}

    
}
}


#endif
