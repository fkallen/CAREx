#include <config.hpp>

#include <cpu_alignment.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>
#include <alignmentorientation.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <msa.hpp>
#include <msasplits.hpp>
#include <readextender_common.hpp>
#include <qualityscoreweights.hpp>
#include <hostdevicefunctions.cuh>
#include <numeric>

#include <vector>
#include <algorithm>
#include <string>

namespace care{

#define DO_ONLY_REMOVE_MATE_IDS

//forward declaration
struct GpuExtensionStepper;

struct ReadExtenderCpu{
    friend struct GpuExtensionStepper;
public:

    ReadExtenderCpu() = default;

    ReadExtenderCpu(
        int maxextensionPerStep_,
        int maximumSequenceLength_,
        const CpuReadStorage& rs, 
        const CpuMinhasher& mh,
        const ProgramOptions& programOptions_,
        const cpu::QualityScoreConversion* qualityConversion_
    ) : 
        readStorage(&rs), minhasher(&mh), 
        qualityConversion(qualityConversion_),
        maxextensionPerStep(maxextensionPerStep_),
        maximumSequenceLength(maximumSequenceLength_),
        encodedSequencePitchInInts(SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength_)),
        decodedSequencePitchInBytes(maximumSequenceLength_),
        qualityPitchInBytes(maximumSequenceLength_),
        programOptions(programOptions_),
        minhashHandle{mh.makeMinhasherHandle()}{

        setActiveReadStorage(readStorage);
        setActiveMinhasher(minhasher);
    }

    ~ReadExtenderCpu(){
        if(minhasher!=nullptr){
            minhasher->destroyHandle(minhashHandle);
        }
    }

    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
    }

    std::vector<ExtendResult> extend(const std::vector<ExtendInput>& inputs, bool doExtraHash){
        auto tasks = makePairedEndTasksFromInput4(inputs.begin(), inputs.end(), programOptions.maxFragmentSize);
        
        auto extendedTasks = processPairedEndTasks(tasks, doExtraHash);

        auto extendResults = constructResults(
            extendedTasks
        );

        return extendResults;
    }

    void setMaxExtensionPerStep(int e) noexcept{
        maxextensionPerStep = e;
    }

    void setMinCoverageForExtension(int c) noexcept{
        minCoverageForExtension = c;
    }

    void setActiveMinhasher(const CpuMinhasher* active){
        if(active == nullptr){
            //reset to default
            activeMinhasher = minhasher;
        }else{
            activeMinhasher = active;
        }
    }

    void setActiveReadStorage(const CpuReadStorage* active){
        if(active == nullptr){
            //reset to default
            activeReadStorage = readStorage;
        }else{
            activeReadStorage = active;
        }
    }
     
private:

    struct ExtendWithMsaResult{
        bool mateHasBeenFound = false;
        AbortReason abortReason = AbortReason::None;
        int newLength = 0;
        int newAccumExtensionLength = 0;
        int sizeOfGapToMate = 0;
        std::string newAnchor = "";
        std::string newQuality = "";
    };    

    std::vector<Task>& processPairedEndTasks(
        std::vector<Task>& tasks,
        bool doExtraHash
    ) const{
 
        std::vector<int> indicesOfActiveTasks(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);

        while(indicesOfActiveTasks.size() > 0){
            //perform one extension iteration for active tasks

            doOneExtensionIteration(tasks, indicesOfActiveTasks, doExtraHash);

            //update list of active task indices

            indicesOfActiveTasks.erase(
                std::remove_if(
                    indicesOfActiveTasks.begin(), 
                    indicesOfActiveTasks.end(),
                    [&](int index){
                        return !tasks[index].isActive(programOptions.minFragmentSize, programOptions.maxFragmentSize);
                    }
                ),
                indicesOfActiveTasks.end()
            );
        }

        return tasks;
    }

    void doOneExtensionIteration(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks, bool doExtraHash) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            task.currentAnchor.resize(SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength));

            SequenceHelpers::encodeSequence2Bit(
                task.currentAnchor.data(), 
                task.extendedSequence.data() + task.extendedSequenceLength - task.currentAnchorLength,
                task.currentAnchorLength
            );
        }

        hashTimer.start();

        if(!doExtraHash){
            getCandidateReadIds(tasks, indicesOfActiveTasks);
        }else{
            getCandidateReadIdsWithExtraExtensionHash(tasks, indicesOfActiveTasks);
        }

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "Anchor: " << task.totalDecodedAnchors.back() << "\n";
        //         std::cerr << "iteration " << task.iteration << ", candidates raw\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }

        removeUsedIdsAndMateIds(tasks, indicesOfActiveTasks);

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "iteration " << task.iteration << ", candidates after remove\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }
        

        hashTimer.stop();

        computePairFlags(tasks, indicesOfActiveTasks);                

        collectTimer.start();        

        loadCandidateSequenceData(tasks, indicesOfActiveTasks);

        eraseDataOfRemovedMates(tasks, indicesOfActiveTasks);


        collectTimer.stop();

        /*
            Compute alignments
        */

        alignmentTimer.start();

        calculateAlignments(tasks, indicesOfActiveTasks);

        alignmentTimer.stop();

        alignmentFilterTimer.start();

        filterAlignments(tasks, indicesOfActiveTasks);

        alignmentFilterTimer.stop();

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "iteration " << task.iteration << ", candidates after alignment filter\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }

        msaTimer.start();

        computeMSAsAndExtendTasks(tasks, indicesOfActiveTasks);

        msaTimer.stop();       

        handleEarlyExitOfTasks4(tasks, indicesOfActiveTasks);

        /*
            update book-keeping of used candidates
        */  

        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + task.candidateReadIds.size());
            auto tmp_end = std::set_union(
                task.allUsedCandidateReadIdPairs.begin(),
                task.allUsedCandidateReadIdPairs.end(),
                task.candidateReadIds.begin(),
                task.candidateReadIds.end(),
                tmp.begin()
            );

            tmp.erase(tmp_end, tmp.end());

            std::swap(task.allUsedCandidateReadIdPairs, tmp);

            // const int numCandidates = task.candidateReadIds.size();

            // if(numCandidates > 0 && task.abortReason == AbortReason::None){
            //     assert(task.totalAnchorBeginInExtendedRead.size() >= 2);
            //     const int oldAccumExtensionsLength 
            //         = task.totalAnchorBeginInExtendedRead[task.totalAnchorBeginInExtendedRead.size() - 2];
            //     const int newAccumExtensionsLength = task.totalAnchorBeginInExtendedRead.back();
            //     const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

            //     std::vector<read_number> fullyUsedIds;

            //     for(int c = 0; c < numCandidates; c += 1){
            //         const int candidateLength = task.candidateSequenceLengths[c];
            //         const int shift = task.alignments[c].shift;

            //         if(candidateLength + shift <= task.currentAnchorLength + lengthOfExtension){
            //             fullyUsedIds.emplace_back(task.candidateReadIds[c]);
            //         }
            //     }

            //     std::vector<read_number> tmp2(task.allFullyUsedCandidateReadIdPairs.size() + fullyUsedIds.size());
            //     auto tmp2_end = std::set_union(
            //         task.allFullyUsedCandidateReadIdPairs.begin(),
            //         task.allFullyUsedCandidateReadIdPairs.end(),
            //         fullyUsedIds.begin(),
            //         fullyUsedIds.end(),
            //         tmp2.begin()
            //     );

            //     tmp2.erase(tmp2_end, tmp2.end());
            //     std::swap(task.allFullyUsedCandidateReadIdPairs, tmp2);

            //     assert(task.allFullyUsedCandidateReadIdPairs.size() <= task.allUsedCandidateReadIdPairs.size());
            // }

            task.iteration++;
        }
    }

    void getCandidateReadIdsSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int /*beginPos*/ = 0 // only positions [beginPos, readLength] are hashed
    ) const{

        result.clear();

        bool containsN = false;
        activeReadStorage->areSequencesAmbiguous(&containsN, &readId, 1);

        //exclude anchors with ambiguous bases
        if(!(programOptions.excludeAmbiguousReads && containsN)){

            int numValuesPerSequence = 0;
            int totalNumValues = 0;

            activeMinhasher->determineNumValues(
                minhashHandle,encodedRead,
                encodedSequencePitchInInts,
                &readLength,
                1,
                &numValuesPerSequence,
                totalNumValues
            );

            result.resize(totalNumValues);
            std::array<int, 2> offsets{};

            activeMinhasher->retrieveValues(
                minhashHandle,
                1,
                totalNumValues,
                result.data(),
                &numValuesPerSequence,
                offsets.data()
            );

            std::sort(result.begin(), result.end());
            result.erase(
                std::unique(result.begin(), result.end()),
                result.end()
            );

            //exclude candidates with ambiguous bases

            if(programOptions.excludeAmbiguousReads){
                auto minhashResultsEnd = std::remove_if(
                    result.begin(),
                    result.end(),
                    [&](read_number readId){
                        bool containsN = false;
                        activeReadStorage->areSequencesAmbiguous(&containsN, &readId, 1);
                        return containsN;
                    }
                );

                result.erase(minhashResultsEnd, result.end());
            }            

        }else{
            ; // no candidates
        }
    }

    void getCandidateReadIds(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        #if 0
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            getCandidateReadIdsSingle(
                task.candidateReadIds, 
                task.currentAnchor.data(), 
                task.currentAnchorLength,
                task.currentAnchorReadId
            );

        }
        #else
            const int numSequences = indicesOfActiveTasks.size();

            int totalNumValues = 0;
            std::vector<int> numValuesPerSequence(numSequences);

            {
                std::vector<unsigned int> sequences(encodedSequencePitchInInts * numSequences);
                std::vector<int> lengths(numSequences);

                for(int i = 0; i < numSequences; i++){
                    const auto& task = tasks[indicesOfActiveTasks[i]];
                    std::copy(task.currentAnchor.begin(), task.currentAnchor.end(), sequences.begin() + i * encodedSequencePitchInInts);
                    lengths[i] = task.currentAnchorLength;
                }


                activeMinhasher->determineNumValues(
                    minhashHandle,
                    sequences.data(),
                    encodedSequencePitchInInts,
                    lengths.data(),
                    numSequences,
                    numValuesPerSequence.data(),
                    totalNumValues
                );
            }

            std::vector<read_number> allCandidates(totalNumValues);
            std::vector<int> offsets(numSequences + 1);

            activeMinhasher->retrieveValues(
                minhashHandle,
                numSequences,
                totalNumValues,
                allCandidates.data(),
                numValuesPerSequence.data(),
                offsets.data()
            );


            #if 0
            auto iterator = allCandidates.data();
            for(int s = 0; s < numSequences; s++){
                const int first = offsets[s];
                const int last = first + numValuesPerSequence[s];
                std::sort(&allCandidates[first], &allCandidates[last]);
                auto uniqueEnd = std::unique_copy(&allCandidates[first], &allCandidates[last], iterator);
                numValuesPerSequence[s] = std::distance(iterator, uniqueEnd);
                iterator = uniqueEnd;
            }

            std::inclusive_scan(
                numValuesPerSequence.begin(), 
                numValuesPerSequence.end(),
                offsets.begin() + 1
            );
            offsets[0] = 0;
            #else
            for(int s = 0; s < numSequences; s++){
                const int first = offsets[s];
                const int last = first + numValuesPerSequence[s];
                std::sort(&allCandidates[first], &allCandidates[last]);
                auto uniqueEnd = std::unique(&allCandidates[first], &allCandidates[last]);
                numValuesPerSequence[s] = std::distance(&allCandidates[first], uniqueEnd);
            }

            #endif

            for(int i = 0; i < numSequences; i++){
                auto& task = tasks[indicesOfActiveTasks[i]];

                task.candidateReadIds.resize(numValuesPerSequence[i]);
                std::copy_n(allCandidates.begin() + offsets[i], numValuesPerSequence[i], task.candidateReadIds.begin());
            }

        #endif
    }

    void getCandidateReadIdsWithExtraExtensionHash(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{

        //for each task, two strings will be hashed. the current sequence, and the part which has been extended by the previous iteration
        const int numAnchors = indicesOfActiveTasks.size();        
        const int numSequences = 2 * numAnchors;

        int totalNumValues = 0;
        std::vector<int> numValuesPerSequence(numSequences);

        {
            std::vector<unsigned int> sequences(encodedSequencePitchInInts * numSequences);
            std::vector<int> lengths(numSequences);

            for(int i = 0; i < numAnchors; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                std::copy(task.currentAnchor.begin(), task.currentAnchor.end(), sequences.begin() + i * encodedSequencePitchInInts);
                lengths[i] = task.currentAnchorLength;
            }

            const int kmersize = activeMinhasher->getKmerSize();
            const int extralength = SDIV(maxextensionPerStep + kmersize - 1, 4) * 4; //rounded up to multiple of 4

            std::vector<char> buffer(encodedSequencePitchInInts * SequenceHelpers::basesPerInt2Bit());

            for(int i = 0; i < numAnchors; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                const int dataIndex = numAnchors + i;

                if(task.iteration > 0){
                    const int end = task.currentAnchorLength;
                    const int begin = std::max(0, end - extralength);
                    SequenceHelpers::decode2BitSequence(
                        buffer.data(), 
                        task.currentAnchor.data(), 
                        task.currentAnchorLength
                    );
                    SequenceHelpers::encodeSequence2Bit(
                        sequences.data() + dataIndex * encodedSequencePitchInInts,
                        buffer.data() + begin,
                        end - begin
                    );

                    lengths[dataIndex] = end - begin;
                }else{
                    lengths[dataIndex] = 0;
                }
            }


            activeMinhasher->determineNumValues(
                minhashHandle,
                sequences.data(),
                encodedSequencePitchInInts,
                lengths.data(),
                numSequences,
                numValuesPerSequence.data(),
                totalNumValues
            );
        }

        std::vector<read_number> allCandidates(totalNumValues);
        std::vector<int> offsets(numSequences + 1);

        activeMinhasher->retrieveValues(
            minhashHandle,
            numSequences,
            totalNumValues,
            allCandidates.data(),
            numValuesPerSequence.data(),
            offsets.data()
        );

        #if 0
        auto iterator = allCandidates.data();
        for(int s = 0; s < numSequences; s++){
            const int first = offsets[s];
            const int last = first + numValuesPerSequence[s];
            std::sort(&allCandidates[first], &allCandidates[last]);
            auto uniqueEnd = std::unique_copy(&allCandidates[first], &allCandidates[last], iterator);
            numValuesPerSequence[s] = std::distance(iterator, uniqueEnd);
            iterator = uniqueEnd;
        }

        std::inclusive_scan(
            numValuesPerSequence.begin(), 
            numValuesPerSequence.end(),
            offsets.begin() + 1
        );
        offsets[0] = 0;
        #else
        for(int s = 0; s < numSequences; s++){
            const int first = offsets[s];
            const int last = first + numValuesPerSequence[s];
            std::sort(&allCandidates[first], &allCandidates[last]);
            auto uniqueEnd = std::unique(&allCandidates[first], &allCandidates[last]);
            numValuesPerSequence[s] = std::distance(&allCandidates[first], uniqueEnd);
        }

        #endif

        for(int i = 0; i < numAnchors; i++){
            auto& task = tasks[indicesOfActiveTasks[i]];

            //merge both lists of hash values per task
            const int numNormal = numValuesPerSequence[i];
            const int numExtra = numValuesPerSequence[numAnchors + i];
            const int offsetNormal = offsets[i];
            const int offsetExtra = offsets[numAnchors + i];

            task.candidateReadIds.resize(numNormal + numExtra);
            auto newend = std::set_union(
                allCandidates.begin() + offsetNormal,
                allCandidates.begin() + offsetNormal + numNormal,
                allCandidates.begin() + offsetExtra,
                allCandidates.begin() + offsetExtra + numExtra,
                task.candidateReadIds.begin()
            );
            task.candidateReadIds.erase(newend, task.candidateReadIds.end());
        }
    }

    void removeUsedIdsAndMateIds(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            // remove self from candidate list
            auto readIdPos = std::lower_bound(
                task.candidateReadIds.begin(),                                            
                task.candidateReadIds.end(),
                task.myReadId
            );

            if(readIdPos != task.candidateReadIds.end() && *readIdPos == task.myReadId){
                task.candidateReadIds.erase(readIdPos);
            }

            if(task.pairedEnd){

                //remove mate of input from candidate list
                auto mateReadIdPos = std::lower_bound(
                    task.candidateReadIds.begin(),                                            
                    task.candidateReadIds.end(),
                    task.mateReadId
                );

                if(mateReadIdPos != task.candidateReadIds.end() && *mateReadIdPos == task.mateReadId){
                    task.candidateReadIds.erase(mateReadIdPos);
                    task.mateRemovedFromCandidates = true;
                }
            }

            /*
                Remove candidate pairs which have already been used for extension
            */
            #ifndef DO_ONLY_REMOVE_MATE_IDS

            // for(int indexOfActiveTask : indicesOfActiveTasks){
            //     auto& task = tasks[indexOfActiveTask];

            //     std::vector<read_number> tmp(task.candidateReadIds.size());

            //     auto end = std::set_difference(
            //         task.candidateReadIds.begin(),
            //         task.candidateReadIds.end(),
            //         task.allFullyUsedCandidateReadIdPairs.begin(),
            //         task.allFullyUsedCandidateReadIdPairs.end(),
            //         tmp.begin()
            //     );

            //     tmp.erase(end, tmp.end());

            //     std::swap(task.candidateReadIds, tmp);
            // }

            #endif
        }
    }

    void loadCandidateSequenceData(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        #if 0
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
            task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            activeReadStorage->gatherSequenceLengths(
                task.candidateSequenceLengths.data(),
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            activeReadStorage->gatherSequences(
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            for(int c = 0; c < numCandidates; c++){
                const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;
                unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                SequenceHelpers::reverseComplementSequence2Bit(
                    seqrevcPtr,  
                    seqPtr,
                    task.candidateSequenceLengths[c]
                );
            }
        }
        #else
            const int numSequences = indicesOfActiveTasks.size();

            std::vector<int> offsets(numSequences);
            offsets[0] = 0;

            int totalNumberOfCandidates = 0;
            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                totalNumberOfCandidates += task.candidateReadIds.size();
                if(i < numSequences - 1){
                    offsets[i+1] = totalNumberOfCandidates;
                }
            }

            std::vector<read_number> readIds(totalNumberOfCandidates);
            std::vector<int> lengths(totalNumberOfCandidates);
            std::vector<unsigned int> forwarddata(totalNumberOfCandidates * encodedSequencePitchInInts);

            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                
                std::copy(task.candidateReadIds.begin(), task.candidateReadIds.end(), readIds.begin() + offsets[i]);
            }

            activeReadStorage->gatherSequenceLengths(
                lengths.data(),
                readIds.data(),
                totalNumberOfCandidates
            );

            activeReadStorage->gatherSequences(
                forwarddata.data(),
                encodedSequencePitchInInts,
                readIds.data(),
                totalNumberOfCandidates
            );

            for(int i = 0; i < numSequences; i++){
                auto& task = tasks[indicesOfActiveTasks[i]];
                const int numCandidates = task.candidateReadIds.size();
                const int offset = offsets[i];

                task.candidateSequenceLengths.resize(numCandidates);
                task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                
                std::copy_n(lengths.begin() + offset, numCandidates, task.candidateSequenceLengths.begin());
                std::copy_n(
                    forwarddata.begin() + offset * encodedSequencePitchInInts, 
                    numCandidates * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.begin()
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    SequenceHelpers::reverseComplementSequence2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        task.candidateSequenceLengths[c]
                    );
                }
            }

        #endif
    }

    void computePairFlags(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        const int numTasks = indicesOfActiveTasks.size();

        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            task.isPairedCandidate.resize(task.candidateReadIds.size());
            std::fill(task.isPairedCandidate.begin(), task.isPairedCandidate.end(), false);
        }

        for(int first = 0, second = 1; second < numTasks; ){
            const int taskindex1 = indicesOfActiveTasks[first];
            const int taskindex2 = indicesOfActiveTasks[second];

            const bool areConsecutiveTasks = tasks[taskindex1].id + 1 == tasks[taskindex2].id;
            const bool arePairedTasks = (tasks[taskindex1].id % 2) + 1 == (tasks[taskindex2].id % 2);

            assert(tasks[taskindex1].isPairedCandidate.size() ==  tasks[taskindex1].candidateReadIds.size());
            assert(tasks[taskindex2].isPairedCandidate.size() ==  tasks[taskindex2].candidateReadIds.size());

            if(areConsecutiveTasks && arePairedTasks){
                //check for pairs in current candidates
                {
                    const int begin1 = 0;
                    const int end1 = tasks[taskindex1].candidateReadIds.size();
                    const int begin2 = 0;
                    const int end2 = tasks[taskindex2].candidateReadIds.size();

                    // assert(std::is_sorted(pairIds + begin1, pairIds + end1));
                    // assert(std::is_sorted(pairIds + begin2, pairIds + end2));

                    std::vector<int> pairedPositions(std::min(end1-begin1, end2-begin2));
                    std::vector<int> pairedPositions2(std::min(end1-begin1, end2-begin2));

                    auto endIters = findPositionsOfPairedReadIds(
                        tasks[taskindex1].candidateReadIds.begin() + begin1,
                        tasks[taskindex1].candidateReadIds.begin() + end1,
                        tasks[taskindex2].candidateReadIds.begin() + begin2,
                        tasks[taskindex2].candidateReadIds.begin() + end2,
                        pairedPositions.begin(),
                        pairedPositions2.begin()
                    );

                    pairedPositions.erase(endIters.first, pairedPositions.end());
                    pairedPositions2.erase(endIters.second, pairedPositions2.end());
                    
                    for(auto i : pairedPositions){
                        tasks[taskindex1].isPairedCandidate[begin1 + i] = true;
                    }
                    for(auto i : pairedPositions2){
                        tasks[taskindex2].isPairedCandidate[begin2 + i] = true;
                    }
                }

                //check for pairs in candidates of previous extension iterations
                {
                    const int end1 = tasks[taskindex1].candidateReadIds.size();
                    const int end2 =  tasks[taskindex2].allUsedCandidateReadIdPairs.size();

                    const int maxNumPositions = std::min(end1, end2);

                    std::vector<int> pairedPositions(maxNumPositions);
                    std::vector<int> pairedPositions2(maxNumPositions);

                    auto endIters = findPositionsOfPairedReadIds(
                        tasks[taskindex1].candidateReadIds.begin(),
                        tasks[taskindex1].candidateReadIds.begin() + end1,
                        tasks[taskindex2].allUsedCandidateReadIdPairs.begin(),
                        tasks[taskindex2].allUsedCandidateReadIdPairs.begin() + end2,
                        pairedPositions.begin(),
                        pairedPositions2.begin()
                    );

                    pairedPositions.erase(endIters.first, pairedPositions.end());
                    
                    for(auto i : pairedPositions){
                        tasks[taskindex1].isPairedCandidate[i] = true;
                    }
                }

                //check for pairs in candidates of previous extension iterations
                {
                    const int end1 = tasks[taskindex2].candidateReadIds.size();
                    const int end2 =  tasks[taskindex1].allUsedCandidateReadIdPairs.size();

                    const int maxNumPositions = std::min(end1, end2);

                    std::vector<int> pairedPositions(maxNumPositions);
                    std::vector<int> pairedPositions2(maxNumPositions);

                    auto endIters = findPositionsOfPairedReadIds(
                        tasks[taskindex2].candidateReadIds.begin(),
                        tasks[taskindex2].candidateReadIds.begin() + end1,
                        tasks[taskindex1].allUsedCandidateReadIdPairs.begin(),
                        tasks[taskindex1].allUsedCandidateReadIdPairs.begin() + end2,
                        pairedPositions.begin(),
                        pairedPositions2.begin()
                    );

                    pairedPositions.erase(endIters.first, pairedPositions.end());
                    
                    for(auto i : pairedPositions){
                        tasks[taskindex2].isPairedCandidate[i] = true;
                    }
                }
                
                first += 2; second += 2;
            }else{
                first += 1; second += 1;
            }

            
        }
    }

    void eraseDataOfRemovedMates(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            if(task.mateRemovedFromCandidates){
                const int numCandidates = task.candidateReadIds.size();

                std::vector<int> positionsOfCandidatesToKeep;
                positionsOfCandidatesToKeep.reserve(numCandidates);

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                    auto mismatchIters = std::mismatch(
                        task.encodedMate.begin(), task.encodedMate.end(),
                        seqPtr, seqPtr + encodedSequencePitchInInts
                    );

                    //candidate differs from mate
                    if(mismatchIters.first != task.encodedMate.end()){                            
                        positionsOfCandidatesToKeep.emplace_back(c);
                    }else{
                        ;//std::cerr << "";
                    }
                }

                //compact
                const int toKeep = positionsOfCandidatesToKeep.size();
                for(int c = 0; c < toKeep; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    task.candidateReadIds[c] = task.candidateReadIds[index];
                    task.candidateSequenceLengths[c] = task.candidateSequenceLengths[index];

                    std::copy_n(
                        task.candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts
                    );

                    task.isPairedCandidate[c] = task.isPairedCandidate[index];

                    
                }

                //erase
                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + toKeep, 
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + toKeep, 
                    task.candidateSequenceLengths.end()
                );
                task.candidateSequencesFwdData.erase(
                    task.candidateSequencesFwdData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.end()
                );
                task.candidateSequencesRevcData.erase(
                    task.candidateSequencesRevcData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesRevcData.end()
                );
                task.isPairedCandidate.erase(
                    task.isPairedCandidate.begin() + toKeep,
                    task.isPairedCandidate.end()
                );

                task.mateRemovedFromCandidates = false;
            }

        }
    }

    void calculateAlignments(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            std::vector<care::cpu::SHDResult> forwardAlignments;
            std::vector<care::cpu::SHDResult> revcAlignments;

            forwardAlignments.resize(numCandidates);
            revcAlignments.resize(numCandidates);
            task.alignmentFlags.resize(numCandidates);
            task.alignments.resize(numCandidates);

            care::cpu::shd::cpuShiftedHammingDistance<care::cpu::shd::ShiftDirection::Right>(
                forwardAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                programOptions.min_overlap,
                programOptions.maxErrorRate,
                programOptions.min_overlap_ratio
            );

            care::cpu::shd::cpuShiftedHammingDistance<care::cpu::shd::ShiftDirection::Right>(
                revcAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesRevcData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                programOptions.min_overlap,
                programOptions.maxErrorRate,
                programOptions.min_overlap_ratio
            );

            //decide whether to keep forward or reverse complement, and keep it

            for(int c = 0; c < numCandidates; c++){
                const auto& forwardAlignment = forwardAlignments[c];
                const auto& revcAlignment = revcAlignments[c];
                const int candidateLength = task.candidateSequenceLengths[c];

                task.alignmentFlags[c] = care::chooseBestAlignmentOrientation(
                    forwardAlignment,
                    revcAlignment,
                    task.currentAnchorLength,
                    candidateLength,
                    programOptions.min_overlap_ratio,
                    programOptions.min_overlap,
                    0.06f
                );

                if(task.alignmentFlags[c] == AlignmentOrientation::Forward){
                    task.alignments[c] = forwardAlignment;
                }else{
                    task.alignments[c] = revcAlignment;
                }
            }
        }
    }

    void filterAlignments(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            /*
                Remove bad alignments
            */        

            const int size = task.alignments.size();

            std::vector<bool> keepflags(size, true);
            int removed = 0;
            bool goodAlignmentExists = false;
            float relativeOverlapThreshold = 0.0f;

            for(int c = 0; c < size; c++){
                const AlignmentOrientation alignmentFlag0 = task.alignmentFlags[c];
                const int shift = task.alignments[c].shift;
                
                if(alignmentFlag0 != AlignmentOrientation::None && shift >= 0){
                    if(!task.isPairedCandidate[c]){
                        const float overlap = task.alignments[c].overlap;
                        const float relativeOverlap = overlap / float(task.currentAnchorLength);

                        if(relativeOverlap < 1.0f && fgeq(relativeOverlap, programOptions.min_overlap_ratio)){
                            goodAlignmentExists = true;
                            const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                            relativeOverlapThreshold = fmaxf(relativeOverlapThreshold, tmp);
                            // if(task.id == 1 && task.iteration == 14){
                            //     printf("%d %f %f %f\n", c, relativeOverlap, tmp, relativeOverlapThreshold);
                            // }
                        }
                    }
                }else{
                    keepflags[c] = false;
                    removed++;
                }
            }

            if(goodAlignmentExists){

                for(int c = 0; c < size; c++){
                    if(!task.isPairedCandidate[c]){
                        if(keepflags[c]){
                            const float overlap = task.alignments[c].overlap;
                            const float relativeOverlap = overlap / float(task.currentAnchorLength);                

                            if(!fgeq(relativeOverlap, relativeOverlapThreshold)){
                                keepflags[c] = false;
                                removed++;
                            }
                        }
                    }
                }
            }else{
                //NOOP.
                //if no good alignment exists, no other candidate is removed. we will try to work with the not-so-good alignments
            }


            // if(task.pairId == 87680 / 2 && task.id == 1){
            //     if(task.iteration == 7){
                    // std::cout << "candidates before filter\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << task.candidateReadIds[i] << " ";
                    // }
                    // std::cout << "\n";
                    // std::cout << "isPairedCandidate\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << task.isPairedCandidate[i] << " ";
                    // }
                    // std::cout << "\n";
                    // std::cout << "orientations\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << int(task.alignmentFlags[i]) << " ";
                    // }
                    // std::cout << "\n";
                    // std::cout << "overlaps\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << task.alignments[i].overlap << " ";
                    // }
                    // std::cout << "\n";

                    // std::cout << "keepflags\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << keepflags[i] << " ";
                    // }
                    // std::cout << "\n";

                    // std::cout << "anchor\n";
                    // for(int i = 0; i < task.currentAnchorLength; i++){
                    //     std::cout << SequenceHelpers::decodeBase(SequenceHelpers::getEncodedNuc2Bit(task.currentAnchor.data(), task.currentAnchorLength, i));
                    // }
                    // std::cout << "\n";

                    // std::cout << "candidates\n";
                    // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
                    //     std::cout << task.candidateReadIds[i] << " " << task.isPairedCandidate[i] << " " 
                    //         << int(task.alignmentFlags[i]) << " " << task.alignments[i].overlap << " " 
                    //         << task.alignments[i].shift << " " << keepflags[i] << "\n";
                    // }
            //     }
            // }


            task.numRemainingCandidates = 0;

            //compact inplace
            task.candidateSequenceData.resize((size - removed) * encodedSequencePitchInInts);

            for(int c = 0; c < size; c++){
                if(keepflags[c]){
                    task.alignments[task.numRemainingCandidates] = task.alignments[c];
                    task.alignmentFlags[task.numRemainingCandidates] = task.alignmentFlags[c];
                    task.candidateReadIds[task.numRemainingCandidates] = task.candidateReadIds[c];
                    task.candidateSequenceLengths[task.numRemainingCandidates] = task.candidateSequenceLengths[c];
                    task.isPairedCandidate[task.numRemainingCandidates] = task.isPairedCandidate[c];

                    assert(task.alignmentFlags[c] != AlignmentOrientation::None);

                    if(task.alignmentFlags[c] == AlignmentOrientation::Forward){
                        std::copy_n(
                            task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            task.candidateSequenceData.data() + task.numRemainingCandidates * encodedSequencePitchInInts
                        );
                    }else{
                        //AlignmentOrientation::ReverseComplement

                        std::copy_n(
                            task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            task.candidateSequenceData.data() + task.numRemainingCandidates * encodedSequencePitchInInts
                        );
                    }

                    task.numRemainingCandidates++;
                }                
            }

            assert(task.numRemainingCandidates + removed == size);

            //erase past-end elements
            task.alignments.erase(
                task.alignments.begin() + task.numRemainingCandidates, 
                task.alignments.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + task.numRemainingCandidates, 
                task.alignmentFlags.end()
            );
            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + task.numRemainingCandidates, 
                task.candidateReadIds.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + task.numRemainingCandidates, 
                task.candidateSequenceLengths.end()
            );
            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + task.numRemainingCandidates, 
                task.isPairedCandidate.end()
            );

            task.candidateSequencesFwdData.clear();
            task.candidateSequencesRevcData.clear();

            if(task.numRemainingCandidates == 0){
                task.abort = true;
                task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
            }

            // if(task.pairId == 87680 / 2 && task.id == 1){
            //     if(task.iteration == 7){
            //         std::cout << "candidates after filter\n";
            //         for(int i = 0; i < int(task.candidateReadIds.size()); i++){
            //             std::cout << task.candidateReadIds[i] << " ";
            //         }
            //         std::cout << "\n";
            //     }
            // }

            // std::cerr << "candidates of task " << task.id << " after filter in iteration "<< task.iteration << ":\n";
            // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
            //     std::cerr << task.candidateReadIds[i] << " ";
            // }
            // std::cerr << "\n";

        }
    }

    MultipleSequenceAlignment constructMSA(Task& task, char* candidateQualities) const{
        MultipleSequenceAlignment msa(qualityConversion);

        const bool useQualityScoresForMSA = true;

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    programOptions.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = useQualityScoresForMSA;
            msaInput.anchorLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = qualityPitchInBytes;
            msaInput.anchor = task.extendedSequence.data() + task.extendedSequenceLength - task.currentAnchorLength;
            msaInput.candidates = task.candidateStrings.data();
            msaInput.anchorQualities = task.qualityOfExtendedSequence.data() + task.extendedSequenceLength - task.currentAnchorLength;
            //msaInput.candidateQualities = candidateQualities.data();
            msaInput.candidateQualities = candidateQualities;
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
            const int numCandidates = task.candidateReadIds.size();

            int insertpos = 0;
            for(int i = 0; i < numCandidates; i++){
                if(!minimizationResult.differentRegionCandidate[i]){               
                    //keep candidate

                    task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                    std::copy_n(
                        task.candidateSequenceData.data() + i * size_t(encodedSequencePitchInInts),
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.data() + insertpos * size_t(encodedSequencePitchInInts)
                    );

                    task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i];
                    task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                    task.candidateShifts[insertpos] = task.candidateShifts[i];
                    task.isPairedCandidate[insertpos] = task.isPairedCandidate[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    std::copy_n(                        
                        candidateQualities + i * size_t(qualityPitchInBytes),
                        qualityPitchInBytes,
                        candidateQualities + insertpos * size_t(qualityPitchInBytes)
                    );

                    insertpos++;
                }
            }

            task.numRemainingCandidates = insertpos;

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + insertpos, 
                task.candidateReadIds.end()
            );
            task.candidateSequenceData.erase(
                task.candidateSequenceData.begin() + encodedSequencePitchInInts * insertpos, 
                task.candidateSequenceData.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + insertpos, 
                task.candidateSequenceLengths.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + insertpos, 
                task.alignmentFlags.end()
            );
            task.alignments.erase(
                task.alignments.begin() + insertpos, 
                task.alignments.end()
            );

            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + insertpos, 
                task.isPairedCandidate.end()
            );

            // candidateQualities.erase(
            //     candidateQualities.begin() + qualityPitchInBytes * insertpos, 
            //     candidateQualities.end()
            // );

            task.candidateStrings.erase(
                task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                task.candidateStrings.end()
            );
            task.candidateOverlapWeights.erase(
                task.candidateOverlapWeights.begin() + insertpos, 
                task.candidateOverlapWeights.end()
            );
            task.candidateShifts.erase(
                task.candidateShifts.begin() + insertpos, 
                task.candidateShifts.end()
            );
            
        };

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
                const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                    programOptions.estimatedCoverage
                );

                if(minimizationResult.performedMinimization){
                    removeCandidatesOfDifferentRegion(minimizationResult);

                    //build minimized multiple sequence alignment
                    build();
                }else{
                    break;
                }               
                
            }
        }   

        #endif

        // if(task.pairId == 87680 / 2 && task.id == 1){
        //     if(task.iteration <= 7){
        //         std::cout << "candidates after msa refinement\n";
        //         for(int i = 0; i < int(task.candidateReadIds.size()); i++){
        //             std::cout << task.candidateReadIds[i] << " ";
        //         }
        //         std::cout << "\n";

        //         std::cout << "consensus\n";
        //         for(int i = 0; i < msa.nColumns; i++){
        //             std::cout << msa.consensus[i];
        //         }
        //         std::cout << "\n";

        //         msa.print(std::cout);
        //         std::cout << "\n";
        //     }
        // }

        return msa;
    }

    MultipleSequenceAlignment constructMSA(Task& task) const{
        MultipleSequenceAlignment msa(qualityConversion);

        std::vector<char> candidateQualities(task.numRemainingCandidates * qualityPitchInBytes);

        if(programOptions.useQualityScores){

            activeReadStorage->gatherQualities(
                candidateQualities.data(),
                qualityPitchInBytes,
                task.candidateReadIds.data(),
                task.numRemainingCandidates
            );

            for(int c = 0; c < task.numRemainingCandidates; c++){
                if(task.alignmentFlags[c] == AlignmentOrientation::ReverseComplement){
                    std::reverse(
                        candidateQualities.data() + c * qualityPitchInBytes,
                        candidateQualities.data() + c * qualityPitchInBytes + task.candidateSequenceLengths[c]
                    );
                }
            }

        }else{
            std::fill(candidateQualities.begin(), candidateQualities.end(), 'I');
        }

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    programOptions.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = true;
            msaInput.anchorLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = qualityPitchInBytes;
            msaInput.anchor = task.extendedSequence.data() + task.extendedSequenceLength - task.currentAnchorLength;
            msaInput.candidates = task.candidateStrings.data();
            msaInput.anchorQualities = task.qualityOfExtendedSequence.data() + task.extendedSequenceLength - task.currentAnchorLength;
            msaInput.candidateQualities = candidateQualities.data();
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
            const int numCandidates = task.candidateReadIds.size();

            int insertpos = 0;
            for(int i = 0; i < numCandidates; i++){
                if(!minimizationResult.differentRegionCandidate[i]){               
                    //keep candidate

                    task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                    std::copy_n(
                        task.candidateSequenceData.data() + i * size_t(encodedSequencePitchInInts),
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.data() + insertpos * size_t(encodedSequencePitchInInts)
                    );

                    task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i];
                    task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                    task.candidateShifts[insertpos] = task.candidateShifts[i];
                    task.isPairedCandidate[insertpos] = task.isPairedCandidate[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    std::copy_n(                        
                        candidateQualities.data() + i * size_t(qualityPitchInBytes),
                        qualityPitchInBytes,
                        candidateQualities.data() + insertpos * size_t(qualityPitchInBytes)
                    );

                    insertpos++;
                }
            }

            task.numRemainingCandidates = insertpos;

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + insertpos, 
                task.candidateReadIds.end()
            );
            task.candidateSequenceData.erase(
                task.candidateSequenceData.begin() + encodedSequencePitchInInts * insertpos, 
                task.candidateSequenceData.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + insertpos, 
                task.candidateSequenceLengths.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + insertpos, 
                task.alignmentFlags.end()
            );
            task.alignments.erase(
                task.alignments.begin() + insertpos, 
                task.alignments.end()
            );

            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + insertpos, 
                task.isPairedCandidate.end()
            );

            candidateQualities.erase(
                candidateQualities.begin() + qualityPitchInBytes * insertpos, 
                candidateQualities.end()
            );

            task.candidateStrings.erase(
                task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                task.candidateStrings.end()
            );
            task.candidateOverlapWeights.erase(
                task.candidateOverlapWeights.begin() + insertpos, 
                task.candidateOverlapWeights.end()
            );
            task.candidateShifts.erase(
                task.candidateShifts.begin() + insertpos, 
                task.candidateShifts.end()
            );
            
        };

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
                const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                    programOptions.estimatedCoverage
                );

                if(minimizationResult.performedMinimization){
                    removeCandidatesOfDifferentRegion(minimizationResult);

                    //build minimized multiple sequence alignment
                    build();
                }else{
                    break;
                }               
                
            }
        }   

        #endif

        // if(task.pairId == 87680 / 2 && task.id == 1){
        //     if(task.iteration == 7){
        //         std::cout << "candidates after msa refinement\n";
        //         for(int i = 0; i < int(task.candidateReadIds.size()); i++){
        //             std::cout << task.candidateReadIds[i] << " ";
        //         }
        //         std::cout << "\n";
        //     }
        // }

        return msa;
    }

    void extendWithMsa(Task& task, const MultipleSequenceAlignment& msa) const{
        
        // if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //     std::cerr << "task id " << task.id << " myReadId " << task.myReadId << "\n";
        //     std::cerr << "candidates\n";
        //     for(auto x : task.candidateReadIds){
        //         std::cerr << x << " ";
        //     }
        //     std::cerr << "\n";

        //     std::cerr << "consensus\n";
        //     for(auto x : msa.consensus){
        //         std::cerr << x;
        //     }
        //     std::cerr << "\n";

        //     if(task.iteration == 3){
        //         const int num = task.numRemainingCandidates;
        //         std::cerr << "cand strings\n";
        //         for(int k = 0; k < num; k++){
        //             for(int c = 0; c < task.candidateSequenceLengths[k]; c++){
        //                 std::cerr << task.candidateStrings[k * decodedSequencePitchInBytes + c];
        //             }
        //             std::cerr << " " << task.alignments[k].shift;
        //             std::cerr << "\n";
        //         }
        //     }
        // }
        
        const int consensusLength = msa.consensus.size();
        const int anchorLength = task.currentAnchorLength;
        const int mateLength = task.mateLength;
        const int currentExtensionLength = task.extendedSequenceLength;
        const int accumExtensionsLength = currentExtensionLength - anchorLength;

        //can extend by at most maxextensionPerStep bps
        int extendBy = std::min(
            consensusLength - anchorLength, 
            std::max(0, maxextensionPerStep)
        );
        //cannot extend over fragment 
        const int maxExtendBy_forFragmentSize = programOptions.maxFragmentSize - currentExtensionLength;
        const int maxExtendBy_toReachMate = (programOptions.maxFragmentSize - mateLength) - accumExtensionsLength;
        extendBy = std::min(extendBy, std::min(maxExtendBy_forFragmentSize, maxExtendBy_toReachMate));

        if(maxextensionPerStep <= 0){

            auto iter = std::find_if(
                msa.coverage.begin() + anchorLength,
                msa.coverage.end(),
                [&](int cov){
                    return cov < minCoverageForExtension;
                }
            );

            extendBy = std::distance(msa.coverage.begin() + anchorLength, iter);
            extendBy = std::min(extendBy, std::min(maxExtendBy_forFragmentSize, maxExtendBy_toReachMate));
        }

        auto makeAnchorForNextIteration = [&](){            
            if(extendBy == 0){
                task.abortReason = AbortReason::MsaNotExtended;
                // if(task.pairId == 87680 / 2 && task.id == 1){
                //     std::cout << "makeAnchorForNextIteration abort\n";
                // }
            }else{
                std::copy(
                    msa.consensus.begin() + extendBy, 
                    msa.consensus.begin() + extendBy + anchorLength, 
                    task.extendedSequence.begin() + currentExtensionLength - anchorLength + extendBy
                );
                std::transform(
                    msa.support.begin() + extendBy, 
                    msa.support.begin() + extendBy + anchorLength, 
                    task.qualityOfExtendedSequence.begin() + currentExtensionLength - anchorLength + extendBy,
                    [](const float f){
                        return getQualityChar(f);
                    }
                );
                task.extendedSequenceLength = currentExtensionLength + extendBy;
                // if(task.pairId == 87680 / 2 && task.id == 1){
                //     std::cout << "makeAnchorForNextIteration extendBy " << extendBy << "\n";
                // }
            }
        };

        constexpr int requiredOverlapMate = 70; //TODO relative overlap 
        constexpr float maxRelativeMismatchesInOverlap = 0.06f;
        constexpr int maxAbsoluteMismatchesInOverlap = 10;

        const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);


        if(task.pairedEnd && accumExtensionsLength + consensusLength - requiredOverlapMate + mateLength >= programOptions.minFragmentSize){
            //check if mate can be overlapped with consensus 
            //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [minFragmentSize, maxFragmentSize]

            const int firstStartpos = std::max(0, programOptions.minFragmentSize - accumExtensionsLength - mateLength);
            const int lastStartposExcl = std::min(
                std::max(0, programOptions.maxFragmentSize - accumExtensionsLength - mateLength) + 1,
                consensusLength - requiredOverlapMate
            );

            int bestOverlapMismatches = std::numeric_limits<int>::max();
            int bestOverlapStartpos = -1;

            for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                //compute metrics of overlap
                    
                const int ham = cpu::hammingDistanceOverlap(
                    msa.consensus.begin() + startpos, msa.consensus.end(), 
                    task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                );

                if(bestOverlapMismatches > ham){
                    bestOverlapMismatches = ham;
                    bestOverlapStartpos = startpos;
                }

                if(bestOverlapMismatches == 0){
                    break;
                }
            }
            
            if(bestOverlapMismatches <= maxNumMismatches){
                const int mateStartposInConsensus = bestOverlapStartpos;
                const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - task.currentAnchorLength);

                if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                    //bridge the gap between current anchor and mate
                    auto it1 = std::copy(
                        msa.consensus.begin() + anchorLength, 
                        msa.consensus.begin() + anchorLength + missingPositionsBetweenAnchorEndAndMateBegin, 
                        task.extendedSequence.begin() + currentExtensionLength
                    );
                    auto it2 = std::transform(
                        msa.support.begin() + anchorLength, 
                        msa.support.begin() + anchorLength + missingPositionsBetweenAnchorEndAndMateBegin, 
                        task.qualityOfExtendedSequence.begin() + currentExtensionLength,
                        [](const float f){
                            return getQualityChar(f);
                        }
                    );

                    //copy mate
                    std::copy(
                        task.decodedMateRevC.begin(),
                        task.decodedMateRevC.begin() + task.mateLength,
                        it1
                    );
                    std::copy(
                        task.mateQualityScoresReversed.begin(),
                        task.mateQualityScoresReversed.begin() + task.mateLength,
                        it2
                    );
                    task.extendedSequenceLength = currentExtensionLength + missingPositionsBetweenAnchorEndAndMateBegin + mateLength;
                    task.mateHasBeenFound = true;
                    // if(task.pairId == 87680 / 2 && task.id == 1){
                    //     std::cout << "finished missingPositionsBetweenAnchorEndAndMateBegin " << missingPositionsBetweenAnchorEndAndMateBegin << "\n";
                    // }
                }else{
                    std::copy(
                        task.decodedMateRevC.begin(),
                        task.decodedMateRevC.begin() + task.mateLength,
                        task.extendedSequence.begin() + currentExtensionLength - anchorLength + mateStartposInConsensus
                    );
                    std::copy(
                        task.mateQualityScoresReversed.begin(),
                        task.mateQualityScoresReversed.begin() + task.mateLength,
                        task.qualityOfExtendedSequence.begin() + currentExtensionLength - anchorLength + mateStartposInConsensus
                    );

                    task.extendedSequenceLength = currentExtensionLength - anchorLength + mateStartposInConsensus + mateLength;
                    task.mateHasBeenFound = true;
                    // if(task.pairId == 87680 / 2 && task.id == 1){
                    //     std::cout << "finished missingPositionsBetweenAnchorEndAndMateBegin " << missingPositionsBetweenAnchorEndAndMateBegin << "\n";
                    // }
                }
            }else{
                return makeAnchorForNextIteration();
            }
        }else{
            return makeAnchorForNextIteration();
        }
    }

    void computeMSAsAndExtendTasks(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        const int numSequences = indicesOfActiveTasks.size();

        std::vector<int> offsets(numSequences);
        offsets[0] = 0;

        int totalNumberOfCandidates = 0;
        for(int i = 0; i < numSequences; i++){
            const auto& task = tasks[indicesOfActiveTasks[i]];
            totalNumberOfCandidates += task.candidateReadIds.size();
            if(i < numSequences - 1){
                offsets[i+1] = totalNumberOfCandidates;
            }
        }

        std::vector<read_number> readIds(totalNumberOfCandidates);

        for(int i = 0; i < numSequences; i++){
            const auto& task = tasks[indicesOfActiveTasks[i]];
            assert(task.numRemainingCandidates == int(task.candidateReadIds.size()));
            std::copy(task.candidateReadIds.begin(), task.candidateReadIds.end(), readIds.begin() + offsets[i]);
        }

        std::vector<char> candidateQualities(totalNumberOfCandidates * qualityPitchInBytes);

        if(programOptions.useQualityScores){

            activeReadStorage->gatherQualities(
                candidateQualities.data(),
                qualityPitchInBytes,
                readIds.data(),
                totalNumberOfCandidates
            );

            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    if(task.alignmentFlags[c] == AlignmentOrientation::ReverseComplement){
                        std::reverse(
                            candidateQualities.data() + (offsets[i] + c) * qualityPitchInBytes,
                            candidateQualities.data() + (offsets[i] + c) * qualityPitchInBytes + task.candidateSequenceLengths[c]
                        );
                    }
                }
            }

        }else{
            std::fill(candidateQualities.begin(), candidateQualities.end(), 'I');
        }

        for(const auto& indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];


            if(task.numRemainingCandidates > 0){
                // if(task.id == 1 && task.iteration == 14){
                //     std::cerr << "task.numRemainingCandidates = " << task.numRemainingCandidates << "\n";
                // }

                const auto msa = constructMSA(task, candidateQualities.data() + offsets[&indexOfActiveTask - indicesOfActiveTasks.data()] * qualityPitchInBytes);
                // if(task.pairId == 87680 / 2 && task.id == 1){
                //     std::cout << "iteration " << task.iteration << "\n";
                // }
                // if(task.pairId == 87680 / 2 && task.id == 1){
                //     std::cout << "numCandidates " << task.numRemainingCandidates << "\n";
                // }
                extendWithMsa(task, msa);

                if(task.abortReason == AbortReason::None){
                    if(!task.mateHasBeenFound){
                        // const int newAccumExtensionLength = task.accumExtensionLengths;
                        // const int extendBy = newAccumExtensionLength - oldAccumExtensionLength;

                        // CheckAmbiguousColumns columnChecker(msa);

                        // auto splitInfos = columnChecker.getSplitInfos(task.currentAnchorLength, task.currentAnchorLength + extendBy, 0.4f, 0.6f);
                        // int numSplits = columnChecker.getNumberOfSplits(
                        //     splitInfos, msa
                        // );
                        // //printf("id %d, iteration %d, numSplits %d\n", task.id, task.iteration, numSplits);
                        // task.goodscore += numSplits;
                        task.goodscore += 0;
                    }
                    // if(task.pairId == 87680 / 2 && task.id == 1){
                    //     for(int i = 0; i < task.extendedSequenceLength; i++){
                    //         std::cout << task.extendedSequence[i];
                    //     }
                    //     std::cout << "\n";
                    // }

                }
            }else{
                task.mateHasBeenFound = false;
                task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                //std::cerr << "did not extend task id " << task.id << " readid " << task.myReadId << " iteration " << task.iteration << " because no candidates.\n";
            }
            task.abort = task.abortReason != AbortReason::None;
        }
    }

    std::vector<ExtendResult> constructResults(const std::vector<Task>& tasks) const{
        assert(tasks.size() % 4 == 0);
        std::vector<ExtendResult> extendResults;
        extendResults.reserve(tasks.size());

        for(const auto& task : tasks){

            ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;
            extendResult.originalMateLength = task.mateLength;
            extendResult.read1begin = 0;
            extendResult.goodscore = task.goodscore;
            extendResult.mateHasBeenFound = task.mateHasBeenFound;

            extendResult.extendedRead.resize(task.extendedSequenceLength);
            std::copy(
                task.inputAnchor.begin(),
                task.inputAnchor.end(),
                extendResult.extendedRead.begin()
            );
            std::copy(
                task.extendedSequence.begin() + task.myLength, 
                task.extendedSequence.begin() + task.extendedSequenceLength,
                extendResult.extendedRead.begin() + task.myLength
            );
            extendResult.read1Quality = task.inputAnchorQualityScores;
            if(task.mateHasBeenFound){
                //replace mate positions by original mate
                std::copy(
                    task.decodedMateRevC.begin(),
                    task.decodedMateRevC.end(),
                    extendResult.extendedRead.begin() + task.extendedSequenceLength - task.decodedMateRevC.length()
                );

                extendResult.read2Quality = task.mateQualityScoresReversed;                
                extendResult.read2begin = task.extendedSequenceLength - task.decodedMateRevC.length();
            }else{
                extendResult.read2begin = -1;
            }



            //if(extendResult.readId1 == 316 || extendResult.readId2 == 316){
            // if(task.pairId == 87680 / 2){
            //     // if(foo == 0){
            //     //     foo = 1;

            //     //     for(int i = 0; i < 4; i++){
            //     //         std::cout << tasks[i].myReadId << " " << tasks[i].mateReadId << "\n";
            //     //     }
            //     // }
            //     std::cout << "taskid " << task.id << ", mateHasBeenFound " << extendResult.mateHasBeenFound << ", abort reason " << int(task.abortReason) 
            //         << ", read1begin " << extendResult.read1begin << ", read2begin " << extendResult.read2begin << ", goodscore " << extendResult.goodscore 
            //         << ", iteration " << task.iteration << "\n";
            //     std::cout << extendResult.extendedRead << "\n";
            //     std::cout << extendResult.qualityScores << "\n";
            // }
            

            // std::cout << "quality\n";
            // for(int i = 0; i < task.extendedSequenceLength; i++){
            //     std::cout << extendResult.qualityScores[i];
            // }
            // std::cout << "\n";

            extendResults.emplace_back(std::move(extendResult));
        }


        std::vector<ExtendResult> extendResultsCombined;
        
        if(programOptions.strictExtensionMode != 0){
            MakePairResultsStrictConfig makePairResultConfig;
            makePairResultConfig.allowSingleStrand = programOptions.strictExtensionMode == 1;
            makePairResultConfig.maxLengthDifferenceIfBothFoundMate = 0;
            makePairResultConfig.singleStrandMinOverlapWithOtherStrand = 0.5f;
            makePairResultConfig.singleStrandMinMatchRateWithOtherStrand = 0.95f;

            extendResultsCombined = combinePairedEndDirectionResults4_strict(
                extendResults,
                programOptions.minFragmentSize,
                programOptions.maxFragmentSize,
                makePairResultConfig
            );
        }else{
            extendResultsCombined = combinePairedEndDirectionResults4(
                extendResults,
                programOptions.minFragmentSize,
                programOptions.maxFragmentSize
            );
        }

        //DEBUG REMOVE
        // for(auto& extendResult : extendResultsCombined){
        //     extendResult.qualityScores.clear();             
        // }

        
        // std::cout << "combined quality\n";
        // for(int i = 0; i < extendResultsCombined[0].qualityScores.size(); i++){
        //     std::cout << extendResultsCombined[0].qualityScores[i];
        // }
        // std::cout << "\n";

        return extendResultsCombined;
    }

    std::vector<ExtendResult> combinePairedEndDirectionResults4(
        std::vector<ExtendResult>& pairedEndDirectionResults,
        int /*minFragmentSize*/,
        int /*maxFragmentSize*/
    ) const {
        auto idcomp = [](const auto& l, const auto& r){ return l.getReadPairId() < r.getReadPairId();};
        //auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        std::vector<ExtendResult>& combinedResults = pairedEndDirectionResults;

        bool isSorted = std::is_sorted(
            combinedResults.begin(), 
            combinedResults.end(),
            idcomp
        );

        if(!isSorted){
            throw std::runtime_error("Error not sorted");
        }

        const int numinputs = combinedResults.size();
        assert(numinputs % 4 == 0);

        auto dest = combinedResults.begin();

        //std::cerr << "first pass\n";

        const int reads = numinputs / 4;

        auto merge = [&](auto& l, auto& r){
            const int beginOfNewPositions = l.extendedRead.size();

            auto overlapstart = l.read2begin;
            l.extendedRead.resize(overlapstart + r.extendedRead.size());
            //l.qualityScores.resize(overlapstart + r.extendedRead.size());

            //assert(int(std::distance(r.qualityScores.begin() + r.originalLength, r.qualityScores.end())) <= int(l.qualityScores.size() - beginOfNewPositions));

            std::copy(r.extendedRead.begin() + r.originalLength, r.extendedRead.end(), l.extendedRead.begin() + beginOfNewPositions);
            //std::copy(r.qualityScores.begin() + r.originalLength, r.qualityScores.end(), l.qualityScores.begin() + beginOfNewPositions);
        };

        for(int i = 0; i < reads; i += 1){
            auto& r1 = combinedResults[4 * i + 0];
            auto& r2 = combinedResults[4 * i + 1];
            auto& r3 = combinedResults[4 * i + 2];
            auto& r4 = combinedResults[4 * i + 3];

            auto r1matefoundfunc = [&](){
                merge(r1,r2);
                r1.read2Quality = std::move(r2.read1Quality);

                //std::cout << "r1matefoundfunc. read2quality = " << r1.read2Quality  << "\n";

                if(int(r4.extendedRead.size()) > r4.originalLength){
                    //insert extensions of reverse complement of r4 at beginning of r1

                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(r4.extendedRead.data() + r4.originalLength, r4.extendedRead.size() - r4.originalLength);
                    //std::string r4revNewQualities(r4.qualityScores.data() + r4.originalLength, r4.qualityScores.size() - r4.originalLength);
                    //std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.extendedRead.insert(r1.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    //r1.qualityScores.insert(r1.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.read1begin += r4revcNewPositions.size();
                    r1.read2begin += r4revcNewPositions.size();
                }

                r1.mergedFromReadsWithoutMate = false;

                //avoid self move
                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                
                ++dest;
            };

            auto r3matefoundfunc = [&](){
                merge(r3,r4);

                r3.read2Quality = std::move(r4.read1Quality);

                int extlength = r3.extendedRead.size();


                SequenceHelpers::reverseComplementSequenceDecodedInplace(r3.extendedRead.data(), extlength);
                std::reverse(r3.read1Quality.begin(), r3.read1Quality.end());
                std::reverse(r3.read2Quality.begin(), r3.read2Quality.end());
                std::swap(r3.read1Quality, r3.read2Quality);

                //const int sizeOfGap = r3.read2begin - (r3.read1begin + r3.originalLength);
                const int sizeOfRightExtension = extlength - (r3.read2begin + r3.originalMateLength);

                int newread2begin = extlength - (r3.read1begin + r3.originalLength);
                int newread2length = r3.originalLength;
                int newread1begin = sizeOfRightExtension;
                int newread1length = r3.originalMateLength;

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                assert(newread1begin + newread1length <= extlength);
                assert(newread2begin + newread2length <= extlength);

                r3.read1begin = newread1begin;
                r3.read2begin = newread2begin;
                r3.originalLength = newread1length;
                r3.originalMateLength = newread2length;



                if(int(r2.extendedRead.size()) > r2.originalLength){
                    //insert extensions of r2 at end of r3
                    r3.extendedRead.insert(r3.extendedRead.end(), r2.extendedRead.begin() + r2.originalLength, r2.extendedRead.end());
                    //r3.qualityScores.insert(r3.qualityScores.end(), r2.qualityScores.begin() + r2.originalLength, r2.qualityScores.end());
                }
                
                r3.mergedFromReadsWithoutMate = false;

                if(&(*dest) != &r3){
                    *dest = std::move(r3);
                }
                ++dest;
            };

            auto discardExtensionFunc = [&](){
                r1.extendedRead.erase(r1.extendedRead.begin() + r1.originalLength, r1.extendedRead.end());
                r1.read2Quality.clear();
                r1.read2begin = -1;
                r1.mateHasBeenFound = false;
                r1.aborted = false;
                r1.mergedFromReadsWithoutMate = false;

                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                
                ++dest;
            };

            // std::cerr << r1 << "\n";
            // std::cerr << r2 << "\n";
            // std::cerr << r3 << "\n";
            // std::cerr << r4 << "\n";

            //std::cerr << r1.mateHasBeenFound << " " << r3.mateHasBeenFound << ", " << r1.goodscore << " " << r3.goodscore << "\n";


            if(r1.mateHasBeenFound && r3.mateHasBeenFound){
                if(r1.goodscore < r3.goodscore){
                    r1matefoundfunc();
                }else{
                    r3matefoundfunc();
                } 
            }else if(r1.mateHasBeenFound){
                r1matefoundfunc();
            }else if(r3.mateHasBeenFound){
                r3matefoundfunc();                
            }else{
                assert(int(r1.extendedRead.size()) >= r1.originalLength);
                #if 1
                discardExtensionFunc();
                #else

                //try to find an overlap between r1 and r3 to create an extended read with proper length which reaches the mate

                const int r1l = r1.extendedRead.size();
                const int r3l = r3.extendedRead.size();

                constexpr int minimumOverlap = 40;
                constexpr float maxRelativeErrorInOverlap = 0.05;

                bool didMergeDifferentStrands = false;

                //if the longest achievable pseudo read reaches the minimum required pseudo read length
                if(r1l + r3l - minimumOverlap >= minFragmentSize){
                    std::string r3revc = SequenceHelpers::reverseComplementSequenceDecoded(r3.extendedRead.data(), r3.extendedRead.size());
                    std::reverse(r3.qualityScores.begin(), r3.qualityScores.end());

                    MismatchRatioGlueDecider decider(minimumOverlap, maxRelativeErrorInOverlap);
                    //WeightedGapGluer gluer(r1.originalLength);
                    QualityWeightedGapGluer gluer(r1.originalLength, r3.originalLength);

                    std::vector<std::pair<std::string, std::string>> possibleResults;

                    const int minimumResultLength = std::max(r1.originalLength+1, minFragmentSize);
                    const int maximumResultLength = std::min(r1l + r3l - minimumOverlap, maxFragmentSize);

                    for(int resultLength = minimumResultLength; resultLength <= maximumResultLength; resultLength++){
                        auto decision = decider(
                            r1.extendedRead, 
                            r3revc, 
                            resultLength,
                            r1.qualityScores, 
                            r3.qualityScores
                        );

                        if(decision.has_value()){
                            possibleResults.emplace_back(gluer(*decision));
                            break;
                        }
                    }

                    if(possibleResults.size() > 0){

                        didMergeDifferentStrands = true;

                        auto& mergeresult = possibleResults.front();

                        r1.extendedRead = std::move(mergeresult.first);
                        r1.qualityScores = std::move(mergeresult.second);
                        r1.read2begin = r1.extendedRead.size() - r3.originalLength;
                        r1.originalMateLength = r3.originalLength;
                        r1.mateHasBeenFound = true;
                        r1.aborted = false;
                    }
                }
                

                if(didMergeDifferentStrands && int(r2.extendedRead.size()) > r2.originalLength){
                    //insert extensions of r2 at end of r3
                    r1.extendedRead.insert(r1.extendedRead.end(), r2.extendedRead.begin() + r2.originalLength, r2.extendedRead.end());
                    r1.qualityScores.insert(r1.qualityScores.end(), r2.qualityScores.begin() + r2.originalLength, r2.qualityScores.end());
                } 

                if(int(r4.extendedRead.size()) > r4.originalLength){
                    //insert extensions of reverse complement of r4 at beginning of r1

                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(r4.extendedRead.data() + r4.originalLength, r4.extendedRead.size() - r4.originalLength);
                    
                    assert(r4.originalLength > 0);
                    std::string r4revNewQualities(r4.qualityScores.data() + r4.originalLength, r4.qualityScores.size() - r4.originalLength);
                    std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.extendedRead.insert(r1.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    r1.qualityScores.insert(r1.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.read1begin += r4revcNewPositions.size();
                    if(r1.mateHasBeenFound){
                        r1.read2begin += r4revcNewPositions.size();
                    }
                }
                r1.mergedFromReadsWithoutMate = didMergeDifferentStrands;

                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                ++dest;

                #endif


            }
        }


        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }

    std::vector<ExtendResult> combinePairedEndDirectionResults4_strict(
        std::vector<ExtendResult>& pairedEndDirectionResults,
        int /*minFragmentSize*/,
        int /*maxFragmentSize*/,
        MakePairResultsStrictConfig config
    ) const {
        auto idcomp = [](const auto& l, const auto& r){ return l.getReadPairId() < r.getReadPairId();};
        //auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        std::vector<ExtendResult>& combinedResults = pairedEndDirectionResults;

        bool isSorted = std::is_sorted(
            combinedResults.begin(), 
            combinedResults.end(),
            idcomp
        );

        if(!isSorted){
            throw std::runtime_error("Error not sorted");
        }

        const int numinputs = combinedResults.size();
        assert(numinputs % 4 == 0);

        auto dest = combinedResults.begin();

        //std::cerr << "first pass\n";

        const int reads = numinputs / 4;

        auto merge = [&](auto& l, auto& r){
            const int beginOfNewPositions = l.extendedRead.size();

            auto overlapstart = l.read2begin;
            l.extendedRead.resize(overlapstart + r.extendedRead.size());
            //l.qualityScores.resize(overlapstart + r.extendedRead.size());

            //assert(int(std::distance(r.qualityScores.begin() + r.originalLength, r.qualityScores.end())) <= int(l.qualityScores.size() - beginOfNewPositions));

            std::copy(r.extendedRead.begin() + r.originalLength, r.extendedRead.end(), l.extendedRead.begin() + beginOfNewPositions);
            //std::copy(r.qualityScores.begin() + r.originalLength, r.qualityScores.end(), l.qualityScores.begin() + beginOfNewPositions);
        };

        for(int i = 0; i < reads; i += 1){
            auto& r1 = combinedResults[4 * i + 0];
            auto& r2 = combinedResults[4 * i + 1];
            auto& r3 = combinedResults[4 * i + 2];
            auto& r4 = combinedResults[4 * i + 3];

            auto r1matefoundfunc = [&](){
                merge(r1,r2);
                r1.read2Quality = std::move(r2.read1Quality);

                if(int(r4.extendedRead.size()) > r4.originalLength){
                    //insert extensions of reverse complement of r4 at beginning of r1

                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(r4.extendedRead.data() + r4.originalLength, r4.extendedRead.size() - r4.originalLength);
                    //std::string r4revNewQualities(r4.qualityScores.data() + r4.originalLength, r4.qualityScores.size() - r4.originalLength);
                    //std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.extendedRead.insert(r1.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    //r1.qualityScores.insert(r1.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.read1begin += r4revcNewPositions.size();
                    r1.read2begin += r4revcNewPositions.size();
                }

                r1.mergedFromReadsWithoutMate = false;

                //avoid self move
                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                
                ++dest;
            };

            auto r3matefoundfunc = [&](){
                merge(r3,r4);
                r3.read2Quality = std::move(r4.read1Quality);
                int extlength = r3.extendedRead.size();


                SequenceHelpers::reverseComplementSequenceDecodedInplace(r3.extendedRead.data(), extlength);
                //std::reverse(r3.qualityScores.begin(), r3.qualityScores.end());

                //const int sizeOfGap = r3.read2begin - (r3.read1begin + r3.originalLength);
                const int sizeOfRightExtension = extlength - (r3.read2begin + r3.originalMateLength);

                int newread2begin = extlength - (r3.read1begin + r3.originalLength);
                int newread2length = r3.originalLength;
                int newread1begin = sizeOfRightExtension;
                int newread1length = r3.originalMateLength;

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                assert(newread1begin + newread1length <= extlength);
                assert(newread2begin + newread2length <= extlength);

                r3.read1begin = newread1begin;
                r3.read2begin = newread2begin;
                r3.originalLength = newread1length;
                r3.originalMateLength = newread2length;

                if(int(r2.extendedRead.size()) > r2.originalLength){
                    //insert extensions of r2 at end of r3
                    r3.extendedRead.insert(r3.extendedRead.end(), r2.extendedRead.begin() + r2.originalLength, r2.extendedRead.end());
                    //r3.qualityScores.insert(r3.qualityScores.end(), r2.qualityScores.begin() + r2.originalLength, r2.qualityScores.end());
                }
                
                r3.mergedFromReadsWithoutMate = false;

                if(&(*dest) != &r3){
                    *dest = std::move(r3);
                }
                ++dest;
            };

            // std::cerr << r1 << "\n";
            // std::cerr << r2 << "\n";
            // std::cerr << r3 << "\n";
            // std::cerr << r4 << "\n";

            //std::cerr << r1.mateHasBeenFound << " " << r3.mateHasBeenFound << ", " << r1.goodscore << " " << r3.goodscore << "\n";

            auto discardExtensionFunc = [&](){
                r1.extendedRead.erase(r1.extendedRead.begin() + r1.originalLength, r1.extendedRead.end());
                //r1.qualityScores.erase(r1.qualityScores.begin() + r1.originalLength, r1.qualityScores.end());
                r1.read2begin = -1;
                r1.mateHasBeenFound = false;
                r1.aborted = false;
                r1.mergedFromReadsWithoutMate = false;

                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                
                ++dest;
            };

            if(r1.mateHasBeenFound && r3.mateHasBeenFound){
                const int lengthDifference = r1.extendedRead.size() > r3.extendedRead.size() ? 
                    r1.extendedRead.size() - r3.extendedRead.size() 
                    : r3.extendedRead.size() - r1.extendedRead.size();
                if(lengthDifference > config.maxLengthDifferenceIfBothFoundMate){
                    discardExtensionFunc();
                }else{
                    const int gapBegin = r1.originalLength;
                    const int gapEnd = r1.extendedRead.size() - r3.originalLength;
                    const int gapSize = std::max(0, gapEnd - gapBegin);


                    const int extendedPositionsOtherStrand = r3.extendedRead.size() - r3.originalLength;
                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);
                    int matchingPositionsInGap = 0;
                    if(extendedPositionsOtherStrand > 0){
                        //find hamming distance of overlap betweens filled gaps of both strands
                        const int begin_i2 = r3.originalLength;
                        const int end_i2 = begin_i2 + overlapsize;

                        for(int k = 0; k < overlapsize; k++){
                            const int pos_i0 = gapEnd - overlapsize + k;
                            const int pos_i2 = end_i2 - 1 - k;
                            const char c0 = r1.extendedRead[pos_i0];
                            const char c2 = SequenceHelpers::complementBaseDecoded(r3.extendedRead[pos_i2]);
                            matchingPositionsInGap += (c0 == c2);
                        }
                    }

                    if(overlapsize == matchingPositionsInGap){
                        r1matefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }
                }
            }else if(r1.mateHasBeenFound && !r3.mateHasBeenFound){
                if(!config.allowSingleStrand){
                    discardExtensionFunc();
                }else{
                    const int gapBegin = r1.originalLength;
                    const int gapEnd = r1.extendedRead.size() - r3.originalLength;
                    const int gapSize = std::max(0, gapEnd - gapBegin);

                    const int extendedPositionsOtherStrand = r3.extendedRead.size() - r3.originalLength;
                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);
                    int matchingPositions = 0;
                    if(extendedPositionsOtherStrand > 0){

                        //find hamming distance of overlap betweens filled gaps of both strands

                        const int begin_i2 = r3.originalLength;
                        const int end_i2 = begin_i2 + overlapsize;
                        for(int k = 0; k < overlapsize; k++){
                            const int pos_i0 = gapEnd - overlapsize + k;
                            const int pos_i2 = end_i2 - 1 - k;
                            const char c0 = r1.extendedRead[pos_i0];
                            const char c2 = SequenceHelpers::complementBaseDecoded(r3.extendedRead[pos_i2]);
                            matchingPositions += (c0 == c2);
                        }
                    }

                    if(overlapsize >= gapSize * config.singleStrandMinOverlapWithOtherStrand 
                            && float(matchingPositions) / float(overlapsize) >= config.singleStrandMinMatchRateWithOtherStrand){
                        r1matefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }
                }
            }else if(!r1.mateHasBeenFound && r3.mateHasBeenFound){
                if(!config.allowSingleStrand){
                    discardExtensionFunc();
                }else{
                    const int gapBegin = r3.originalLength;
                    const int gapEnd = r3.extendedRead.size() - r1.originalLength;
                    const int gapSize = std::max(0, gapEnd - gapBegin);

                    const int extendedPositionsOtherStrand = r1.extendedRead.size() - r1.originalLength;
                    const int overlapsize = std::min(gapSize, extendedPositionsOtherStrand);
                    int matchingPositions = 0;
                    if(extendedPositionsOtherStrand > 0){

                        //find hamming distance of overlap betweens filled gaps of both strands

                        const int begin_i0 = r1.originalLength;
                        const int end_i0 = begin_i0 + overlapsize;
                        for(int k = 0; k < overlapsize; k++){
                            const int pos_i2 = gapEnd - overlapsize + k;
                            const int pos_i0 = end_i0 - 1 - k;
                            const char c0 = SequenceHelpers::complementBaseDecoded(r1.extendedRead[pos_i0]);
                            const char c2 = r3.extendedRead[pos_i2];
                            matchingPositions += (c0 == c2);
                        }
                    }

                    if(overlapsize >= gapSize * config.singleStrandMinOverlapWithOtherStrand 
                            && float(matchingPositions) / float(overlapsize) >= config.singleStrandMinMatchRateWithOtherStrand){
                        r3matefoundfunc();
                    }else{
                        discardExtensionFunc();
                    }
                }
            }else{
                discardExtensionFunc();
            }
        }


        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }



    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }


    const CpuReadStorage* readStorage{};
    const CpuMinhasher* minhasher{};
    const cpu::QualityScoreConversion* qualityConversion{};

    const CpuReadStorage* activeReadStorage{};
    const CpuMinhasher* activeMinhasher{};

    int maxextensionPerStep{1};
    int minCoverageForExtension{1};
    int maximumSequenceLength{};
    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    ProgramOptions programOptions{};

    mutable MinhasherHandle minhashHandle;

    mutable helpers::CpuTimer hashTimer{"hashtimer"};
    mutable helpers::CpuTimer collectTimer{"gathertimer"};
    mutable helpers::CpuTimer alignmentTimer{"alignmenttimer"};
    mutable helpers::CpuTimer alignmentFilterTimer{"filtertimer"};
    mutable helpers::CpuTimer msaTimer{"msatimer"};

};


#ifdef DO_ONLY_REMOVE_MATE_IDS
#undef DO_ONLY_REMOVE_MATE_IDS
#endif

}

