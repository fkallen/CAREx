#ifndef CARE_READEXTENDER_COMMON_HPP
#define CARE_READEXTENDER_COMMON_HPP

#include <stringglueing.hpp>
#include <hostdevicefunctions.cuh>
#include <extendedread.hpp>

#include <cassert>

namespace care{
namespace extension{

    enum class ExtensionDirection {LR, RL};


    enum class AbortReason{
        MsaNotExtended, 
        NoPairedCandidates, 
        NoPairedCandidatesAfterAlignment, 
        PairedAnchorFinished,
        OtherStrandFoundMate,
        AmbiguousMatePositionInPseudoread,
        None
    };
    
    struct MakePairResultsStrictConfig{
        bool allowSingleStrand = false;
        int maxLengthDifferenceIfBothFoundMate = 0;
        float singleStrandMinOverlapWithOtherStrand = 0.5f;
        float singleStrandMinMatchRateWithOtherStrand = 0.95f;
    };

    struct ExtendInput{
        int readLength1{};
        int readLength2{};
        read_number readId1{};
        read_number readId2{};
        std::vector<unsigned int> encodedRead1{};
        std::vector<unsigned int> encodedRead2{};
        std::vector<char> qualityScores1{};
        std::vector<char> qualityScores2{};
    };

    struct ExtendResult{
        bool mateHasBeenFound = false;
        bool mergedFromReadsWithoutMate = false;
        bool aborted = false;
        int numIterations = 0;
        int originalLength = 0;
        int originalMateLength = 0;
        int read1begin = 0;
        int read2begin = 0;

        float goodscore = 0.0f;

        ExtensionDirection direction = ExtensionDirection::LR;
        AbortReason abortReason = AbortReason::None;

        read_number readId1{}; //same as input ids
        read_number readId2{}; //same as input ids

        std::string extendedRead{};
        std::string qualityScores{};
        

        read_number getReadPairId() const noexcept{
            return readId1 / 2;
        }

        bool operator==(const ExtendResult& rhs) const noexcept{
            if(mateHasBeenFound != rhs.mateHasBeenFound){ std::cerr << "error mateHasBeenFound\n"; return false;}
            if(mergedFromReadsWithoutMate != rhs.mergedFromReadsWithoutMate){ std::cerr << "error mergedFromReadsWithoutMate\n"; return false;}
            if(aborted != rhs.aborted){ std::cerr << "error aborted\n"; return false;}
            if(numIterations != rhs.numIterations){ std::cerr << "error numIterations\n"; return false;}
            if(originalLength != rhs.originalLength){ std::cerr << "error originalLength\n"; return false;}
            if(originalMateLength != rhs.originalMateLength){ std::cerr << "error originalMateLength\n"; return false;}
            if(read1begin != rhs.read1begin){ std::cerr << "error read1begin\n"; return false;}
            if(read2begin != rhs.read2begin){ std::cerr << "error read2begin\n"; return false;}
            if(!feq(goodscore,rhs.goodscore)){ std::cerr << "error goodscore\n"; return false;}
            if(direction != rhs.direction){ std::cerr << "error direction\n"; return false;}
            if(abortReason != rhs.abortReason){ std::cerr << "error abortReason\n"; return false;}
            if(readId1 != rhs.readId1){ std::cerr << "error readId1\n"; return false;}
            if(readId2 != rhs.readId2){ std::cerr << "error readId2\n"; return false;}
            if(extendedRead != rhs.extendedRead){
                std::cerr << "error extendedRead " << readId1 << "\n";
                std::cerr << extendedRead << "\n";
                std::cerr << rhs.extendedRead << "\n";
                 return false;
            } 
            if(qualityScores != rhs.qualityScores){
                std::cerr << "error qualityScores " << readId1 << "\n";
                std::cerr << qualityScores << "\n";
                std::cerr << rhs.qualityScores << "\n";
                 return false;
            }
            return true;

            // auto tup = [](const auto& res){
            //     return std::tie(
            //         res.mateHasBeenFound,
            //         res.mergedFromReadsWithoutMate,
            //         res.aborted,
            //         res.numIterations,
            //         res.originalLength,
            //         res.originalMateLength,
            //         res.read1begin,
            //         res.read2begin,
            //         res.goodscore,
            //         res.direction,
            //         res.abortReason,
            //         res.readId1,
            //         res.readId2,
            //         res.extendedRead,
            //         res.qualityScores
            //     );
            // };

            //return tup(*this) == tup(rhs);
        }
    };

    struct Task{
        bool dataIsAvailable = false;
        bool pairedEnd = false;
        bool abort = false;
        bool mateHasBeenFound = false;
        bool mateRemovedFromCandidates = false;
        AbortReason abortReason = AbortReason::None;
        int id = 0;
        int myLength = 0;
        int currentAnchorLength = 0;
        int iteration = 0;
        int mateLength = 0;
        int numRemainingCandidates = 0;
        int splitDepth = 0;

        float goodscore = 0.0f;

        ExtensionDirection direction{};
        int pairId = 0;
        read_number myReadId = 0;
        read_number mateReadId = 0;
        read_number currentAnchorReadId = 0;
        std::string decodedMate;
        std::string decodedMateRevC;
        std::string mateQualityScoresReversed;
        std::string inputAnchor;
        std::string inputAnchorQualityScores;
        std::vector<read_number> candidateReadIds;
        std::vector<read_number>::iterator mateIdLocationIter{};
        std::vector<unsigned int> currentAnchor;
        std::vector<unsigned int> encodedMate;
        std::vector<int> candidateSequenceLengths;
        std::vector<unsigned int> candidateSequencesFwdData;
        std::vector<unsigned int> candidateSequencesRevcData;
        std::vector<unsigned int> candidateSequenceData;
        std::vector<care::cpu::SHDResult> alignments;
        std::vector<AlignmentOrientation> alignmentFlags;

        std::vector<read_number> allUsedCandidateReadIdPairs; //sorted
        //std::vector<read_number> allFullyUsedCandidateReadIdPairs;
        std::vector<char> candidateStrings;
        std::vector<int> candidateShifts;
        std::vector<float> candidateOverlapWeights;
        std::vector<bool> isPairedCandidate;

        std::vector<char> extendedSequence;
        std::vector<char> qualityOfExtendedSequence;
        int extendedSequenceLength;



        bool operator==(const Task& rhs) const noexcept{
            #if 1
                if(pairedEnd != rhs.pairedEnd) std::cerr << "pairedEnd differs\n";
                if(abort != rhs.abort) std::cerr << "abort differs\n";
                if(mateHasBeenFound != rhs.mateHasBeenFound) std::cerr << "mateHasBeenFound differs\n";
                if(mateRemovedFromCandidates != rhs.mateRemovedFromCandidates) std::cerr << "mateRemovedFromCandidates differs\n";
                if(abortReason != rhs.abortReason) std::cerr << "abortReason differs\n";
                if(id != rhs.id) std::cerr << "id differs\n";
                if(myLength != rhs.myLength) std::cerr << "myLength differs\n";
                if(currentAnchorLength != rhs.currentAnchorLength) std::cerr << "currentAnchorLength differs\n";
                if(iteration != rhs.iteration) std::cerr << "iteration differs\n";
                if(mateLength != rhs.mateLength) std::cerr << "mateLength differs\n";
                if(numRemainingCandidates != rhs.numRemainingCandidates) std::cerr << "numRemainingCandidates differs\n";
                if(splitDepth != rhs.splitDepth) std::cerr << "splitDepth differs\n";
                if(direction != rhs.direction) std::cerr << "direction differs\n";
                if(pairId != rhs.pairId)  std::cerr << "pairId differs\n";
                if(myReadId != rhs.myReadId) std::cerr << "myReadId differs\n";
                if(mateReadId != rhs.mateReadId) std::cerr << "mateReadId differs\n";
                if(currentAnchorReadId != rhs.currentAnchorReadId) std::cerr << "currentAnchorReadId differs\n";
                if(decodedMate != rhs.decodedMate) std::cerr << "decodedMate differs\n";
                if(decodedMateRevC != rhs.decodedMateRevC) std::cerr << "decodedMateRevC differs\n";
                if(mateQualityScoresReversed != rhs.mateQualityScoresReversed) std::cerr << "mateQualityScoresReversed differs\n";     
                if(inputAnchor != rhs.inputAnchor) std::cerr << "inputAnchor differs\n";     
                if(inputAnchorQualityScores != rhs.inputAnchorQualityScores) std::cerr << "inputAnchorQualityScores differs\n";           
                if(candidateReadIds != rhs.candidateReadIds) std::cerr << "candidateReadIds differs\n";
                if(mateIdLocationIter != rhs.mateIdLocationIter) std::cerr << "mateIdLocationIter differs\n";
                if(currentAnchor != rhs.currentAnchor) std::cerr << "currentAnchor differs\n";
                if(encodedMate != rhs.encodedMate) std::cerr << "encodedMate differs\n";
                if(candidateSequenceLengths != rhs.candidateSequenceLengths) std::cerr << "candidateSequenceLengths differs\n";
                if(candidateSequencesFwdData != rhs.candidateSequencesFwdData) std::cerr << "candidateSequencesFwdData differs\n";
                if(candidateSequencesRevcData != rhs.candidateSequencesRevcData) std::cerr << "candidateSequencesRevcData differs\n";
                if(candidateSequenceData != rhs.candidateSequenceData) std::cerr << "candidateSequenceData differs\n";
                if(alignments != rhs.alignments) std::cerr << "alignments differs\n";
                if(alignmentFlags != rhs.alignmentFlags) std::cerr << "alignmentFlags differs\n";
                if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) std::cerr << "allUsedCandidateReadIdPairs differs\n";
                //if(allFullyUsedCandidateReadIdPairs != rhs.allFullyUsedCandidateReadIdPairs) std::cerr << "allFullyUsedCandidateReadIdPairs differs\n";                
                if(candidateStrings != rhs.candidateStrings) std::cerr << "candidateStrings differs\n";
                if(candidateShifts != rhs.candidateShifts) std::cerr << "candidateShifts differs\n";
                if(candidateOverlapWeights != rhs.candidateOverlapWeights) std::cerr << "candidateOverlapWeights differs\n";
                if(isPairedCandidate != rhs.isPairedCandidate) std::cerr << "isPairedCandidate differs\n";
                if(extendedSequence != rhs.extendedSequence) std::cerr << "extendedSequence differs\n";
                if(qualityOfExtendedSequence != rhs.qualityOfExtendedSequence) std::cerr << "qualityOfExtendedSequence differs\n";
                if(extendedSequenceLength != rhs.extendedSequenceLength) std::cerr << "extendedSequenceLength differs\n";
            #endif
            if(pairedEnd != rhs.pairedEnd) return false;
            if(abort != rhs.abort) return false;
            if(mateHasBeenFound != rhs.mateHasBeenFound) return false;
            if(mateRemovedFromCandidates != rhs.mateRemovedFromCandidates) return false;
            if(abortReason != rhs.abortReason) return false;
            if(id != rhs.id) return false;
            if(myLength != rhs.myLength) return false;
            if(currentAnchorLength != rhs.currentAnchorLength) return false;
            if(iteration != rhs.iteration) return false;
            if(mateLength != rhs.mateLength) return false;
            if(numRemainingCandidates != rhs.numRemainingCandidates) return false;
            if(splitDepth != rhs.splitDepth) return false;
            if(direction != rhs.direction) return false;
            if(pairId != rhs.pairId) return false;
            if(myReadId != rhs.myReadId) return false;
            if(mateReadId != rhs.mateReadId) return false;
            if(currentAnchorReadId != rhs.currentAnchorReadId) return false;
            if(decodedMate != rhs.decodedMate) return false;
            if(decodedMateRevC != rhs.decodedMateRevC) return false;
            if(mateQualityScoresReversed != rhs.mateQualityScoresReversed) return false;
            if(inputAnchor != rhs.inputAnchor) return false;
            if(inputAnchorQualityScores != rhs.inputAnchorQualityScores) return false;   
            if(candidateReadIds != rhs.candidateReadIds) return false;
            if(mateIdLocationIter != rhs.mateIdLocationIter) return false;
            if(currentAnchor != rhs.currentAnchor) return false;
            if(encodedMate != rhs.encodedMate) return false;
            if(candidateSequenceLengths != rhs.candidateSequenceLengths) return false;
            if(candidateSequencesFwdData != rhs.candidateSequencesFwdData) return false;
            if(candidateSequencesRevcData != rhs.candidateSequencesRevcData) return false;
            if(candidateSequenceData != rhs.candidateSequenceData) return false;
            if(alignments != rhs.alignments) return false;
            if(alignmentFlags != rhs.alignmentFlags) return false;
            if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) return false;
            //if(allFullyUsedCandidateReadIdPairs != rhs.allFullyUsedCandidateReadIdPairs) return false;            
            if(candidateStrings != rhs.candidateStrings) return false;
            if(candidateShifts != rhs.candidateShifts) return false;
            if(candidateOverlapWeights != rhs.candidateOverlapWeights) return false;
            if(isPairedCandidate != rhs.isPairedCandidate) return false;

            if(extendedSequence != rhs.extendedSequence) return false;
            if(qualityOfExtendedSequence != rhs.qualityOfExtendedSequence) return false;
            if(extendedSequenceLength != rhs.extendedSequenceLength) return false;

            return true;
        }

        bool operator!=(const Task& rhs) const noexcept{
            return !operator==(rhs);
        }
        

        bool isActive(int minFragmentSize, int maxFragmentSize) const noexcept{
            return (iteration < minFragmentSize 
                && extendedSequenceLength < maxFragmentSize
                && !abort 
                && !mateHasBeenFound);
        }

        void reset(){
            auto clear = [](auto& vec){vec.clear();};

            dataIsAvailable = false;
            pairedEnd = false;
            abort = false;
            mateHasBeenFound = false;
            mateRemovedFromCandidates = false;
            abortReason = AbortReason::None;
            id = 0;
            myLength = 0;
            currentAnchorLength = 0;
            iteration = 0;
            mateLength = 0;
            direction = ExtensionDirection::LR;
            myReadId = 0;
            mateReadId = 0;
            currentAnchorReadId = 0;
            numRemainingCandidates = 0;
            splitDepth = 0;

            clear(decodedMate);
            clear(decodedMateRevC);
            clear(mateQualityScoresReversed);
            clear(inputAnchor);
            clear(inputAnchorQualityScores);
            clear(candidateReadIds);
            mateIdLocationIter = candidateReadIds.end();
            clear(currentAnchor);
            clear(encodedMate);
            clear(candidateSequenceLengths);
            clear(candidateSequencesFwdData);
            clear(candidateSequencesRevcData);
            clear(candidateSequenceData);
            clear(alignments);
            clear(alignmentFlags);
            clear(allUsedCandidateReadIdPairs);
            //clear(allFullyUsedCandidateReadIdPairs);
            clear(candidateStrings);
            clear(candidateShifts);
            clear(candidateOverlapWeights);
            clear(isPairedCandidate);

            clear(extendedSequence);
            clear(qualityOfExtendedSequence);
            extendedSequenceLength = 0;
        }
    };


    __inline__
    std::string to_string(extension::AbortReason r){
        using ar = extension::AbortReason;

        switch(r){
            case ar::MsaNotExtended: return "MsaNotExtended";
            case ar::NoPairedCandidates: return "NoPairedCandidates";
            case ar::NoPairedCandidatesAfterAlignment: return "NoPairedCandidatesAfterAlignment";
            case ar::PairedAnchorFinished: return "PairedAnchorFinished";
            case ar::OtherStrandFoundMate: return "OtherStrandFoundMate";
            case ar::AmbiguousMatePositionInPseudoread: return "AmbiguousMatePositionInPseudoread";
            case ar::None:
            default: return "None";
        }
    }

    __inline__
    std::string to_string(extension::ExtensionDirection r){
        using ar = extension::ExtensionDirection;

        switch(r){
            case ar::LR: return "LR";
            case ar::RL: return "RL";
        }

        return "INVALID ENUM VALUE to_string(ExtensionDirection)";
    }

    __inline__
    std::ostream & operator<<(std::ostream &os, const ExtendResult& r){
        os << "ExtendResult{ "
            << "found: " << r.mateHasBeenFound
            << ", mergedFromReadsWithoutMate: " << r.mergedFromReadsWithoutMate
            << ", aborted: " << r.aborted
            << ", numIterations: " << r.numIterations
            << ", originalLength: " << r.originalLength
            << ", read1begin: " << r.read1begin
            << ", read2begin: " << r.read2begin
            << ", direction: " << to_string(r.direction)
            << ", abortReason: " << to_string(r.abortReason)
            << ", readId1: " << r.readId1
            << ", readId2: " << r.readId2
            << ", extendedRead: " << r.extendedRead
            << ", qualityScores: " << r.qualityScores
            << "}";
        return os;
    }


    template<class InputIter, class TaskOutIter>
    TaskOutIter makePairedEndTasksFromInput4(InputIter inputsBegin, InputIter inputsEnd, TaskOutIter outputBegin){
        TaskOutIter cur = outputBegin;

        std::for_each(
            inputsBegin, 
            inputsEnd,
            [&cur](auto&& input){
                /*
                    5-3 input.encodedRead1 --->
                    3-5                           <--- input.encodedRead2
                */

                std::vector<unsigned int> enc1_53 = std::move(input.encodedRead1);
                std::vector<unsigned int> enc2_35 = std::move(input.encodedRead2);

                std::vector<unsigned int> enc1_35(enc1_53);
                SequenceHelpers::reverseComplementSequenceInplace2Bit(enc1_35.data(), input.readLength1);
                std::vector<unsigned int> enc2_53(enc2_35);
                SequenceHelpers::reverseComplementSequenceInplace2Bit(enc2_53.data(), input.readLength2);

                std::string dec1_53 = SequenceHelpers::get2BitString(enc1_53.data(), input.readLength1);
                std::string dec1_35 = SequenceHelpers::get2BitString(enc1_35.data(), input.readLength1);
                std::string dec2_53 = SequenceHelpers::get2BitString(enc2_53.data(), input.readLength2);
                std::string dec2_35 = SequenceHelpers::get2BitString(enc2_35.data(), input.readLength2);

                constexpr std::size_t extendedSequencePitch = 2048;
                //task1, extend encodedRead1 to the right on 5-3 strand
                auto& task1 = *cur;
                task1.reset();

                task1.pairId = input.readId1 / 2;
                task1.pairedEnd = true;
                task1.direction = ExtensionDirection::LR;      
                task1.currentAnchor = enc1_53;
                task1.encodedMate = enc2_35;
                task1.currentAnchorLength = input.readLength1;
                task1.currentAnchorReadId = input.readId1;
                task1.myLength = input.readLength1;
                task1.myReadId = input.readId1;
                task1.mateLength = input.readLength2;
                task1.mateReadId = input.readId2;
                task1.decodedMate = dec2_35;
                task1.decodedMateRevC = dec2_53;

                task1.inputAnchor = dec1_53;
                task1.inputAnchorQualityScores = std::string{input.qualityScores1.begin(), input.qualityScores1.end()};
                task1.mateQualityScoresReversed.insert(task1.mateQualityScoresReversed.begin(), input.qualityScores2.begin(), input.qualityScores2.end());
                std::reverse(task1.mateQualityScoresReversed.begin(), task1.mateQualityScoresReversed.end());

                task1.extendedSequenceLength = task1.inputAnchor.size();
                task1.extendedSequence.resize(extendedSequencePitch);
                std::copy(task1.inputAnchor.begin(), task1.inputAnchor.end(), task1.extendedSequence.begin());

                task1.qualityOfExtendedSequence.resize(extendedSequencePitch);
                std::copy(task1.inputAnchorQualityScores.begin(), task1.inputAnchorQualityScores.end(), task1.qualityOfExtendedSequence.begin());


                ++cur;

                auto& task2 = *cur;
                task2.reset();

                task2.pairId = input.readId1 / 2;
                task2.pairedEnd = false;
                task2.direction = ExtensionDirection::LR;      
                task2.currentAnchor = enc2_53;
                //task2.encodedMate
                task2.currentAnchorLength = input.readLength2;
                task2.currentAnchorReadId = input.readId2;
                task2.myLength = input.readLength2;
                task2.myReadId = input.readId2;
                task2.mateLength = 0;
                task2.mateReadId = std::numeric_limits<read_number>::max();
                //task2.decodedMate
                //task2.decodedMateRevC

                task2.inputAnchor = dec2_53;
                task2.inputAnchorQualityScores = std::string{input.qualityScores2.begin(), input.qualityScores2.end()};
                std::reverse(task2.inputAnchorQualityScores.begin(), task2.inputAnchorQualityScores.end());


                task2.extendedSequenceLength = task2.inputAnchor.size();
                task2.extendedSequence.resize(extendedSequencePitch);
                std::copy(task2.inputAnchor.begin(), task2.inputAnchor.end(), task2.extendedSequence.begin());

                task2.qualityOfExtendedSequence.resize(extendedSequencePitch);
                std::copy(task2.inputAnchorQualityScores.begin(), task2.inputAnchorQualityScores.end(), task2.qualityOfExtendedSequence.begin());

                ++cur;

                auto& task3 = *cur;
                task3.reset();

                task3.pairId = input.readId1 / 2;
                task3.pairedEnd = true;
                task3.direction = ExtensionDirection::RL;      
                task3.currentAnchor = enc2_35;
                task3.encodedMate = enc1_53;
                task3.currentAnchorLength = input.readLength2;
                task3.currentAnchorReadId = input.readId2;
                task3.myLength = input.readLength2;
                task3.myReadId = input.readId2;
                task3.mateLength = input.readLength1;
                task3.mateReadId = input.readId1;
                task3.decodedMate = dec1_53;
                task3.decodedMateRevC = dec1_35;

                task3.inputAnchor = dec2_35;
                task3.inputAnchorQualityScores = std::string{input.qualityScores2.begin(), input.qualityScores2.end()};
                task3.mateQualityScoresReversed.insert(task3.mateQualityScoresReversed.begin(), input.qualityScores1.begin(), input.qualityScores1.end());
                std::reverse(task3.mateQualityScoresReversed.begin(), task3.mateQualityScoresReversed.end());

                task3.extendedSequenceLength = task3.inputAnchor.size();
                task3.extendedSequence.resize(extendedSequencePitch);
                std::copy(task3.inputAnchor.begin(), task3.inputAnchor.end(), task3.extendedSequence.begin());

                task3.qualityOfExtendedSequence.resize(extendedSequencePitch);
                std::copy(task3.inputAnchorQualityScores.begin(), task3.inputAnchorQualityScores.end(), task3.qualityOfExtendedSequence.begin());


                ++cur;

                auto& task4 = *cur;
                task4.reset();

                task4.pairId = input.readId1 / 2;
                task4.pairedEnd = false;
                task4.direction = ExtensionDirection::RL;      
                task4.currentAnchor = enc1_35;
                //task4.encodedMate
                task4.currentAnchorLength = input.readLength1;
                task4.currentAnchorReadId = input.readId1;
                task4.myLength = input.readLength1;
                task4.myReadId = input.readId1;
                task4.mateLength = 0;
                task4.mateReadId = std::numeric_limits<read_number>::max();
                //task4.decodedMate = dec1_53;
                //task4.decodedMateRevC = dec1_35;

                task4.inputAnchor = dec1_35;
                task4.inputAnchorQualityScores = std::string{input.qualityScores1.begin(), input.qualityScores1.end()};
                std::reverse(task4.inputAnchorQualityScores.begin(), task4.inputAnchorQualityScores.end());

                task4.extendedSequenceLength = task4.inputAnchor.size();
                task4.extendedSequence.resize(extendedSequencePitch);
                std::copy(task4.inputAnchor.begin(), task4.inputAnchor.end(), task4.extendedSequence.begin());

                task4.qualityOfExtendedSequence.resize(extendedSequencePitch);
                std::copy(task4.inputAnchorQualityScores.begin(), task4.inputAnchorQualityScores.end(), task4.qualityOfExtendedSequence.begin());

                ++cur;
            }
        );

        int i = 0;
        for(auto it = outputBegin; it != cur; ++it){
            it->id = i;
            i++;
        }

        return cur;
    }

    template<class InputIter>
    std::vector<Task> makePairedEndTasksFromInput4(InputIter inputsBegin, InputIter inputsEnd){
        auto num = std::distance(inputsBegin, inputsEnd);

        std::vector<Task> vec(num * 4);

        auto endIter = makePairedEndTasksFromInput4(inputsBegin, inputsEnd, vec.begin());
        if(endIter != vec.end())
            throw std::runtime_error("Error initializing batch");

        return vec;
    }


    __inline__
    void handleEarlyExitOfTasks4(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        constexpr bool disableOtherStrand = false;

        //std::cout << "Check early exit\n";
        for(int i = 0; i < int(indicesOfActiveTasks.size()); i++){ 
            const int indexOfActiveTask = indicesOfActiveTasks[i];
            const auto& task = tasks[indexOfActiveTask];
            const int whichtype = task.id % 4;

            assert(indexOfActiveTask % 4 == whichtype);

            //whichtype 0: LR, strand1 searching mate to the right.
            //whichtype 1: LR, strand1 just extend to the right.
            //whichtype 2: RL, strand2 searching mate to the right.
            //whichtype 3: RL, strand2 just extend to the right.

            //printf("i %d, whichtype %d, matefound %d, abort %d, candisableother %d\n", i, whichtype, task.mateHasBeenFound, task.abort, disableOtherStrand);

            if(whichtype == 0){
                assert(task.direction == extension::ExtensionDirection::LR);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){        
                    //disable LR partner task            
                    tasks[indexOfActiveTask + 1].abort = true;
                    tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                    //disable RL search task
                    if(disableOtherStrand){
                        tasks[indexOfActiveTask + 2].abort = true;
                        tasks[indexOfActiveTask + 2].abortReason = extension::AbortReason::OtherStrandFoundMate;
                    }
                }else if(task.abort){
                    //disable LR partner task  
                    tasks[indexOfActiveTask + 1].abort = true;
                    tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                }
            }else if(whichtype == 2){
                assert(task.direction == extension::ExtensionDirection::RL);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){
                    //disable RL partner task
                    tasks[indexOfActiveTask + 1].abort = true;
                    tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                    //disable LR search task
                    if(disableOtherStrand){
                        tasks[indexOfActiveTask - 2].abort = true;
                        tasks[indexOfActiveTask - 2].abortReason = extension::AbortReason::OtherStrandFoundMate;
                    }                    
                }else if(task.abort){
                    //disable RL partner task
                    tasks[indexOfActiveTask + 1].abort = true;
                    tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                }
            }
        }
    }





    struct ExtensionResultConversionOptions{
        bool computedAfterRepetition = false;
    };

    __inline__
    ExtendedRead makeExtendedReadFromExtensionResult(ExtendResult extensionOutput, const ExtensionResultConversionOptions& opt){

        ExtendedRead er;

        er.readId = extensionOutput.readId1;
        er.mergedFromReadsWithoutMate = extensionOutput.mergedFromReadsWithoutMate;
        er.setSequence(std::move(extensionOutput.extendedRead));
        er.setQuality(std::move(extensionOutput.qualityScores));
        er.read1begin = extensionOutput.read1begin;
        er.read1end = extensionOutput.read1begin + extensionOutput.originalLength;
        er.read2begin = extensionOutput.read2begin;
        if(er.read2begin != -1){
            er.read2end = extensionOutput.read2begin + extensionOutput.originalMateLength;
        }else{
            er.read2end = -1;
        }

        auto printerror = [&](){
            std::cerr << "unexpected error for read id " << er.readId << "\n";
            std::cerr << er.mergedFromReadsWithoutMate << ", " << er.read1begin << ", " << er.read1end << ", " << er.read2begin << ", " << er.read2end << "\n";
            std::cerr << er.getSequence() << "\n";
        };

        if(
            er.read1begin < 0
            || er.read1end > int(er.getSequence().size())
            || (
                (er.read2end != -1 && er.read2begin != -1) && (
                    er.read2begin < er.read1begin
                    || er.read2end > int(er.getSequence().size())
                )
            )
        ){
            printerror();
        }

        assert(er.read1begin >= 0);
        assert(er.read1end <= int(er.getSequence().size()));
        if(er.read2end != -1 && er.read2begin != -1){
            assert(er.read2begin >= 0);
            assert(er.read2begin >= er.read1begin);
            assert(er.read2end <= int(er.getSequence().size()));
        }

        if(extensionOutput.mateHasBeenFound){
            er.status = ExtendedReadStatus::FoundMate;
        }else{
            if(extensionOutput.aborted){
                if(extensionOutput.abortReason == extension::AbortReason::NoPairedCandidates
                        || extensionOutput.abortReason == extension::AbortReason::NoPairedCandidatesAfterAlignment){

                    er.status = ExtendedReadStatus::CandidateAbort;
                }else if(extensionOutput.abortReason == extension::AbortReason::MsaNotExtended){
                    er.status = ExtendedReadStatus::MSANoExtension;
                }
            }else{
                er.status = ExtendedReadStatus::LengthAbort;
            }
        }

        if(opt.computedAfterRepetition){
            reinterpret_cast<unsigned char&>(er.status) |= static_cast<unsigned char>(ExtendedReadStatus::Repeated);
        }

        return er;
    }

    struct SplittedExtensionOutput{
        std::vector<ExtendedRead> extendedReads{}; //mate has been found
        std::vector<read_number> idsOfPartiallyExtendedReads; //extended a bit, but did not find mate
        std::vector<read_number> idsOfNotExtendedReads; //did not extend even a single base
    };

    __inline__
    SplittedExtensionOutput splitExtensionOutput(
        std::vector<extension::ExtendResult> extensionResults,
        bool isRepeatedIteration
    ){

        SplittedExtensionOutput returnvalue{};

        const std::size_t maxNumExtendedReads = extensionResults.size();
        returnvalue.extendedReads.reserve(maxNumExtendedReads);

        for(std::size_t i = 0; i < maxNumExtendedReads; i++){
            auto& extensionOutput = extensionResults[i];
            const int extendedReadLength = extensionOutput.extendedRead.size();

            if(extendedReadLength > extensionOutput.originalLength){
                if(extensionOutput.mateHasBeenFound){
                    extension::ExtensionResultConversionOptions opts;
                    opts.computedAfterRepetition = isRepeatedIteration;            
                    
                    returnvalue.extendedReads.push_back(extension::makeExtendedReadFromExtensionResult(std::move(extensionOutput), opts));
                }else{
                    returnvalue.idsOfPartiallyExtendedReads.push_back(extensionOutput.readId1);
                    returnvalue.idsOfPartiallyExtendedReads.push_back(extensionOutput.readId2);
                }
            }else{
                returnvalue.idsOfNotExtendedReads.push_back(extensionOutput.readId1);
                returnvalue.idsOfNotExtendedReads.push_back(extensionOutput.readId2);
            }
                            
        }

        return returnvalue;
    }
    



} //namespace extension
} //namespace care



#endif