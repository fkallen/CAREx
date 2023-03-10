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
        None
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
        int accumExtensionLengths = 0;
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
        std::string resultsequence;
        std::string resultQualityScores;
        std::string currentQualityScores;
        std::string mateQualityScoresReversed;
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
        std::vector<std::string> totalDecodedAnchors;
        std::vector<std::string> totalAnchorQualityScores;
        std::vector<int> totalAnchorBeginInExtendedRead;
        std::vector<read_number> allUsedCandidateReadIdPairs; //sorted
        //std::vector<read_number> allFullyUsedCandidateReadIdPairs;
        std::vector<char> candidateStrings;
        std::vector<int> candidateShifts;
        std::vector<float> candidateOverlapWeights;
        std::vector<bool> isPairedCandidate;
        std::vector<char> totalDecodedAnchorsFlat;
        std::vector<int> totalDecodedAnchorsLengths;
        std::vector<char> totalAnchorQualityScoresFlat;



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
                if(accumExtensionLengths != rhs.accumExtensionLengths) std::cerr << "accumExtensionLengths differs\n";
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
                if(resultsequence != rhs.resultsequence) std::cerr << "resultsequence differs\n";
                if(resultQualityScores != rhs.resultQualityScores) std::cerr << "resultQualityScores differs\n";
                if(currentQualityScores != rhs.currentQualityScores) std::cerr << "currentQualityScores differs\n";
                if(mateQualityScoresReversed != rhs.mateQualityScoresReversed) std::cerr << "mateQualityScoresReversed differs\n";                
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
                if(totalDecodedAnchors != rhs.totalDecodedAnchors) std::cerr << "totalDecodedAnchors differs\n";
                if(totalAnchorQualityScores != rhs.totalAnchorQualityScores) std::cerr << "totalAnchorQualityScores differs\n";                
                if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead) std::cerr << "totalAnchorBeginInExtendedRead differs\n";
                if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) std::cerr << "allUsedCandidateReadIdPairs differs\n";
                //if(allFullyUsedCandidateReadIdPairs != rhs.allFullyUsedCandidateReadIdPairs) std::cerr << "allFullyUsedCandidateReadIdPairs differs\n";                
                if(candidateStrings != rhs.candidateStrings) std::cerr << "candidateStrings differs\n";
                if(candidateShifts != rhs.candidateShifts) std::cerr << "candidateShifts differs\n";
                if(candidateOverlapWeights != rhs.candidateOverlapWeights) std::cerr << "candidateOverlapWeights differs\n";
                if(isPairedCandidate != rhs.isPairedCandidate) std::cerr << "isPairedCandidate differs\n";
                if(totalDecodedAnchorsFlat != rhs.totalDecodedAnchorsFlat) std::cerr << "totalDecodedAnchorsFlat differs\n";
                if(totalDecodedAnchorsLengths != rhs.totalDecodedAnchorsLengths) std::cerr << "totalDecodedAnchorsLengths differs\n";
                if(totalAnchorQualityScoresFlat != rhs.totalAnchorQualityScoresFlat) std::cerr << "totalAnchorQualityScoresFlat differs\n";
            #endif
            if(pairedEnd != rhs.pairedEnd) return false;
            if(abort != rhs.abort) return false;
            if(mateHasBeenFound != rhs.mateHasBeenFound) return false;
            if(mateRemovedFromCandidates != rhs.mateRemovedFromCandidates) return false;
            if(abortReason != rhs.abortReason) return false;
            if(id != rhs.id) return false;
            if(myLength != rhs.myLength) return false;
            if(currentAnchorLength != rhs.currentAnchorLength) return false;
            if(accumExtensionLengths != rhs.accumExtensionLengths) return false;
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
            if(resultsequence != rhs.resultsequence) return false;
            if(resultQualityScores != rhs.resultQualityScores) return false;
            if(currentQualityScores != rhs.currentQualityScores) return false;
            if(mateQualityScoresReversed != rhs.mateQualityScoresReversed) return false;
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
            if(totalDecodedAnchors != rhs.totalDecodedAnchors) return false;
            if(totalAnchorQualityScores != rhs.totalAnchorQualityScores) return false;            
            if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead) return false;
            if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) return false;
            //if(allFullyUsedCandidateReadIdPairs != rhs.allFullyUsedCandidateReadIdPairs) return false;            
            if(candidateStrings != rhs.candidateStrings) return false;
            if(candidateShifts != rhs.candidateShifts) return false;
            if(candidateOverlapWeights != rhs.candidateOverlapWeights) return false;
            if(isPairedCandidate != rhs.isPairedCandidate) return false;
            if(totalDecodedAnchorsFlat != rhs.totalDecodedAnchorsFlat) return false;
            if(totalDecodedAnchorsLengths != rhs.totalDecodedAnchorsLengths) return false;
            if(totalAnchorQualityScoresFlat != rhs.totalAnchorQualityScoresFlat) return false;

            return true;
        }

        bool operator!=(const Task& rhs) const noexcept{
            return !operator==(rhs);
        }
        

        bool isActive(int minFragmentSize, int maxFragmentSize) const noexcept{
            return (iteration < minFragmentSize 
                && accumExtensionLengths < maxFragmentSize - (mateLength)
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
            accumExtensionLengths = 0;
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
            clear(resultsequence);
            clear(resultQualityScores);
            clear(currentQualityScores);
            clear(mateQualityScoresReversed);
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
            clear(totalDecodedAnchors);
            clear(totalAnchorQualityScores);
            clear(totalAnchorBeginInExtendedRead);
            clear(allUsedCandidateReadIdPairs);
            //clear(allFullyUsedCandidateReadIdPairs);
            clear(candidateStrings);
            clear(candidateShifts);
            clear(candidateOverlapWeights);
            clear(isPairedCandidate);
            clear(totalDecodedAnchorsFlat);
            clear(totalDecodedAnchorsLengths);
            clear(totalAnchorQualityScoresFlat);
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
                task1.resultsequence = dec1_53;
                task1.totalDecodedAnchors.emplace_back(task1.resultsequence);
                task1.totalAnchorBeginInExtendedRead.emplace_back(0);

                task1.currentQualityScores.insert(task1.currentQualityScores.begin(), input.qualityScores1.begin(), input.qualityScores1.end());
                task1.resultQualityScores = task1.currentQualityScores;
                task1.totalAnchorQualityScores.emplace_back(task1.currentQualityScores);

                task1.mateQualityScoresReversed.insert(task1.mateQualityScoresReversed.begin(), input.qualityScores2.begin(), input.qualityScores2.end());
                std::reverse(task1.mateQualityScoresReversed.begin(), task1.mateQualityScoresReversed.end());

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
                task2.resultsequence = dec2_53;
                task2.totalDecodedAnchors.emplace_back(task2.resultsequence);
                task2.totalAnchorBeginInExtendedRead.emplace_back(0);

                task2.currentQualityScores.insert(task2.currentQualityScores.begin(), input.qualityScores2.begin(), input.qualityScores2.end());
                std::reverse(task2.currentQualityScores.begin(), task2.currentQualityScores.end());
                task2.resultQualityScores = task2.currentQualityScores;
                task2.totalAnchorQualityScores.emplace_back(task2.currentQualityScores);

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
                task3.resultsequence = dec2_35;
                task3.totalDecodedAnchors.emplace_back(task3.resultsequence);
                task3.totalAnchorBeginInExtendedRead.emplace_back(0);

                task3.currentQualityScores.insert(task3.currentQualityScores.begin(), input.qualityScores2.begin(), input.qualityScores2.end());
                task3.resultQualityScores = task3.currentQualityScores;
                task3.totalAnchorQualityScores.emplace_back(task3.currentQualityScores);

                task3.mateQualityScoresReversed.insert(task3.mateQualityScoresReversed.begin(), input.qualityScores1.begin(), input.qualityScores1.end());
                std::reverse(task3.mateQualityScoresReversed.begin(), task3.mateQualityScoresReversed.end());

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
                task4.resultsequence = dec1_35;
                task4.totalDecodedAnchors.emplace_back(task4.resultsequence);
                task4.totalAnchorBeginInExtendedRead.emplace_back(0);

                task4.currentQualityScores.insert(task4.currentQualityScores.begin(), input.qualityScores1.begin(), input.qualityScores1.end());
                std::reverse(task4.currentQualityScores.begin(), task4.currentQualityScores.end());
                task4.resultQualityScores = task4.currentQualityScores;
                task4.totalAnchorQualityScores.emplace_back(task4.currentQualityScores);

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



    __inline__
    std::vector<ExtendResult> combinePairedEndDirectionResults4(
        std::vector<ExtendResult>& pairedEndDirectionResults,
        int minFragmentSize,
        int maxFragmentSize
    ){
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
            l.qualityScores.resize(overlapstart + r.extendedRead.size());

            assert(int(std::distance(r.qualityScores.begin() + r.originalLength, r.qualityScores.end())) <= int(l.qualityScores.size() - beginOfNewPositions));

            std::copy(r.extendedRead.begin() + r.originalLength, r.extendedRead.end(), l.extendedRead.begin() + beginOfNewPositions);
            std::copy(r.qualityScores.begin() + r.originalLength, r.qualityScores.end(), l.qualityScores.begin() + beginOfNewPositions);
        };

        for(int i = 0; i < reads; i += 1){
            auto& r1 = combinedResults[4 * i + 0];
            auto& r2 = combinedResults[4 * i + 1];
            auto& r3 = combinedResults[4 * i + 2];
            auto& r4 = combinedResults[4 * i + 3];

            auto r1matefoundfunc = [&](){
                merge(r1,r2);

                if(int(r4.extendedRead.size()) > r4.originalLength){
                    //insert extensions of reverse complement of r4 at beginning of r1

                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(r4.extendedRead.data() + r4.originalLength, r4.extendedRead.size() - r4.originalLength);
                    std::string r4revNewQualities(r4.qualityScores.data() + r4.originalLength, r4.qualityScores.size() - r4.originalLength);
                    std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    r1.extendedRead.insert(r1.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    r1.qualityScores.insert(r1.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

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

                int extlength = r3.extendedRead.size();


                SequenceHelpers::reverseComplementSequenceDecodedInplace(r3.extendedRead.data(), extlength);
                std::reverse(r3.qualityScores.begin(), r3.qualityScores.end());

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
                    r3.qualityScores.insert(r3.qualityScores.end(), r2.qualityScores.begin() + r2.originalLength, r2.qualityScores.end());
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
                #if 0
                r1.extendedRead.erase(r1.extendedRead.begin() + r1.originalLength, r1.extendedRead.end());
                r1.mergedFromReadsWithoutMate = false;
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

                #endif

                r1.mergedFromReadsWithoutMate = didMergeDifferentStrands;

                if(&(*dest) != &r1){
                    *dest = std::move(r1);
                }
                ++dest;
            }
        }


        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
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