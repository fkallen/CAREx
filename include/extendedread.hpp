#ifndef CARE_EXTENDEDREAD_HPP
#define CARE_EXTENDEDREAD_HPP


#include <config.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <readlibraryio.hpp>
#include <options.hpp>
#include <bitcompressedstring.hpp>

#include <cstring>
#include <string>
#include <vector>

//#define CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED

namespace care{

    enum class ExtendedReadStatus : unsigned char{
        FoundMate = 1,
        MSANoExtension = 2,
        LengthAbort = 4,
        CandidateAbort = 8,
        Repeated = 16
    };

    // struct ExtendedReadStatus{
    //     static constexpr unsigned char FoundMate = 1;
    //     static constexpr unsigned char MSANoExtension = 2;
    //     static constexpr unsigned char LengthAbort = 4;
    //     static constexpr unsigned char CandidateAbort = 8;
    //     static constexpr unsigned char Repeated = 16;

    //     unsigned char status;
    // };

    struct EncodedExtendedRead{
        std::uint32_t encodedflags{}; //contains size of data in bytes, and mergedFromReadsWithoutMate
        read_number readId{};
        std::unique_ptr<std::uint8_t[]> data{};

        EncodedExtendedRead() = default;
        EncodedExtendedRead(const EncodedExtendedRead& rhs){
            auto bytes = rhs.getNumBytes();
            encodedflags = rhs.encodedflags;
            readId = rhs.readId;
            data = std::make_unique<std::uint8_t[]>(bytes);
            std::copy(rhs.data.get(), rhs.data.get() + bytes, data.get());
        }
        EncodedExtendedRead(EncodedExtendedRead&& rhs){
            *this = std::move(rhs);
        }

        EncodedExtendedRead& operator=(const EncodedExtendedRead& rhs){
            auto bytes = rhs.getNumBytes();
            encodedflags = rhs.encodedflags;
            readId = rhs.readId;
            data = std::make_unique<std::uint8_t[]>(bytes);
            std::copy(rhs.data.get(), rhs.data.get() + bytes, data.get());

            return *this;
        }

        EncodedExtendedRead& operator=(EncodedExtendedRead&& rhs){
            encodedflags = std::exchange(rhs.encodedflags, 0);
            readId = std::exchange(rhs.readId, 0);
            data = std::move(rhs.data);

            return *this;
        }

        // EncodedExtendedRead(const EncodedExtendedRead& rhs){
        //     *this = rhs;
        // }

        // EncodedExtendedRead& operator=(const EncodedExtendedRead& rhs){
        //     encodedflags = rhs.encodedflags;
        //     readId = rhs.readId;

        //     const int numBytes = rhs.getNumBytes();
        //     data = std::make_unique<std::uint8_t[]>(numBytes);
        //     std::memcpy(data.get(), rhs.data.get(), numBytes);

        //     return *this;
        // }

        int getSerializedNumBytes() const noexcept{
            const int dataBytes = getNumBytes();
            return sizeof(read_number) + sizeof(std::uint32_t) + dataBytes;
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const int dataBytes = getNumBytes();

            const std::size_t availableBytes = std::distance(ptr, endPtr);
            const std::size_t requiredBytes = getSerializedNumBytes();
            if(requiredBytes <= availableBytes){
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
                ptr += sizeof(std::uint32_t);
                std::memcpy(ptr, data.get(), dataBytes);
                ptr += dataBytes;
                return ptr;
            }else{
                return nullptr;
            } 
        }

        const std::uint8_t* copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&encodedflags, ptr, sizeof(std::uint32_t));
            ptr += sizeof(read_number);

            const int numBytes = getNumBytes();
            data = std::make_unique<std::uint8_t[]>(numBytes);

            std::memcpy(data.get(), ptr, numBytes);
            ptr += numBytes;

            return ptr;
        }


        int getNumBytes() const{
            constexpr std::uint32_t mask = (std::uint32_t(1) << 31)-1;
            return (encodedflags & mask);
        }

        static read_number parseReadId(const std::uint8_t* ptr){
            read_number id;
            std::memcpy(&id, ptr, sizeof(read_number));
            return id;
        }

        read_number getReadId() const noexcept{
            return readId;
        }
    };



    struct ExtendedRead{
    public:
        bool mergedFromReadsWithoutMate = false;
        ExtendedReadStatus status{};
        read_number readId{};
        int read1begin = 0;
        int read1end = 0;
        int read2begin = 0;
        int read2end = 0;
    private:
        std::string extendedSequence_raw{};
        std::string qualityScores_read1_raw{};
        std::string qualityScores_read2_raw{};

    public:

        ExtendedRead() = default;

        bool operator==(const ExtendedRead& rhs) const noexcept{
            if(mergedFromReadsWithoutMate != rhs.mergedFromReadsWithoutMate) return false;
            if(status != rhs.status) return false;
            if(readId != rhs.readId) return false;
            if(read1begin != rhs.read1begin) return false;
            if(read1end != rhs.read1end) return false;
            if(read2begin != rhs.read2begin) return false;
            if(read2end != rhs.read2end) return false;
            if(getSequence() != rhs.getSequence()) return false;
            if(qualityScores_read1_raw != rhs.qualityScores_read1_raw) return false;
            if(qualityScores_read2_raw != rhs.qualityScores_read2_raw) return false;
            return true;
        }

        bool operator!=(const ExtendedRead& rhs) const noexcept{
            return !(operator==(rhs));
        }

        int getSerializedNumBytes() const noexcept{
            return sizeof(bool) // mergedFromReadsWithoutMate
                + sizeof(ExtendedReadStatus) //status
                + sizeof(read_number) //readid
                + sizeof(int) * 4  //original ranges
                + sizeof(int) + getSequence().length() //sequence
                + sizeof(int) + qualityScores_read1_raw.length() // quality scores
                + sizeof(int) + qualityScores_read2_raw.length(); // quality scores
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const std::size_t requiredBytes = getSerializedNumBytes();                

            const std::size_t availableBytes = std::distance(ptr, endPtr);

            if(requiredBytes <= availableBytes){                
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &mergedFromReadsWithoutMate, sizeof(bool));
                ptr += sizeof(bool);
                std::memcpy(ptr, &status, sizeof(ExtendedReadStatus));
                ptr += sizeof(ExtendedReadStatus);

                std::memcpy(ptr, &read1begin, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read1end, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read2begin, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read2end, sizeof(int));
                ptr += sizeof(int);

                int length = getSequence().length();
                std::memcpy(ptr, &length, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, getSequence().data(), sizeof(char) * length);
                ptr += sizeof(char) * length;

                length = qualityScores_read1_raw.length();
                std::memcpy(ptr, &length, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, qualityScores_read1_raw.data(), sizeof(char) * length);
                ptr += sizeof(char) * length;

                length = qualityScores_read2_raw.length();
                std::memcpy(ptr, &length, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, qualityScores_read2_raw.data(), sizeof(char) * length);
                ptr += sizeof(char) * length;

                return ptr;
            }else{
                return nullptr;
            }        
        }

        const std::uint8_t* copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&mergedFromReadsWithoutMate, ptr, sizeof(bool));
            ptr += sizeof(bool);
            std::memcpy(&status, ptr, sizeof(ExtendedReadStatus));
            ptr += sizeof(ExtendedReadStatus);

            std::memcpy(&read1begin, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read1end, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read2begin, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read2end, ptr, sizeof(int));
            ptr += sizeof(int);

            int length = 0;
            std::memcpy(&length, ptr, sizeof(int));
            ptr += sizeof(int);
            extendedSequence_raw.resize(length);
            std::memcpy(&extendedSequence_raw[0], ptr, sizeof(char) * length);
            ptr += sizeof(char) * length;

            std::memcpy(&length, ptr, sizeof(int));
            ptr += sizeof(int);
            qualityScores_read1_raw.resize(length);
            std::memcpy(&qualityScores_read1_raw[0], ptr, sizeof(char) * length);
            ptr += sizeof(char) * length;

            std::memcpy(&length, ptr, sizeof(int));
            ptr += sizeof(int);
            qualityScores_read2_raw.resize(length);
            std::memcpy(&qualityScores_read2_raw[0], ptr, sizeof(char) * length);
            ptr += sizeof(char) * length;

            return ptr;
        }

        //from serialized object beginning at ptr, return the read id of this object
        static read_number parseReadId(const std::uint8_t* ptr){
            read_number id;
            std::memcpy(&id, ptr, sizeof(read_number));
            return id;
        }

        read_number getReadId() const noexcept{
            return readId;
        }

        

        void encodeInto(EncodedExtendedRead& target) const{

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(getSequence().size());
            std::size_t requiredBytes = 0;
            requiredBytes += sizeof(ExtendedReadStatus); // status
            requiredBytes += sizeof(int); // read1begin
            requiredBytes += sizeof(int); // read1end
            requiredBytes += sizeof(int); // read2begin
            requiredBytes += sizeof(int); // read2end
            requiredBytes += sizeof(int); // seq length
            requiredBytes += sizeof(int); // qual1 length
            requiredBytes += sizeof(int); // qual2 length
            requiredBytes += sizeof(unsigned int) * numEncodedSequenceInts; // enc seq
            requiredBytes += sizeof(char) * qualityScores_read1_raw.size(); //qual1
            requiredBytes += sizeof(char) * qualityScores_read2_raw.size(); //qual2

            assert(requiredBytes < (1u << 31)); // 1 bit reserved for flag

            if(int(requiredBytes) > target.getNumBytes()){
                target.data = std::make_unique<std::uint8_t[]>(requiredBytes);
            }else{
                ; //reuse buffer
            }

            target.readId = readId;
            target.encodedflags = (std::uint32_t(mergedFromReadsWithoutMate) << 31);
            target.encodedflags |= std::uint32_t(requiredBytes);

            //fill buffer

            std::uint8_t* ptr = target.data.get();
            auto saveint = [&](int value){
                std::memcpy(ptr, &value, sizeof(int)); ptr += sizeof(int);
            };

            std::memcpy(ptr, &status, sizeof(ExtendedReadStatus)); ptr += sizeof(ExtendedReadStatus); 

            saveint(read1begin);
            saveint(read1end);
            saveint(read2begin);
            saveint(read2end);
            saveint(getSequence().size());
            saveint(qualityScores_read1_raw.size());
            saveint(qualityScores_read2_raw.size());

            SequenceHelpers::encodeSequence2Bit(
                reinterpret_cast<unsigned int*>(ptr), 
                getSequence().data(), 
                getSequence().size()
            );
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;

            ptr = std::copy(qualityScores_read1_raw.begin(), qualityScores_read1_raw.end(), ptr);
            ptr = std::copy(qualityScores_read2_raw.begin(), qualityScores_read2_raw.end(), ptr);

            assert(target.data.get() + requiredBytes == ptr);
        }
    
        EncodedExtendedRead encode() const{
            EncodedExtendedRead result;
            encodeInto(result);
            return result;
        }
    
        void decode(const EncodedExtendedRead& rhs){
            mergedFromReadsWithoutMate = bool(rhs.encodedflags >> 31);
            readId = rhs.readId;

            const std::uint8_t* ptr = rhs.data.get();
            auto loadint = [&](int& value){
                std::memcpy(&value, ptr, sizeof(int)); ptr += sizeof(int);
            };

            std::memcpy(&status, ptr, sizeof(ExtendedReadStatus)); ptr += sizeof(ExtendedReadStatus); 

            loadint(read1begin);
            loadint(read1end);
            loadint(read2begin);
            loadint(read2end);
            int seqlen;
            int qual1len;
            int qual2len;
            loadint(seqlen);
            loadint(qual1len);
            loadint(qual2len);

            extendedSequence_raw.resize(seqlen);

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(seqlen);

            SequenceHelpers::decode2BitSequence(extendedSequence_raw.data(), reinterpret_cast<const unsigned int*>(ptr), seqlen);
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;

            qualityScores_read1_raw.resize(qual1len);
            std::copy(ptr, ptr + qual1len, qualityScores_read1_raw.begin());
            ptr += sizeof(char) * qual1len;

            qualityScores_read2_raw.resize(qual2len);
            std::copy(ptr, ptr + qual2len, qualityScores_read2_raw.begin());
            ptr += sizeof(char) * qual2len;

            assert(rhs.data.get() + rhs.getNumBytes() == ptr);
        }

        void removeOutwardExtension(){
            const int newlength = (read2end == -1) ? extendedSequence_raw.size() : (read2end - read1begin);

            extendedSequence_raw.erase(extendedSequence_raw.begin(), extendedSequence_raw.begin() + read1begin);
            extendedSequence_raw.erase(extendedSequence_raw.begin() + newlength, extendedSequence_raw.end());

            const int curRead1begin = read1begin;
            read1begin -= curRead1begin;
            read1end -= curRead1begin;
            if(read2begin != -1){
                read2begin -= curRead1begin;
                read2end -= curRead1begin;

                assert(read2end - read1begin == newlength);
            }
        }

        void setSequence(std::string newseq){
            extendedSequence_raw = std::move(newseq);
        }

        void setRead1Quality(std::string quality){
            qualityScores_read1_raw = std::move(quality);
        }

        void setRead2Quality(std::string quality){
            qualityScores_read2_raw = std::move(quality);
        }

        std::string_view getSequence() const noexcept{
            return extendedSequence_raw;
        }

        std::string_view getRead1Quality() const noexcept{
            return qualityScores_read1_raw;
        }

        std::string_view getRead2Quality() const noexcept{
            return qualityScores_read2_raw;
        }
    
    };

}



#ifdef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
#undef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
#endif




#endif