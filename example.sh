

input_1=elegans30cov_300_5_1.fq.gz
input_2=elegans30cov_300_5_2.fq.gz

threads=64
memorylimit=32G
outputdir=.
output_extended=extended.fastq
output_remaining=remaining.fastq


if [ ! -e  $input_1 ]; then
    wget -O $input_1 https://zenodo.org/records/10378908/files/elegans30cov_300_5_1.fq.gz?download=1
fi

if [ ! -e  $input_2 ]; then
    wget -O $input_2 https://zenodo.org/records/10378908/files/elegans30cov_300_5_2.fq.gz?download=1
fi

make -j $threads cpu
./carex-cpu -i $input_1 -i $input_2 -c 30 -d $outputdir --minFragmentSize 280 --maxFragmentSize 320 -t $threads -p -q -m $memorylimit \
    --eo $output_extended --outputRemaining --ro $output_remaining

#make -j $threads gpu
#./carex-gpu -i $input_1 -i $input_2 -c 30 -d $outputdir --minFragmentSize 280 --maxFragmentSize 320 -t $threads -p -q -m $memorylimit -g 0 \
    --eo $output_extended --outputRemaining --ro $output_remaining



