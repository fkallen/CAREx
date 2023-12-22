#!/bin/bash

if [ $# -lt 4 ]
then
    echo "Usage: $0 genome sam_or_bam_file outputfastafile isMatePair"
    echo "Extract genome region of read pairs from paired-end simulated ART reads"
    echo "genome: fasta genome, sam_or_bam_file: reads.sam file output from ART, isMatePair: must be 1 for ART -mp simulations"
    exit 1
fi

genomefile=$1
bamsamfile=$2
outputfile=$3
isMatePair=$4

bedfile="tmp_bed"

mydir=$(dirname "$0")

if [ $isMatePair -eq 1 ] 
then
    time samtools view -@ 16 -O SAM -f 16 $bamsamfile | $mydir/makepairregionbed > $bedfile
else
    time samtools view -@ 16 -O SAM -f 32 $bamsamfile | $mydir/makepairregionbed > $bedfile
fi
time bedtools getfasta -fi $genomefile -bed $bedfile -fo $outputfile

#rm $bedfile
