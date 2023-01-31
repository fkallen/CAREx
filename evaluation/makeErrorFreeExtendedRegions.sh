#!/bin/bash

if [ $# -lt 3 ]
then
    echo "Usage: $0 genome sam_or_bam_file outputfastafile"
    echo "Extract genome region of read pairs from paired-end simulated ART reads"
    exit 1
fi

genomefile=$1
bamsamfile=$2
outputfile=$3

bedfile="tmp_bed"

mydir=$(dirname "$0")

time samtools view -@ 16 -O SAM -f 32 $bamsamfile | $mydir/makepairregionbed > $bedfile
time bedtools getfasta -fi $genomefile -bed $bedfile -fo $outputfile

#rm $bedfile
