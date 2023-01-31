#!/bin/bash

if [ $# -lt 6 ]
then
    echo "Usage: $0 konnectorinputprefix konnectoroutputprefix interleavedinputfile insertsize insertsizedev genome.fai"
    echo "Convert konnector output _pseudoreads.fa _reads_1.fq reads_2.fq into _total.fa _remaining.fa _extended.fa with mate information"
    exit 1
fi

inputprefix=$1
outputprefix=$2
interleavedinputfile=$3
insertsize=$4
insertsizedev=$5
genomefai=$6

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#flatten fasta reads into single lines, then sort
sed '$!N;s/\n/\t/' $inputprefix"_pseudoreads.fa" > tmpfile
sort -k1,1 -k2,2rn -t - tmpfile > tmpfile1

#convert fastq to fasta, flatten, sort
awk '{if(NR%4==1) {printf(">%s\n",substr($0,2));} else if(NR%4==2) print;}' $inputprefix"_reads_1.fq" > tmpfile
sed '$!N;s/\n/\t/' tmpfile > tmpfile2
sort -k1,1 -k2,2rn -t - tmpfile2 > tmpfile

mv tmpfile tmpfile2

#merge both sorted flattened fasta files and convert into proper fasta
sort -k1,1 -k2,2rn -t - tmpfile1 tmpfile2 | tr '\t' '\n' > tmpfile

$SCRIPT_DIR/reorderReadsByFai $genomefai tmpfile tmpfile1

$SCRIPT_DIR/konnectoraddmateinfo $insertsize $insertsizedev tmpfile1 $interleavedinputfile $outputprefix"_total.fa" $outputprefix"_remaining.fa" $outputprefix"_extended.fa"


rm -f tmpfile tmpfile1 tmpfile2

