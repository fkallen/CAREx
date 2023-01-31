This folder contains scripts and programs that can be used compute error statistics for extended reads.
Run `make` to compile the programs.

# Evaluation on simulated data for CAREx, Konnector2, and GapFiller

Generate paired-end data using the ART simulator: 
`art_illumina -ef -f 30 -i c_elegans.WS222.genomic.fa -l 100 -m 500 -na -o elegans30cov_500_10 -p -s 10 -ss HS20` 

Generate reference information of reads: 
`./makeErrorFreeExtendedRegions.sh c_elegans.WS222.genomic.fa elegans30cov_500_10.sam elegans30cov_500_10_pairgenomeregions.fasta `

Evaluate:
`./counteditsinpseudoreads 100 c_elegans.WS222.genomic.fa elegans30cov_500_10_pairgenomeregions.fasta carex_extended.fastq`

This requires specific metadata in the read headers of pseudo-reads. For example, they must begin with the numerical read id of one of the pair's reads, to find the corresponding genome sequence in pairgenomeregions.fasta. In addition, pseudo-reads must be sorted by that id in ascending order. For CAREx, the headers are in expected format. Sorting can be achieved by running CAREx with `--sortedOutput`. For other tools, additional processing of output files may be required before evaluation.

For Konnector2 without outward extension:
Konnector2 produces files konnector_reads_1.fq, konnector_reads_2.fq, konnector_pseudoreads.fa.

`./convertKonnectorOutput.sh konnector konnector_mod elegans30cov_500_10.fq 500 10 c_elegans.WS222.genomic.fa.fai`
adds required information and sorts the files. produces konnector_mod_extended.fa, konnector_mod_remaining.fa, konnector_mod_total.fa . Use konnector_mod_extended.fa for evaluation.

For GapFiller:
GapFiller produces gapfiller.fasta, gapfiller_trash.fasta

Run `./convertgapfilleroutput gapfiller.fasta gapfiller_trash.fasta gapfiller_mod_total.fa` 
Use gapfiller_mod_total for evaluation


# Evaluation on real data

Align pseudo-reads that don't have outward extension using bwa:
`bwa mem -t 64 c_elegans.WS222.genomic.fa extended.fastq > extended.sam`

Remove secondary and supplementary alignments using samtools:
`samtools view -@ 16 -h -F 2304 extended.sam > extended_nosec_nosup.sam`

Evaluate:
`./countgapeditsfromsam extended_nosec_nosup.sam c_elegans.WS222.genomic.fa 100`

