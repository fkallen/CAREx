# CAREx 1.0: Context-Aware Read Extension of Illumina reads.

CAREx is a software to produce longer pseudo-reads from paired-end Illumnina reads. Single-end reads are not supported.
CAREx targets fragment sizes (a.k.a. insert size) which exceed twice the read length.


## Prerequisites
* C++17 
* OpenMP
* Zlib
* GNU Make

## Additional prerequisites for GPU version
* CUDA Toolkit 11 or newer
* A CUDA-capable graphics card with Pascal architecture (e.g. Nvidia GTX 1080) or newer.

# Download
Clone the repository


# Build
The build process assumes that the required compilers are available in your PATH.

## Make
Run make to generate the executables.

CPU version: This produces an executable file carex-cpu in the top-level directory of CAREx
```
make / make cpu
```

GPU version: This produces an executable file carex-gpu in the top-level directory of CAREx
```
make gpu
```

Optionally, after executables have been built they can be copied to an installation directory via `make install`.
This will copy available executables to the directory PREFIX/bin. The default value for PREFIX is `/usr/local`.
A custom prefix can be set as follows:

```
make install PREFIX=/my/custom/prefix
```



# Example commands
The simplest command which only includes mandatory options is

```
./carex-cpu -i interleaved_paired_reads.fq -c 10 -d outputdir --minFragmentSize 280 --maxFragmentSize 320
```

This command will attempt to create a pseudo-read for each read pair in file interleaved_paired_reads.fq. The fragment size is assumed to be in range [280, 320]. 
The dataset coverage is assumed to be 10x. Pseudo-reads will be written to file outputdir/carex_extended.fastq. Only pseudo-reads of lengths 280-320 which connect both reads of a pair will be output. 
The available program parameters are listed below.

NOTE: CAREx does not work with single-end reads.
NOTE: CAREx is not intended for cases where minFragmentSize <= 2 * read length

Input files must be in fasta or fastq format, and may be gzip'ed. Specifying both fasta files and fastq files together is not allowed.
Output file will be uncompressed. Output pseudo-reads will be unsorted unless `--sortedOutput` is used. In this case the relative ordering of input pairs in preserved.

A more advanced usage could look like the following command. It enables progress counter `-p` and uses quality scores `-q` . The program should use 16 threads `-t 16` with a memory limit of 22 gigabyte `-m 22G`. Sequences which contain other letters than A,C,G,T, e.g. N, will be skipped `--excludeAmbiguous`. `-k` and `-h` specify the parameters of the hashing, namely the k-mer size and the number of hash tables.


```
./carex-cpu -i interleaved_paired_reads.fq -c 10 -d outputdir --minFragmentSize 280 --maxFragmentSize 320 -p -q --excludeAmbiguous -m 22G -t 16 -k 20 -h 32 
```

One of our benchmark datasets can be downloaded (3 gigabyte) and extended by executing the example script: 
```
bash example.sh
```

# GPU execution
The equivalent execution of the previous CPU command with the GPU version using the first GPU (gpu id 0) would be:

```
./carex-gpu -i interleaved_paired_reads.fq -c 10 -d outputdir --minFragmentSize 280 --maxFragmentSize 320 -p -q --excludeAmbiguous -m 22G -t 16 -k 20 -h 32 -g 0
```
Note the additional mandatory parameter `-g` which accepts a comma-separated list of integers to indicate which GPUs can be used. The integers must be between 0 and N-1, where N is the number of available GPUs in the system. At the moment, using multiple GPUs may work, but is not fully supported yet!

CAREx uses hash tables which, by default (`--warpcore 0`), are placed in CPU memory. To use accelerated hash tables in GPU memory, specify `--warpcore 1`. 

For best performance, we recommend setting a custom thread config using `--gpuExtenderThreadConfig numExtenders:numHashers`.  In this case, numHashers threads will be used to perform hash table accesses. The number of threads performing read extension is numExtenders. numHashers may be 0, indicating that extender threads should perform hashing on their own.
The best schedule depends on the hash table location and total number of threads to use.

Examples:
* `-t 4 --warpcore 1 --gpuExtenderThreadConfig 4:0` , GPU hash tables are fast. Don't need extra hashing threads
* `-t 16 --warpcore 0 --gpuExtenderThreadConfig 2:14` , CPU hash tables are slow. Need more hashing threads to keep extender threads (which use the GPU) busy
* `-t 4 --warpcore 0 --gpuExtenderThreadConfig 4:0` , CPU hash tables are slow, but using separate hashing threads is not efficient for small total number of threads.


 




# Specifying input files
## Interleaved
Two consecutive reads form a read pair. Use `-i reads_interleaved` .

## Split
Read number N in file 1 and read number N in file 2 form a read pair. Use `-i reads_1 -i reads_2`.


# Specifying output files
In addition to produced pseudo-reads, the remaining unconnected read pairs can be output by specifying `--outputRemaining` . This will write remaining reads in interleaved format to file outputdir/carex_remaining.fastq.

The default file names can be overwritten: `--eo extendedfile.fastq` `--ro remainingfile.fastq` 

# Output format
Headers of extended reads contain metadata describing the extension process, supporting filtering and evaluation. For example "reached:1 m:0 a:0 lens:0,100,208,308"

* reached: 0: pair not connected, 1: pair connected
* m: 0: connected directly, 1: connected by merging results of both strands
* a: 0: produces in first pass, 1: produced in additional pass
* len: stores begin and end of original reads within the pseudo-read


# Available program parameters
Please execute `./carex-cpu --help` or `./carex-gpu --help` to print a list of available parameters. Both versions share a common subset of parameters.

The following list is a subset of options.

```
--allowOutwardExtension       Will try to fill the gap and extend to the
                              outside. Default: false

-h, --hashmaps arg            The requested number of hash maps. Must be
                              greater than 0. The actual number of used hash
                              maps may be lower to respect the set memory
                              limit. Default: 48

-t, --threads arg             Maximum number of thread to use. Default: 1

-q, --useQualityScores        If set, quality scores (if any) are
                              considered during read extension. Default: false


-p, --showProgress            If set, progress information is shown during
                              execution

-m, --memTotal arg            Total memory limit in bytes. Can use suffix
                              K,M,G , e.g. 20G means 20 gigabyte. This option
                              is not a hard limit. Default: All free
                              memory.
```

If an option allows multiple values to be specified, the option can be repeated with different values.
As an alternative, multiple values can be separated by comma (,). Both ways can be used simultaneously.





