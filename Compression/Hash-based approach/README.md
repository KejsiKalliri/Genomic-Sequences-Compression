# Hash-based Deterministic DNA Compression

This folder contains the implementation of the Hash-based deterministic compression method for genomic sequences. The method is based on compact 4-mer encoding, adaptive dictionary selection, bit-packing, and run-length encoding of consecutive repeated 4-mers.

## Overview

The objective of this method is to provide a simple, deterministic, and fully lossless compression strategy for genomic sequences. Unlike the RL-based method, this approach does not require training, feature extraction, or model inference. It uses predefined dictionaries to map fixed-length DNA fragments to compact numerical codes.

The method is inspired by the paper:

A. Mehta and B. Patel, "DNA Compression Using Hash Based Data Structure."

The implementation in this repository extends the original idea by supporting real FASTA files that may contain not only the standard DNA bases A, C, G, and T, but also N and other IUPAC ambiguity symbols. It also adds run-length encoding for consecutive repeated 4-mers.

## Main Idea

The method scans a DNA sequence in consecutive groups of four bases, called 4-mers. Each 4-mer is mapped to a numerical code using a deterministic dictionary.

For the standard ACGT alphabet, there are:

```text
4^4 = 256 possible 4-mers
```
Therefore, each 4-mer can be represented using 8 bits instead of 4 ASCII characters.

## Encoding Modes

The method uses different encoding modes depending on the symbols present in each sequence segment:

```text
ACGT mode   -> 4^4 = 256 possible 4-mers     -> 8 bits per 4-mer
ACGTN mode  -> 5^4 = 625 possible 4-mers     -> 10 bits per 4-mer
LONG mode   -> 15^4 = 50,625 possible 4-mers -> 16 bits per 4-mer
```

The LONG mode supports the full IUPAC alphabet used in this implementation:

```text
A, C, T, G, N, R, Y, S, W, K, M, B, D, H, V
```

This adaptive dictionary selection avoids using the largest dictionary when the sequence contains only standard DNA bases.

## Requirements

This method uses only Python standard libraries.

## Running the Test Script

A test script is provided in the ```Test/``` folder. It compresses and decompresses the provided example FASTA files and verifies exact reconstruction.

From the repository root, run:

```text
bash Test/run_test_hash.sh
```

The test script performs the following operations:

1. Reads each test FASTA file.
2. Compresses each sequence using the Hash-based method.
3. Saves the compressed sequence as a .hash file.
4. Decompresses each compressed sequence.
5. Compares the reconstructed sequence with the original sequence.
6. Reports compression time, decompression time, compression gain, and exact reconstruction status.

For multi-sequence FASTA files, each sequence is compressed independently and stored as a separate compressed file inside an output folder corresponding to the original FASTA file.

## Test Data

The test data are located in: ```Test/test_data/```

Example test files include:
```text
Sequence_ACGT.fa
Sequence_ACGTN.fa
Sequence_IUPAC.fa
Full_FASTA_file.fa
```

These files are small example FASTA files used only to verify that the implementation works correctly. They are not intended to reproduce all experimental results from the thesis.

## Output

The test script creates compressed files inside:
```text
Test/test_outputs/hash_based/
```

For single-sequence FASTA files, one compressed file is generated.

For multi-sequence FASTA files, one folder is created, and each sequence is stored as a separate compressed file:
```text
Full_FASTA_file_compressed_sequences/
├── sequence_1.hash
├── sequence_2.hash
└── sequence_3.hash
```

The output folder is generated automatically during testing and does not need to be committed to the repository.

## Metrics Reported

For each compressed sequence, the test script reports:

* compression time;
* decompression time;
* original size;
* compressed size;
* compression gain;
* exact reconstruction status.

For multi-sequence FASTA files, the script also reports global metrics for the complete FASTA file.

## Notes on Exact Reconstruction

This method is lossless. After decompression, the reconstructed nucleotide sequence is compared with the original sequence. The test passes only if the original and reconstructed sequences are identical.

The FASTA header and original line width are preserved in the compressed data and recovered during decompression.