# RL-based Adaptive DNA Compression

This folder contains the implementation of the RL-based adaptive compression method for genomic sequences. The method combines a pretrained Double Deep Q-Network model with a bzip2-based compression core. The RL model is used to select segment lengths dynamically according to the local statistical properties of the DNA sequence, while the selected segments are compressed using the C-based compression core.

## Overview

The objective of this method is to improve lossless compression of genomic sequences by replacing fixed-size segmentation with adaptive segmentation. Instead of compressing the whole sequence as one block or using a fixed block size, the method analyzes the sequence through window-based features and selects one of several possible segment lengths.

The main steps are:

1. Read the input DNA sequence.
2. Extract window-based features from the sequence.
3. Use a pretrained Double DQN model to select segment lengths.
4. Compress each selected segment using the bzip2-based C compression core.
5. Store the compressed segments and the metadata required for decompression.
6. Reconstruct the original sequence exactly during decompression.

The pretrained model is used only during compression. During decompression, the model is not required because the compressed file contains the metadata needed to reconstruct the original sequence.

## Main Files

* `compressor_runner.py`
  Contains the main compression and decompression functions used by the RL-based method.

* `feature_cache.py`
  Extracts window-based features from DNA sequences.

* `RL.py`
  Defines the RL state representation, reward calculation, environment, replay buffer, and Double DQN agent.

* `pretrain_model.py`
  Contains the training pipeline used to pretrain the Double DQN model.

* `pretrained_dqn_model_2.pt`
  Pretrained Double DQN model used for adaptive segment selection.

* `setup.py`
  Builds the C extension used by the bzip2-based compression core.

* `C_core/`
  Contains the C wrapper used by Python to call the compression and decompression core.

* `third_party/`
  Contains third-party source files used by the bzip2-based compression core.

## Requirements

The method requires Python and the following Python packages:

```bash
numpy
torch
matplotlib
```

## Building the C Extension

Before running the RL-based compression method, the C extension must be built. From the repository root, run:

```bash
cd "Compression/RL-based approach"
python setup.py build_ext --inplace
```

After the build, return to the repository root:

```bash
cd ../..
```

## Running the Test Script

A test script is provided in the `Test/` folder. It compresses and decompresses the provided example FASTA files and checks that the reconstructed sequence is identical to the original sequence.

From the repository root, run:

```bash
bash Test/run_test_rl.sh
```

The test script performs the following operations:

1. Builds the C extension.
2. Loads the pretrained Double DQN model.
3. Reads each test FASTA file.
4. Compresses each sequence using the RL-based method.
5. Decompresses each compressed sequence.
6. Verifies exact reconstruction.
7. Reports compression time, decompression time, compression gain, and number of selected segments.

For multi-sequence FASTA files, each sequence is compressed independently and stored as a separate compressed file inside an output folder corresponding to the original FASTA file.

## Test Data

The test data are located in:

```text
Test/test_data/
```

Example test files include:

```text
Sequence_ACGT.fa
Sequence_ACGTN.fa
Sequence_IUPAC.fa
Full_FASTA_file.fa
```

These files are small example FASTA files used only to verify that the implementation works correctly. They are not intended to reproduce all experimental results.

## Output

The test script creates compressed files inside:

```text
Test/test_outputs/rl_based/
```

For single-sequence FASTA files, one compressed file is generated.

For multi-sequence FASTA files, one folder is created, and each sequence is stored as a separate compressed file:

```text
Full_FASTA_file_compressed_sequences/
├── sequence_1.rlc
├── sequence_2.rlc
└── sequence_3.rlc
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

The FASTA line width is also stored as metadata and recovered during decompression.

## Third-Party Code Notice

The RL-based method uses a bzip2-based compression core. Parts of this core are based on open-source bzip2/libbzip2 source code originally developed by Julian Seward.

The original copyright and license notices are preserved in the relevant third-party source files. More details are provided in the repository-level `THIRD_PARTY_NOTICES.md` file.
