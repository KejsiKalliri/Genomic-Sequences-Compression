# Compression Methods

This folder contains the two lossless genomic sequence compression methods developed in this repository.

The two implemented methods are:

```text
Compression/
├── RL-based approach/
│   ├── README.md
│   └── ...
├── Hash-based approach/
    ├── README.md
    └── ...

```

Each method has its own dedicated `README.md` file with a detailed explanation of the method, its implementation, requirements, and test procedure.

### 1. RL-based Adaptive Compression

The first method is based on adaptive segmentation using a pretrained reinforcement learning model. The RL agent selects segment lengths dynamically according to sequence features, and the selected segments are compressed using a bzip2-based C compression core.

More details are available in:

```text
Compression/RL-based approach/README.md
```

### 2. Hash-based Deterministic Compression

The second method is a deterministic 4-mer based compression strategy. It uses adaptive dictionary selection, bit-packing, and run-length encoding of consecutive repeated 4-mers.

More details are available in:

```text
Compression/Hash-based approach/README.md
```

## Testing the Methods

Both methods can be tested from the repository root using the provided bash scripts in the `Test/` folder.

### Test the RL-based method

From the repository root, run:

```bash
bash Test/run_test_rl.sh
```

This script builds the required C extension and then runs the RL-based compression/decompression tests.

The corresponding Python test script is located in:

```text
Test/test_scripts/test_rl_based.py
```

The C extension must be built first.


### Test the Hash-based method

From the repository root, run:

```bash
bash Test/run_test_hash.sh
```

The corresponding Python test script is located in:

```text
Test/test_scripts/test_hash_based.py
```

## Test Data

The test FASTA files used by both methods are stored in:

```text
Test/test_data/
```

Example files include:

```text
Sequence_ACGT.fa
Sequence_ACGTN.fa
Sequence_IUPAC.fa
Full_FASTA_file.fa
```

These files are small example FASTA files used to verify that the compression and decompression pipelines work correctly.

## Test Outputs

The test scripts generate compressed files inside:

```text
Test/test_outputs/
```

For example:

```text
Test/test_outputs/
├── rl_based/
└── hash_based/
```

This folder is generated automatically during testing and does not need to be committed to the repository.

## Exact Reconstruction

Both compression methods are lossless. During testing, each compressed sequence is decompressed and compared with the original sequence.

A test is considered successful only if the reconstructed sequence is identical to the original sequence.

## Notes

- The RL-based method requires the pretrained model and the compiled C extension.
- The Hash-based method uses only Python standard libraries.
- Each method can be tested either through the bash script or directly through its corresponding script in `Test/test_scripts/`.
- For detailed explanations, refer to the `README.md` file inside each method folder.
