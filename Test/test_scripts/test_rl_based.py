from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]

# Change this path only if you rename the folder.
RL_DIR = ROOT / "Compression" / "RL-based approach"

TEST_DATA_DIR = ROOT / "Test" / "test_data"
OUTPUT_DIR = ROOT / "Test" / "test_outputs" / "rl_based"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RL_DIR))

from pretrain_model import read_fasta_records
from feature_cache import build_window_feature_cache
from compressor_runner import (
    compress_nucleotides_with_trained_rl,
    decompress_nucleotides_with_metadata,
)


MODEL_PATH = RL_DIR / "pretrained_dqn_model_2.pt"

WINDOW_SIZE = 1000
SEGMENT_LENGTHS = [200000, 400000, 600000, 800000]

TEST_FILES = [
    "Sequence_ACGT.fa",
    "Sequence_ACGTN.fa",
    "Sequence_IUPAC.fa",
    "Full_FASTA_file.fa",
]


for file_name in TEST_FILES:
    print("\n====================================================")
    print(f"Testing RL-based method on FASTA file: {file_name}")
    print("====================================================")

    input_path = TEST_DATA_DIR / file_name

    if not input_path.exists():
        raise FileNotFoundError(f"Missing test file: {input_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing pretrained model: {MODEL_PATH}")

    records = read_fasta_records(input_path)

    if not records:
        raise ValueError(f"No valid FASTA records found in: {input_path}")

    is_multi_sequence = len(records) > 1

    # If the input FASTA has several sequences, create one folder for that file.
    # Each compressed sequence will be stored separately inside this folder.
    if is_multi_sequence:
        compressed_output_path = OUTPUT_DIR / f"{input_path.stem}_compressed_sequences"
        compressed_output_path.mkdir(parents=True, exist_ok=True)
    else:
        compressed_output_path = OUTPUT_DIR

    total_original_size = 0
    total_compressed_size = 0
    total_compression_time = 0.0
    total_decompression_time = 0.0

    for index, record in enumerate(records, start=1):
        header = record["header"]
        sequence = record["sequence"]
        line_width = record["line_width"]

        print("\n----------------------------------------")
        print(f"Sequence {index}")
        print("----------------------------------------")
        print(f"Header: {header}")
        # print(f"Length: {len(sequence):,} bp")

        original_size = len(sequence.encode("ascii"))

        # Build RL feature matrix for the current sequence.
        feature_start = time.perf_counter()
        window_features = build_window_feature_cache(
            current_nucleotides=sequence,
            window_size=WINDOW_SIZE,
        )
        feature_time = time.perf_counter() - feature_start

        # Compress one sequence.
        compression_start = time.perf_counter()
        compressed_blob, segments = compress_nucleotides_with_trained_rl(
            current_nucleotides=sequence,
            line_width=line_width,
            window_features=window_features,
            model_path=str(MODEL_PATH),
            segment_lengths=SEGMENT_LENGTHS,
            level=9,
            parallel=False,
            workers=None,
            window_size=WINDOW_SIZE,
            device="cpu",
        )
        compression_time = time.perf_counter() - compression_start

        if is_multi_sequence:
            compressed_file = compressed_output_path / f"sequence_{index}.rlc"
        else:
            compressed_file = compressed_output_path / f"{input_path.stem}.rlc"

        compressed_file.write_bytes(compressed_blob)

        compressed_size = compressed_file.stat().st_size
        compression_gain = 1.0 - (compressed_size / original_size)

        full_compression_time = feature_time + compression_time

        # Decompress one sequence.
        decompression_start = time.perf_counter()
        reconstructed_bytes, reconstructed_line_width = decompress_nucleotides_with_metadata(
            compressed_file.read_bytes(),
            parallel=False,
            workers=None,
        )
        decompression_time = time.perf_counter() - decompression_start

        reconstructed_sequence = reconstructed_bytes.decode("ascii")

        if reconstructed_sequence != sequence:
            raise AssertionError(f"Exact reconstruction failed for sequence {index} in {file_name}")

        if reconstructed_line_width != line_width:
            raise AssertionError(f"Line width mismatch for sequence {index} in {file_name}")


        total_original_size += original_size
        total_compressed_size += compressed_size
        total_compression_time += full_compression_time
        total_decompression_time += decompression_time

        # print(f"Feature extraction time: {feature_time:.6f} s")
        print(f"Compression time: {full_compression_time:.6f} s")
        print(f"Decompression time: {decompression_time:.6f} s")
        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression gain: {compression_gain:.4f}")
        # print(f"Number of selected segments: {len(segments)}")
        print("Exact reconstruction: PASSED")

    total_compression_gain = 1.0 - (total_compressed_size / total_original_size)

    print("\n====================================================")
    print(f"Summary for FASTA file: {file_name}")
    print("====================================================")
    print(f"Number of sequences: {len(records)}")
    print(f"Total original size: {total_original_size:,} bytes")
    print(f"Total compressed size: {total_compressed_size:,} bytes")
    print(f"Total compression time: {total_compression_time:.6f} s")
    print(f"Total decompression time: {total_decompression_time:.6f} s")
    print(f"Total compression gain: {total_compression_gain:.4f}")
    print("FASTA-level exact reconstruction: PASSED")


print("\nAll RL-based compression/decompression tests passed successfully.")