from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]

HASH_DIR = ROOT / "Compression" / "Hash-based approach"
TEST_DATA_DIR = ROOT / "Test" / "test_data"
OUTPUT_DIR = ROOT / "Test" / "test_outputs" / "hash_based"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HASH_DIR))

from hash_based import (
    iter_fasta_records,
    compress_sequence_to_bytes,
    decompress_bytes_to_sequence,
)


SEGMENT_SIZE = 1_000_000
RLE_MIN_RUN = 2

TEST_FILES = [
    "Sequence_ACGT.fa",
    "Sequence_ACGTN.fa",
    "Sequence_IUPAC.fa",
    "Full_FASTA_file.fa",
]


for file_name in TEST_FILES:
    print("\n====================================================")
    print(f"Testing Hash-based method on FASTA file: {file_name}")
    print("====================================================")

    input_path = TEST_DATA_DIR / file_name

    if not input_path.exists():
        raise FileNotFoundError(f"Missing test file: {input_path}")

    records = list(iter_fasta_records(input_path))

    if not records:
        raise ValueError(f"No valid FASTA records found in: {input_path}")

    is_multi_sequence = len(records) > 1

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
        fasta_header, sequence, line_width = record

        print("\n----------------------------------------")
        print(f"Sequence {index}")
        print("----------------------------------------")
        print(f"Header: {fasta_header}")
        # print(f"Length: {len(sequence):,} bp")

        original_size = len(sequence.encode("ascii"))

        compression_start = time.perf_counter()

        compressed_blob = compress_sequence_to_bytes(
            seq=sequence,
            segment_size=SEGMENT_SIZE,
            rle_min_run=RLE_MIN_RUN,
            line_width=line_width,
            fasta_header=fasta_header,
        )

        compression_time = time.perf_counter() - compression_start

        if is_multi_sequence:
            compressed_file = compressed_output_path / f"sequence_{index}.hash"
        else:
            compressed_file = compressed_output_path / f"{input_path.stem}.hash"

        compressed_file.write_bytes(compressed_blob)

        compressed_size = compressed_file.stat().st_size
        compression_gain = 1.0 - (compressed_size / original_size)

        decompression_start = time.perf_counter()

        reconstructed_sequence, reconstructed_line_width, reconstructed_header = decompress_bytes_to_sequence(
            compressed_file.read_bytes()
        )

        decompression_time = time.perf_counter() - decompression_start

        if reconstructed_sequence != sequence:
            raise AssertionError(f"Exact reconstruction failed for sequence {index} in {file_name}")

        if reconstructed_line_width != line_width:
            raise AssertionError(f"Line width mismatch for sequence {index} in {file_name}")

        if reconstructed_header != fasta_header:
            raise AssertionError(f"Header mismatch for sequence {index} in {file_name}")

        total_original_size += original_size
        total_compressed_size += compressed_size
        total_compression_time += compression_time
        total_decompression_time += decompression_time

        print(f"Compression time: {compression_time:.6f} s")
        print(f"Decompression time: {decompression_time:.6f} s")
        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression gain: {compression_gain:.4f}")
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


print("\nAll Hash-based compression/decompression tests passed successfully.")