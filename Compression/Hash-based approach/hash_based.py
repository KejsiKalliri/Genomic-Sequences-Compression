import argparse
import time
import struct
from collections import Counter
from itertools import product
from pathlib import Path
import re

# Strategy:
#
# 1) Split sequence into segments.
#
# 2) For each segment, choose the smallest alphabet (dictionary):
#       ACGT only      -> 8-bit code per 4-mer
#       ACGTN only     -> 10-bit code per 4-mer
#       full LONG-15  -> 16-bit code per 4-mer
#
# 3) Consecutive repeated 4-mers are encoded as:
#       [REPEAT_MARKER][code][count]
#
#    This applies to every repeated 4-mer, not only AAAA/CCCC/NNNN...
#
# 4) The original FASTA line width is stored in the compressed file header and reused during decompression.


K = 4

ALPHABET_4 = "ACTG"
ALPHABET_5 = "ACTGN"
ALPHABET_15 = "ACTGNRYSWKMBDHV"

VALID_SYMBOLS = set(ALPHABET_15)

MODE_ACGT = 1
MODE_ACGTN = 2
MODE_LONG = 3

MODE_INFO = {
    MODE_ACGT: {
        "name": "ACGT",
        "alphabet": ALPHABET_4,
        "bit_width": 8,
    },
    MODE_ACGTN: {
        "name": "ACGTN",
        "alphabet": ALPHABET_5,
        "bit_width": 10,
    },
    MODE_LONG: {
        "name": "LONG",
        "alphabet": ALPHABET_15,
        "bit_width": 16,
    },
}

MAGIC = b"HASHBASED"
BLOB_LEN = struct.Struct("<Q")  # uint64: length of one compressed sequence blob

# File header after MAGIC:
# line_width:uint32, fasta_header_length:uint32
FILE_HEADER = struct.Struct("<II")

# Segment header:
# mode:uint8, original_segment_length:uint32, payload_length:uint32
SEGMENT_HEADER = struct.Struct("<BII")

# Block markers inside each segment payload
LITERAL_MARKER = 0x00
REPEAT_MARKER = 0x01

DEFAULT_SEGMENT_SIZE = 1_000_000
LITERAL_FLUSH_KMERS = 8192

DEFAULT_FASTA_HEADER = "decompressed_sequence"


#--------------------- FASTA / sequence utilities---------------------

SEQUENCE_FILE_PATTERN = re.compile(r"sequence_(\d+)(?:\..+)?$")

def folder_size_bytes(folder: str | Path) -> int:
    """
    Computes the total size of all files inside a folder.
    Directory metadata is not included, only file sizes.
    """

    folder = Path(folder)

    return sum(
        file.stat().st_size
        for file in folder.rglob("*")
        if file.is_file()
    )


def get_sequence_index(path: Path) -> int:
    """
    Extracts the sequence index from filenames such as:
        sequence_1.bin
        sequence_2.bin
        sequence_10.bin
    """

    match = SEQUENCE_FILE_PATTERN.match(path.name)

    if not match:
        raise ValueError(f"Invalid sequence filename: {path.name}")

    return int(match.group(1))


def list_sequence_files(input_dir: str | Path) -> list[Path]:
    """
    Lists sequence files in the correct biological order.

    Important:
    Lexicographic sorting would put sequence_10 before sequence_2.
    Therefore, we sort using the numeric index in the filename.
    """

    input_dir = Path(input_dir)

    files = [
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.name.startswith("sequence_")
    ]

    files.sort(key=get_sequence_index)

    return files


def append_fasta_record(f, header: str, seq: str, line_width: int):
    """
    Appends one FASTA record to an opened output FASTA file.
    """

    if line_width <= 0:
        line_width = 80

    f.write(f">{header}\n")

    for i in range(0, len(seq), line_width):
        f.write(seq[i:i + line_width] + "\n")


def clean_sequence(seq: str) -> str:
    """
    Removes whitespace/newlines, converts to uppercase, and validates LONG symbols.
    """

    seq = "".join(seq.split()).upper()  # removes whitespaces

    invalid = set(seq) - VALID_SYMBOLS
    if invalid:
        raise ValueError(f"Invalid DNA symbols found: {invalid}")

    return seq


def detect_line_width(line_lengths: list[int], default_line_width: int = 80) -> int:
    """
    Detects the original FASTA line width.

    Usually all sequence lines have the same length except the last one.
    Therefore, the most common line length is used.
    """

    if not line_lengths:
        return default_line_width

    return Counter(line_lengths).most_common(1)[0][0]


def iter_fasta_records(path: str | Path, default_line_width: int = 80):
    """
    Iterates over a FASTA file record by record.

    Yields:
        header, sequence, line_width

    Each FASTA sequence is kept separate, so each one can be compressed
    into its own independent blob/subfile.
    """

    path = Path(path)

    current_header = None
    seq_parts = []
    line_lengths = []

    def yield_current_record():
        nonlocal current_header, seq_parts, line_lengths

        if not seq_parts:
            return None

        seq = clean_sequence("".join(seq_parts))
        line_width = detect_line_width(line_lengths, default_line_width)

        header = (
            current_header
            if current_header is not None
            else DEFAULT_FASTA_HEADER
        )

        return header, seq, line_width

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if not line:
                continue

            if line.startswith(">"):
                record = yield_current_record()

                if record is not None:
                    yield record

                current_header = line[1:].strip() or DEFAULT_FASTA_HEADER
                seq_parts = []
                line_lengths = []
                continue

            seq_line = line.strip()

            if not seq_line:
                continue

            seq_parts.append(seq_line)
            line_lengths.append(len(seq_line))

    record = yield_current_record()

    if record is not None:
        yield record


def read_fasta_or_plain(path: str | Path, default_line_width: int = 80) -> tuple[str, int, str]:
    """
    Reads either FASTA or plain DNA text.

    FASTA headers starting with '>' are ignored for the sequence, but the first header is preserved as metadata.

    Returns:
        sequence, detected_line_width, fasta_header
    """

    path = Path(path)
    parts = []
    line_lengths = []
    fasta_header = DEFAULT_FASTA_HEADER

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if not line:
                continue

            if line.startswith(">"):
                if fasta_header == DEFAULT_FASTA_HEADER:
                    fasta_header = line[1:].strip() or DEFAULT_FASTA_HEADER
                continue

            seq_line = line.strip()

            if not seq_line:
                continue

            parts.append(seq_line)
            line_lengths.append(len(seq_line))

    seq = clean_sequence("".join(parts))
    line_width = detect_line_width(line_lengths, default_line_width)

    if line_width <= 0:
        line_width = default_line_width

    return seq, line_width, fasta_header


#------------------------------ Hash tables for 4-mers-------------------------

_TABLE_CACHE: dict[tuple[str, int], tuple[dict[str, int], dict[int, str]]] = {}

def build_hash_tables(alphabet: str, k: int = K) -> tuple[dict[str, int], dict[int, str]]:
    """
    Builds:
    - kmer_to_code: 4-mer -> integer code
    - code_to_kmer: integer code -> 4-mer
    """

    key = (alphabet, k)

    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    kmers = ["".join(p) for p in product(alphabet, repeat=k)]

    kmer_to_code = {kmer: code for code, kmer in enumerate(kmers)}
    code_to_kmer = {code: kmer for kmer, code in kmer_to_code.items()}

    _TABLE_CACHE[key] = (kmer_to_code, code_to_kmer)

    return kmer_to_code, code_to_kmer


def choose_mode(segment: str) -> int:
    """
    Chooses the most compact encoding mode (dictionary) for a segment.

    If the segment contains only A,C,T,G:
        use 8-bit 4-mer encoding.

    If the segment contains only A,C,T,G,N:
        use 10-bit 4-mer encoding.

    Otherwise:
        use 16-bit 4-mer encoding over the full LONG alphabet.
    """

    symbols = set(segment)

    if symbols <= set(ALPHABET_4):
        return MODE_ACGT

    if symbols <= set(ALPHABET_5):
        return MODE_ACGTN

    return MODE_LONG


#------------------------------- Varint encoding--------------------------

def encode_varint(value: int) -> bytes:
    """
    Encodes an integer using variable-length integer encoding, meaning that it stores a number using as few bytes as possible.
    Small values use fewer bytes.
    """

    if value < 0:
        raise ValueError("Varint cannot encode negative values.")

    output = bytearray()

    while True:
        byte = value & 0x7F
        value >>= 7

        if value:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            break

    return bytes(output)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """
    Decodes a variable-length integer from data starting at offset.
    Returns:
        value, new_offset
    """

    shift = 0
    value = 0

    while True:
        if offset >= len(data):
            raise ValueError("Invalid compressed file: truncated varint.")

        byte = data[offset]
        offset += 1

        value |= (byte & 0x7F) << shift

        if not (byte & 0x80):
            break

        shift += 7

        if shift > 63:
            raise ValueError("Invalid compressed file: varint too large.")

    return value, offset


def varint_size(value: int) -> int:
    """
    Returns the number of bytes needed to encode value as varint.
    """

    return len(encode_varint(value))



#---------------------------- Bit-packing for literal 4-mer codes-------------------------------

def pack_codes(codes: list[int], bit_width: int) -> bytes:
    """
    Takes a list of numeric codes and converts them into bytes, using a fixed bit width (bit_width) for each code
    """

    if not codes:
        return b""

    if bit_width == 8:
        return bytes(codes)

    if bit_width == 16:
        output = bytearray()

        for code in codes:
            output.extend(struct.pack("<H", code))

        return bytes(output)

    output = bytearray()
    buffer = 0
    bits_in_buffer = 0

    for code in codes:
        if code >= (1 << bit_width):
            raise ValueError(f"Code {code} does not fit in {bit_width} bits.")

        buffer |= code << bits_in_buffer
        bits_in_buffer += bit_width

        while bits_in_buffer >= 8:
            output.append(buffer & 0xFF)
            buffer >>= 8
            bits_in_buffer -= 8

    if bits_in_buffer > 0:
        output.append(buffer & 0xFF)

    return bytes(output)


def unpack_codes(data: bytes, n_codes: int, bit_width: int) -> list[int]:
    """
    Unpacks n_codes integer codes encoded using bit_width bits each.
    """

    if n_codes == 0:
        return []

    if bit_width == 8:
        if len(data) != n_codes:
            raise ValueError("Invalid literal block length for 8-bit mode.")

        return list(data)

    if bit_width == 16:
        expected = n_codes * 2

        if len(data) != expected:
            raise ValueError("Invalid literal block length for 16-bit mode.")

        return [
            struct.unpack_from("<H", data, i)[0]
            for i in range(0, expected, 2)
        ]

    codes = []
    buffer = 0
    bits_in_buffer = 0
    offset = 0
    mask = (1 << bit_width) - 1

    for _ in range(n_codes):
        while bits_in_buffer < bit_width:
            if offset >= len(data):
                raise ValueError("Invalid literal block: not enough packed data.")

            buffer |= data[offset] << bits_in_buffer
            bits_in_buffer += 8
            offset += 1

        code = buffer & mask
        codes.append(code)

        buffer >>= bit_width
        bits_in_buffer -= bit_width

    return codes


def packed_size_in_bytes(n_codes: int, bit_width: int) -> int:
    """
    Number of bytes required to store n_codes with bit_width bits each.
    """

    return (n_codes * bit_width + 7) // 8



#--------------------------- Block encoding--------------------------------

def emit_literal_block(payload: bytearray, literal_codes: list[int], bit_width: int): # bit_width -> indicates the dictionary that is used
    """
    Writes a literal block:  [LITERAL_MARKER][number_of_codes][packed_codes]
    """

    if not literal_codes:
        return

    payload.append(LITERAL_MARKER)
    payload.extend(encode_varint(len(literal_codes)))
    payload.extend(pack_codes(literal_codes, bit_width))


def emit_repeat_block(payload: bytearray, code: int, count: int): # count -> number of times that code is repeated
    """
    Writes a repeat block: [REPEAT_MARKER][code][count]
    code and count are stored as varints.
    """

    if count <= 0:
        raise ValueError("Repeat count must be positive.")

    payload.append(REPEAT_MARKER)
    payload.extend(encode_varint(code))
    payload.extend(encode_varint(count))


def should_use_repeat(count: int, code: int, bit_width: int, rle_min_run: int) -> bool:
    """
    Decides whether [REPEAT_MARKER][code][count]
    is smaller than literal encoding.
    """

    if count < rle_min_run:
        return False

    literal_size = packed_size_in_bytes(count, bit_width)

    repeat_size = (1 + varint_size(code) + varint_size(count)) # 1 -> REPEAT_MARKER is one byte

    return repeat_size < literal_size



#--------------------------------- Segment compression/decompression----------------------------------

def compress_segment(segment: str, rle_min_run: int = 2) -> bytes:
    """
    Compresses one segment.

    Segment format: [mode:uint8][segment_len:uint32][payload_len:uint32][payload]

    Payload format: [remainder_raw_bases][blocks...]

    The number of remainder bases is inferred from segment_len % 4.
    """

    if not segment:
        return b""

    mode = choose_mode(segment)

    info = MODE_INFO[mode]

    alphabet = info["alphabet"]
    bit_width = info["bit_width"]

    kmer_to_code, _ = build_hash_tables(alphabet)

    remainder_len = len(segment) % K
    main_len = len(segment) - remainder_len

    main_seq = segment[:main_len]
    remainder = segment[main_len:]

    payload = bytearray()

    # Store leftover 1, 2, or 3 bases directly.
    payload.extend(remainder.encode("ascii"))

    literal_codes = []  # is like a small buffer for normal 4-mers

    def flush_literals():
        nonlocal literal_codes

        if literal_codes:
            emit_literal_block(payload, literal_codes, bit_width)
            literal_codes = []

    i = 0

    while i < main_len:
        kmer = main_seq[i:i + K]
        code = kmer_to_code[kmer]

        count = 1
        j = i + K

        # General repeat detection:works for every repeated 4-mer, not only homopolymers.
        while j < main_len and main_seq[j:j + K] == kmer:
            count += 1
            j += K

        if should_use_repeat(count, code, bit_width, rle_min_run):
            flush_literals()
            emit_repeat_block(payload, code, count)
        else:
            literal_codes.extend([code] * count)

            if len(literal_codes) >= LITERAL_FLUSH_KMERS:
                flush_literals()

        i = j

    flush_literals()

    header = SEGMENT_HEADER.pack(mode, len(segment), len(payload))

    return header + bytes(payload)


def decompress_segment(mode: int, segment_len: int, payload: bytes) -> str:
    """
    Decompresses one segment.
    """

    if mode not in MODE_INFO:
        raise ValueError(f"Invalid segment mode: {mode}")

    info = MODE_INFO[mode]

    alphabet = info["alphabet"]
    bit_width = info["bit_width"]

    _, code_to_kmer = build_hash_tables(alphabet)

    remainder_len = segment_len % K
    main_kmer_count = (segment_len - remainder_len) // K

    if len(payload) < remainder_len:
        raise ValueError("Invalid segment payload: truncated remainder.")

    remainder = payload[:remainder_len].decode("ascii")
    offset = remainder_len

    decompressed_parts = []
    decoded_kmers = 0

    while offset < len(payload):
        block_type = payload[offset]
        offset += 1

        if block_type == LITERAL_MARKER:
            n_codes, offset = decode_varint(payload, offset)
            n_bytes = packed_size_in_bytes(n_codes, bit_width)

            if offset + n_bytes > len(payload):
                raise ValueError("Invalid literal block: truncated packed codes.")

            packed = payload[offset:offset + n_bytes]
            offset += n_bytes

            codes = unpack_codes(packed, n_codes, bit_width)

            for code in codes:
                if code not in code_to_kmer:
                    raise ValueError(f"Invalid literal code: {code}")

                decompressed_parts.append(code_to_kmer[code])

            decoded_kmers += n_codes

        elif block_type == REPEAT_MARKER:
            code, offset = decode_varint(payload, offset)
            count, offset = decode_varint(payload, offset)

            if count == 0:
                raise ValueError("Invalid repeat block: count cannot be zero.")

            if code not in code_to_kmer:
                raise ValueError(f"Invalid repeat code: {code}")

            kmer = code_to_kmer[code]
            decompressed_parts.append(kmer * count)

            decoded_kmers += count

        else:
            raise ValueError(f"Invalid block marker: {block_type}")

    if decoded_kmers != main_kmer_count:
        raise ValueError(
            f"Invalid decompression: expected {main_kmer_count} 4-mers, "
            f"decoded {decoded_kmers}."
        )

    return "".join(decompressed_parts) + remainder



#--------------------------Whole-sequence compression/decompression-------------------------------------

def compress_sequence_to_bytes(seq: str, segment_size: int = DEFAULT_SEGMENT_SIZE, rle_min_run: int = 2, line_width: int = 80, fasta_header: str = DEFAULT_FASTA_HEADER) -> bytes:
    """
    Compresses the full sequence.

    The compressed file stores:
        MAGIC
        line_width
        FASTA header length
        FASTA header
        compressed segments
    """


    if segment_size <= 0:
        raise ValueError("segment_size must be positive.")

    if len(seq) < DEFAULT_SEGMENT_SIZE:
        segment_size = len(seq)

    if line_width <= 0:
        raise ValueError("line_width must be positive.")

    seq = clean_sequence(seq)

    header_bytes = fasta_header.encode("utf-8")

    output = bytearray()
    output.extend(MAGIC)
    output.extend(FILE_HEADER.pack(line_width, len(header_bytes)))
    output.extend(header_bytes)

    for start in range(0, len(seq), segment_size):
        segment = seq[start:start + segment_size]
        output.extend(compress_segment(segment, rle_min_run=rle_min_run))

    return bytes(output)


def decompress_bytes_to_sequence(data: bytes) -> tuple[str, int, str]:
    """
    Decompresses data produced by compress_sequence_to_bytes().

    Supports both formats:

    New format:
        MAGIC
        line_width:uint32
        header_len:uint32
        fasta_header
        compressed segments

    Legacy format:
        MAGIC
        line_width:uint32
        compressed segments
    """

    if not data.startswith(MAGIC):
        raise ValueError("Invalid compressed file: wrong MAGIC header.")

    def decompress_segments_from_offset(offset: int) -> str:
        decompressed_segments = []

        while offset < len(data):
            if offset + SEGMENT_HEADER.size > len(data):
                raise ValueError("Invalid compressed file: truncated segment header.")

            mode, segment_len, payload_len = SEGMENT_HEADER.unpack_from(data, offset)
            offset += SEGMENT_HEADER.size

            if offset + payload_len > len(data):
                raise ValueError("Invalid compressed file: truncated segment payload.")

            payload = data[offset:offset + payload_len]
            offset += payload_len

            segment = decompress_segment(mode, segment_len, payload)
            decompressed_segments.append(segment)

        return "".join(decompressed_segments)

    offset = len(MAGIC)


    # 1. Try new format:
    #    MAGIC + line_width + header_len + fasta_header + segments

    try:
        if offset + FILE_HEADER.size > len(data):
            raise ValueError("Invalid compressed file: missing file header.")

        line_width, header_len = FILE_HEADER.unpack_from(data, offset)
        offset_after_file_header = offset + FILE_HEADER.size

        if line_width <= 0:
            raise ValueError("Invalid compressed file: invalid line width.")

        if header_len < 0:
            raise ValueError("Invalid compressed file: invalid header length.")

        if offset_after_file_header + header_len > len(data):
            raise ValueError("Invalid compressed file: truncated FASTA header.")

        fasta_header_bytes = data[offset_after_file_header: offset_after_file_header + header_len]
        fasta_header = fasta_header_bytes.decode("utf-8")
        segment_offset = offset_after_file_header + header_len
        sequence = decompress_segments_from_offset(segment_offset)

        return sequence, line_width, fasta_header

    except UnicodeDecodeError:
        # This usually means the file is in the old format without header_len/header.
        pass

    except ValueError:
        pass

    # 2. Legacy format:
    #    MAGIC + line_width:uint32 + segments

    legacy_offset = len(MAGIC)

    if legacy_offset + 4 > len(data):
        raise ValueError("Invalid legacy compressed file: missing line width.")

    line_width = int.from_bytes(
        data[legacy_offset:legacy_offset + 4],
        byteorder="little",
        signed=False
    )

    if line_width <= 0:
        raise ValueError("Invalid legacy compressed file: invalid line width.")

    segment_offset = legacy_offset + 4
    sequence = decompress_segments_from_offset(segment_offset)

    return sequence, line_width, "decompressed_sequence"


def compress_nucleotides_to_bytes(seq: str, segment_size: int = DEFAULT_SEGMENT_SIZE, rle_min_run: int = 2, line_width: int = 80) -> bytes:
    """
    Compresses only the nucleotide sequence.

    The compressed file stores:
        MAGIC
        line_width:uint32
        header_len:uint32 = 0
        compressed segments

    """

    if segment_size <= 0:
        raise ValueError("segment_size must be positive.")

    if line_width <= 0:
        raise ValueError("line_width must be positive.")

    seq = clean_sequence(seq)

    output = bytearray()
    output.extend(MAGIC)

    # FILE_HEADER is struct.Struct("<II"), so it expects two integers:
    #   1) line_width
    #   2) header length
    # Here header length is 0 because we do not store a FASTA header.
    output.extend(FILE_HEADER.pack(line_width, 0))

    for start in range(0, len(seq), segment_size):
        segment = seq[start:start + segment_size]
        output.extend(compress_segment(segment, rle_min_run=rle_min_run))

    return bytes(output)


def decompress_bytes_to_nucleotides(data: bytes) -> tuple[str, int]:
    """
    Decompresses data produced by compress_nucleotides_to_bytes().

    Returns:
        sequence, original_line_width
    """

    seq, line_width, fasta_header = decompress_bytes_to_sequence(data)

    return seq, line_width


def compress_file(input_path: str | Path, output_path: str | Path, segment_size: int = DEFAULT_SEGMENT_SIZE, rle_min_run: int = 2):
    """
    Compresses a multi-sequence FASTA file at sequence level.

    Each FASTA sequence is compressed independently and stored in a separate file:
        sequence_1.hash
        sequence_2.hash
        sequence_3.hash
        ...

    The header of each sequence is stored inside its own compressed blob.
    """

    input_path = Path(input_path)
    output_dir = Path(output_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old sequence files from previous runs
    for old_file in output_dir.glob("sequence_*"):
        if old_file.is_file():
            old_file.unlink()

    original_file_size = input_path.stat().st_size

    total_sequence_size = 0
    num_sequences = 0

    full_start = time.time()

    for fasta_header, seq, line_width in iter_fasta_records(input_path):
        num_sequences += 1

        sequence_name = f"sequence_{num_sequences}.hash"
        sequence_output_path = output_dir / sequence_name

        sequence_original_size = len(seq.encode("ascii"))
        total_sequence_size += sequence_original_size

        sequence_start = time.time()
        compressed_blob = compress_sequence_to_bytes( seq, segment_size=segment_size, rle_min_run=rle_min_run, line_width=line_width)
        sequence_output_path.write_bytes(compressed_blob)
        sequence_compression_time = time.time() - sequence_start

        compressed_sequence_size = sequence_output_path.stat().st_size

        sequence_gain = (
            1 - compressed_sequence_size / sequence_original_size
            if sequence_original_size > 0
            else 0.0
        )

        print(f"{sequence_name}")
        print(f"  Header:             {fasta_header}")
        print(f"  Original size:      {sequence_original_size} bytes")    # header + sequence
        print(f"  Compressed size:    {compressed_sequence_size} bytes")  # header + sequence (te kompresuara)
        # print(f"  Compression gain:   {sequence_gain:.4f}")
        print(f"  Compression time:   {sequence_compression_time:.4f} seconds")

    full_compression_time = time.time() - full_start

    compressed_folder_size = folder_size_bytes(output_dir)

    gain_vs_original_file = (
        1 - compressed_folder_size / original_file_size
        if original_file_size > 0
        else 0.0
    )

    print("\nFull FASTA compression finished.")
    print(f"Input FASTA file:          {input_path}")
    print(f"Output folder:             {output_dir}")
    print(f"Number of sequences:       {num_sequences}")
    print(f"Original FASTA file size:  {original_file_size} bytes")
    # print(f"Original sequence size:    {total_sequence_size} bytes")
    print(f"Compressed folder size:    {compressed_folder_size} bytes")
    print(f"Compression gain:    {gain_vs_original_file:.4f}")
    print(f"Full compression time:     {full_compression_time:.4f} seconds")


def decompress_file(input_path: str | Path, output_path: str | Path):
    """
    Decompresses a folder containing sequence-level compressed files.

    The files are read in order:
        sequence_1.hash
        sequence_2.hash
        sequence_3.hash
        ...

    Then all sequences are written back into one FASTA file.
    """

    input_dir = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_files = list_sequence_files(input_dir)

    if not sequence_files:
        raise ValueError(f"No sequence files found in folder: {input_dir}")

    num_sequences = 0
    total_bases = 0

    full_start = time.time()

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for sequence_file in sequence_files:
            sequence_start = time.time()

            compressed_blob = sequence_file.read_bytes()

            seq, line_width, fasta_header = decompress_bytes_to_sequence(compressed_blob)

            append_fasta_record(f=out, header=fasta_header, seq=seq, line_width=line_width)

            sequence_decompression_time = time.time() - sequence_start

            num_sequences += 1
            total_bases += len(seq)

            print(f"{sequence_file.name}")
            print(f"  Header:               {fasta_header}")
            print(f"  Decompressed length:  {len(seq)} bases")
            print(f"  Decompression time:   {sequence_decompression_time:.4f} seconds")

    full_decompression_time = time.time() - full_start

    print("\nFull FASTA decompression finished.")
    print(f"Input folder:              {input_dir}")
    print(f"Output FASTA file:         {output_path}")
    print(f"Number of sequences:       {num_sequences}")
    print(f"Full decompression time:   {full_decompression_time:.4f} seconds")


def evaluate_compression(input_path: str | Path, segment_size: int = DEFAULT_SEGMENT_SIZE, rle_min_run: int = 2):
    """
    Runs sequence-level compression and decompression in memory only.

    It does NOT save the compressed file.
    It does NOT save the decompressed file.

    Each FASTA sequence is compressed as a separate blob.
    """

    input_path = Path(input_path)

    original_file_size = input_path.stat().st_size
    total_sequence_size = 0
    num_sequences = 0

    archive = bytearray()
    archive.extend(MAGIC)

    # Compression in memory

    start = time.time()

    original_records = []

    for fasta_header, seq, line_width in iter_fasta_records(input_path):
        num_sequences += 1
        total_sequence_size += len(seq.encode("ascii"))

        original_records.append((fasta_header, seq, line_width))

        compressed_blob = compress_sequence_to_bytes(seq, segment_size=segment_size, rle_min_run=rle_min_run, line_width=line_width, fasta_header=fasta_header)

        archive.extend(BLOB_LEN.pack(len(compressed_blob)))
        archive.extend(compressed_blob)

    compression_time = time.time() - start

    compressed_size = len(archive)


    # Decompression in memory

    start = time.time()

    offset = len(MAGIC)
    restored_records = []

    while offset < len(archive):
        if offset + BLOB_LEN.size > len(archive):
            raise ValueError("Invalid in-memory archive: truncated blob length.")

        blob_len = BLOB_LEN.unpack_from(archive, offset)[0]
        offset += BLOB_LEN.size

        compressed_blob = bytes(archive[offset:offset + blob_len])
        offset += blob_len

        seq, line_width, fasta_header = decompress_bytes_to_sequence(compressed_blob)
        restored_records.append((fasta_header, seq, line_width))

    decompression_time = time.time() - start

    # Correctness check
    assert len(original_records) == len(restored_records), (
        "Decompression failed: number of sequences is different."
    )

    for original, restored in zip(original_records, restored_records):
        original_header, original_seq, original_line_width = original
        restored_header, restored_seq, restored_line_width = restored

        assert original_header == restored_header, "Header was not preserved."
        assert original_seq == restored_seq, "Sequence was not preserved."
        assert original_line_width == restored_line_width, "Line width was not preserved."

    gain_vs_sequence = (
        1 - compressed_size / total_sequence_size
        if total_sequence_size > 0
        else 0.0
    )

    gain_vs_file = (
        1 - compressed_size / original_file_size
        if original_file_size > 0
        else 0.0
    )

    print("Compression evaluation finished.")
    print(f"Input file:              {input_path}")
    print(f"Number of sequences:     {num_sequences}")
    print(f"Original sequence size:  {total_sequence_size} bytes")
    print(f"Original file size:      {original_file_size} bytes")
    print(f"Compressed size:         {compressed_size} bytes")
    print(f"Gain vs sequence:        {gain_vs_sequence:.4f}")
    print(f"Gain vs original file:   {gain_vs_file:.4f}")
    print(f"Compression time:        {compression_time:.4f} seconds")
    print(f"Decompression time:      {decompression_time:.4f} seconds")


#------------------------------ Tests-------------------------------------

def test_algorithm():

    seq = ("ACTG" * 1000 + "ACTGN" * 500 + "RYSWKMBDHV" * 200 + "AAAA" * 200 + "ACTG" * 300 + "TG")

    seq = clean_sequence(seq)

    start = time.time()
    compressed = compress_sequence_to_bytes(seq, segment_size=10_000, rle_min_run=2, line_width=70, fasta_header="test_sequence")
    print(f"Compression time: {time.time() - start:.6f} seconds")

    start = time.time()
    decompressed, line_width, fasta_header = decompress_bytes_to_sequence(compressed)
    print(f"Decompression time: {time.time() - start:.6f} seconds")

    assert seq == decompressed
    assert line_width == 70
    assert fasta_header == "test_sequence"

    original_size = len(seq.encode("ascii"))
    compressed_size = len(compressed)
    gain = 1 - compressed_size / original_size

    print(f"Original sequence size: {original_size} bytes")
    print(f"Compressed size:        {compressed_size} bytes")
    print(f"Compression gain:       {gain:.4f}")


def test_on_file():
    """
    Example test on a real FASTA file.
    """

    seq_path = Path("./fasta files/seq_10_000.fa")

    seq, line_width, fasta_header = read_fasta_or_plain(seq_path)

    start = time.time()
    compressed = compress_sequence_to_bytes(seq, segment_size=1_000_000, rle_min_run=2, line_width=line_width, fasta_header=fasta_header)
    print(f"Compression time: {time.time() - start:.4f} seconds")

    start = time.time()
    decompressed, restored_line_width, restored_header = decompress_bytes_to_sequence(compressed)
    print(f"Decompression time: {time.time() - start:.4f} seconds")

    assert seq == decompressed
    assert line_width == restored_line_width
    assert fasta_header == restored_header

    original_size = len(seq.encode("ascii"))
    compressed_size = len(compressed)
    gain = 1 - compressed_size / original_size

    print(f"Original sequence size: {original_size} bytes")
    print(f"Compressed size:        {compressed_size} bytes")
    print(f"Compression gain:       {gain:.4f}")
    print(f"Line width:             {restored_line_width}")


# Command-line interface

# def main():
#     parser = argparse.ArgumentParser(
#         description="Adaptive hash-based DNA compressor with RLE for repeated 4-mers."
#     )
#
#     subparsers = parser.add_subparsers(dest="command")
#
#     compress_parser = subparsers.add_parser("compress", help="Compress FASTA/plain DNA file")
#     compress_parser.add_argument("--input", required=True, help="Input FASTA/plain DNA file")
#     compress_parser.add_argument("--output", required=True, help="Output compressed file")
#     compress_parser.add_argument("--segment-size", type=int, default=DEFAULT_SEGMENT_SIZE)
#     compress_parser.add_argument("--rle-min-run", type=int, default=2)
#
#     decompress_parser = subparsers.add_parser("decompress", help="Decompress file")
#     decompress_parser.add_argument("--input", required=True, help="Input compressed file")
#     decompress_parser.add_argument("--output", required=True, help="Output FASTA file")
#
#     subparsers.add_parser("test", help="Run internal correctness test")
#
#     evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an already compressed file without recompressing")
#     evaluate_parser.add_argument("--input", required=True, help="Original FASTA/plain DNA file")
#     # evaluate_parser.add_argument("--output", required=True, help="C DNA file")
#
#     args = parser.parse_args()
#
#     if args.command == "compress":
#         compress_file(
#             input_path=args.input,
#             output_path=args.output,
#             segment_size=args.segment_size,
#             rle_min_run=args.rle_min_run
#         )
#
#     elif args.command == "decompress":
#         decompress_file(
#             input_path=args.input,
#             output_path=args.output
#         )
#
#     elif args.command == "evaluate":
#         evaluate_compression(
#             input_path=args.input,
#             # output_path=args.output,
#             # segment_size=args.segment_size,
#             # rle_min_run=args.rle_min_run
#         )
#
#     elif args.command == "test":
#         test_algorithm()
#
#     else:
#         parser.print_help()


# if __name__ == "__main__":
    # main()
    # test_on_file()

# main()
# py compression_new.py compress --input "./fasta files/seq_100_000_000.fa" --output "./compressed files/seq_100_000_000.dict"
# py compression_new.py decompress --input "./compressed files/seq_100_000_000.dict" --output "./decompressed files/seq_100_000_000.fa"
# py compression_new.py evaluate --input "./fasta files/seq_100_000_000.fa"