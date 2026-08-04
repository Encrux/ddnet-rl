"""Parse DDNet .gho ghost files and extract position waypoints.

Usage:
    python ghost_parser.py path/to/ghost.gho [--output waypoints.json] [--spacing 64]

Output format (JSON):
    {
        "source": "ghost",
        "map": "Tutorial",
        "num_points": 1234,
        "total_distance": 56789.0,
        "points": [[x0, y0], [x1, y1], ...]
    }
"""

import argparse
import json
import math
import struct
import sys
from typing import Optional


# === Huffman Decompression ===

# Static frequency table from DDNet (huffman.cpp)
HUFFMAN_FREQ_TABLE = [
    1 << 30, 4545, 2657, 431, 1950, 919, 444, 482, 2244, 617, 838, 542, 715, 1814, 304, 240, 754, 212, 647, 186,
    283, 131, 146, 166, 543, 164, 167, 136, 179, 859, 363, 113, 157, 154, 204, 108, 137, 180, 202, 176,
    872, 404, 168, 134, 151, 111, 113, 109, 120, 126, 129, 100, 41, 20, 16, 22, 18, 18, 17, 19,
    16, 37, 13, 21, 362, 166, 99, 78, 95, 88, 81, 70, 83, 284, 91, 187, 77, 68, 52, 68,
    59, 66, 61, 638, 71, 157, 50, 46, 69, 43, 11, 24, 13, 19, 10, 12, 12, 20, 14, 9,
    20, 20, 10, 10, 15, 15, 12, 12, 7, 19, 15, 14, 13, 18, 35, 19, 17, 14, 8, 5,
    15, 17, 9, 15, 14, 18, 8, 10, 2173, 134, 157, 68, 188, 60, 170, 60, 194, 62, 175, 71,
    148, 67, 167, 78, 211, 67, 156, 69, 1674, 90, 174, 53, 147, 89, 181, 51, 174, 63, 163, 80,
    167, 94, 128, 122, 223, 153, 218, 77, 200, 110, 190, 73, 174, 69, 145, 66, 277, 143, 141, 60,
    136, 53, 180, 57, 142, 57, 158, 61, 166, 112, 152, 92, 26, 22, 21, 28, 20, 26, 30, 21,
    32, 27, 20, 17, 23, 21, 30, 22, 22, 21, 27, 25, 17, 27, 23, 18, 39, 26, 15, 21,
    12, 18, 18, 27, 20, 18, 15, 19, 11, 17, 33, 12, 18, 15, 19, 18, 16, 26, 17, 18,
    9, 10, 25, 22, 22, 17, 20, 16, 6, 16, 15, 20, 14, 18, 24, 335, 1517,
]

HUFFMAN_EOF_SYMBOL = 256
HUFFMAN_MAX_SYMBOLS = 257
HUFFMAN_LUTBITS = 10
HUFFMAN_LUTSIZE = 1 << HUFFMAN_LUTBITS
HUFFMAN_LUTMASK = HUFFMAN_LUTSIZE - 1


class HuffmanNode:
    __slots__ = ['symbol', 'num_bits', 'bits', 'leaves']

    def __init__(self):
        self.symbol = 0
        self.num_bits = 0
        self.bits = 0
        self.leaves = [0xFFFF, 0xFFFF]


class HuffmanDecoder:
    def __init__(self):
        self.nodes = [HuffmanNode() for _ in range(HUFFMAN_MAX_SYMBOLS * 2)]
        self.decode_lut = [None] * HUFFMAN_LUTSIZE
        self.start_node_idx = 0
        self.num_nodes = 0
        self._build(HUFFMAN_FREQ_TABLE)

    def _build(self, freqs):
        # Initialize symbol nodes
        for i in range(HUFFMAN_MAX_SYMBOLS):
            self.nodes[i].symbol = i
            self.nodes[i].num_bits = 0xFFFFFFFF
            self.nodes[i].leaves = [0xFFFF, 0xFFFF]

        self.num_nodes = HUFFMAN_MAX_SYMBOLS

        # Build node list with frequencies
        node_freqs = []
        for i in range(HUFFMAN_MAX_SYMBOLS):
            f = 1 if i == HUFFMAN_EOF_SYMBOL else freqs[i]
            node_freqs.append([i, f])  # [node_id, frequency]

        # Construct tree by merging lowest frequency nodes
        while len(node_freqs) > 1:
            node_freqs.sort(key=lambda x: -x[1])  # sort descending
            a = node_freqs.pop()  # lowest
            b = node_freqs.pop()  # second lowest

            new_idx = self.num_nodes
            self.nodes[new_idx].num_bits = 0
            self.nodes[new_idx].leaves[0] = a[0]
            self.nodes[new_idx].leaves[1] = b[0]

            node_freqs.append([new_idx, a[1] + b[1]])
            self.num_nodes += 1

        self.start_node_idx = self.num_nodes - 1

        # Set bits recursively
        self._setbits(self.start_node_idx, 0, 0)

        # Build LUT
        for i in range(HUFFMAN_LUTSIZE):
            bits = i
            node_idx = self.start_node_idx
            for k in range(HUFFMAN_LUTBITS):
                node = self.nodes[node_idx]
                leaf = node.leaves[bits & 1]
                if leaf == 0xFFFF:
                    break
                node_idx = leaf
                bits >>= 1
                node = self.nodes[node_idx]
                if node.num_bits and node.num_bits != 0xFFFFFFFF:
                    self.decode_lut[i] = node_idx
                    break
            else:
                self.decode_lut[i] = node_idx

    def _setbits(self, node_idx, bits, depth):
        node = self.nodes[node_idx]
        if node.leaves[1] != 0xFFFF:
            self._setbits(node.leaves[1], bits | (1 << depth), depth + 1)
        if node.leaves[0] != 0xFFFF:
            self._setbits(node.leaves[0], bits, depth + 1)
        if node.num_bits:  # is a leaf
            node.bits = bits
            node.num_bits = depth

    def decompress(self, data: bytes, max_output: int = 65536) -> bytes:
        output = bytearray()
        src = data
        src_idx = 0
        src_end = len(src)

        bits = 0
        bitcount = 0
        eof_idx = HUFFMAN_EOF_SYMBOL

        while True:
            # Try LUT lookup
            node_idx = None
            if bitcount >= HUFFMAN_LUTBITS:
                node_idx = self.decode_lut[bits & HUFFMAN_LUTMASK]

            # Fill bits
            while bitcount < 24 and src_idx < src_end:
                bits |= src[src_idx] << bitcount
                bitcount += 8
                src_idx += 1

            # LUT lookup if not done
            if node_idx is None:
                node_idx = self.decode_lut[bits & HUFFMAN_LUTMASK]

            if node_idx is None:
                raise ValueError("Huffman decode error: no LUT entry")

            node = self.nodes[node_idx]

            if node.num_bits and node.num_bits != 0xFFFFFFFF:
                # Symbol found via LUT
                bits >>= node.num_bits
                bitcount -= node.num_bits
            else:
                # Walk tree bit by bit
                bits >>= HUFFMAN_LUTBITS
                bitcount -= HUFFMAN_LUTBITS

                while True:
                    leaf = node.leaves[bits & 1]
                    if leaf == 0xFFFF:
                        raise ValueError("Huffman decode error: invalid tree")
                    node_idx = leaf
                    node = self.nodes[node_idx]
                    bitcount -= 1
                    bits >>= 1

                    if node.num_bits and node.num_bits != 0xFFFFFFFF:
                        break
                    if bitcount == 0:
                        raise ValueError("Huffman decode error: out of bits")

            # Check EOF
            if node.symbol == eof_idx:
                break

            if len(output) >= max_output:
                raise ValueError("Huffman decode error: output overflow")
            output.append(node.symbol)

        return bytes(output)


# === Variable-Length Integer Decompression ===

def varint_decompress(data: bytes) -> list[int]:
    """Decompress variable-length encoded integers."""
    result = []
    idx = 0
    end = len(data)

    while idx < end:
        sign = (data[idx] >> 6) & 1
        value = data[idx] & 0x3F
        idx += 1

        masks = [0x7F, 0x7F, 0x7F, 0x0F]
        shifts = [6, 13, 20, 27]

        prev_had_extend = (data[idx - 1] & 0x80) != 0
        for i in range(4):
            if not prev_had_extend:
                break
            if idx >= end:
                break
            value |= (data[idx] & masks[i]) << shifts[i]
            prev_had_extend = (data[idx] & 0x80) != 0
            idx += 1

        if sign:
            value = ~value  # two's complement
        result.append(value)

    return result


# === Ghost File Parser ===

# Ghost data types
GHOSTDATA_TYPE_SKIN = 0
GHOSTDATA_TYPE_CHARACTER_NO_TICK = 1
GHOSTDATA_TYPE_CHARACTER = 2
GHOSTDATA_TYPE_START_TICK = 3

# Item sizes in int32s
ITEM_SIZES = {
    GHOSTDATA_TYPE_SKIN: 9,              # 36 bytes
    GHOSTDATA_TYPE_CHARACTER_NO_TICK: 11, # 44 bytes
    GHOSTDATA_TYPE_CHARACTER: 12,         # 48 bytes
    GHOSTDATA_TYPE_START_TICK: 1,         # 4 bytes
}

MAX_NAME_LENGTH = 16


def parse_ghost_header(f) -> dict:
    """Parse the ghost file header."""
    marker = f.read(8)
    if marker != b'TWGHOST\x00':
        raise ValueError(f"Invalid ghost marker: {marker}")

    version = struct.unpack('B', f.read(1))[0]
    if version not in (4, 5, 6):
        raise ValueError(f"Unsupported ghost version: {version}")

    owner = f.read(MAX_NAME_LENGTH).split(b'\x00')[0].decode('utf-8', errors='replace')
    map_name = f.read(64).split(b'\x00')[0].decode('utf-8', errors='replace')

    crc_or_zeroes = struct.unpack('>I', f.read(4))[0]
    num_ticks = struct.unpack('>I', f.read(4))[0]
    time_ms = struct.unpack('>I', f.read(4))[0]

    sha256 = None
    if version >= 6:
        sha256 = f.read(32).hex()
    # v4-5: no SHA256 field at all (header is shorter)

    return {
        'version': version,
        'owner': owner,
        'map': map_name,
        'crc': crc_or_zeroes,
        'num_ticks': num_ticks,
        'time_ms': time_ms,
        'sha256': sha256,
    }


def read_chunks(f, huffman: HuffmanDecoder, version: int) -> list:
    """Read all chunks from the ghost file."""
    items = []
    last_item = None

    while True:
        chunk_header = f.read(4)
        if len(chunk_header) < 4:
            break

        item_type = chunk_header[0]
        num_items = chunk_header[1]
        data_size = (chunk_header[2] << 8) | chunk_header[3]

        if num_items == 0 or data_size == 0:
            break

        compressed = f.read(data_size)
        if len(compressed) < data_size:
            break

        # Decompress: Huffman first, then VarInt
        try:
            huffman_out = huffman.decompress(compressed, max_output=65536)
            int_values = varint_decompress(huffman_out)
        except Exception as e:
            print(f"Warning: decompression error in chunk: {e}", file=sys.stderr)
            break

        # Get item size for this type
        if item_type not in ITEM_SIZES:
            print(f"Warning: unknown item type {item_type}, skipping", file=sys.stderr)
            continue
        item_size = ITEM_SIZES[item_type]

        # v5-6: Reset delta at start of each chunk (matches C++ ReadChunk behavior)
        # v4: Preserve delta across chunks
        if version != 4:
            last_item = None

        # Extract items
        offset = 0
        for _ in range(num_items):
            if offset + item_size > len(int_values):
                break

            raw = int_values[offset:offset + item_size]

            # Apply undiff (first item in chunk after reset is absolute)
            if last_item is not None and len(last_item) == item_size:
                raw = [last_item[j] + raw[j] for j in range(item_size)]

            last_item = raw[:]
            items.append((item_type, raw))
            offset += item_size

    return items


def extract_positions(ghost_path: str) -> list[tuple[float, float]]:
    """Extract (x, y) positions from a ghost file. Returns world coordinates."""
    huffman = HuffmanDecoder()

    with open(ghost_path, 'rb') as f:
        header = parse_ghost_header(f)
        print(f"Ghost: {header['owner']} on {header['map']}, "
              f"{header['num_ticks']} ticks, time={header['time_ms']}ms", file=sys.stderr)

        items = read_chunks(f, huffman, header['version'])

    positions = []
    for item_type, data in items:
        if item_type == GHOSTDATA_TYPE_CHARACTER or item_type == GHOSTDATA_TYPE_CHARACTER_NO_TICK:
            # Fields: X, Y, VelX, VelY, Angle, Direction, Weapon, HookState, HookX, HookY, AttackTick[, Tick]
            x = float(data[0])  # Ghost positions are already in world units (pixels)
            y = float(data[1])
            positions.append((x, y))

    print(f"Extracted {len(positions)} position samples", file=sys.stderr)
    return positions


def resample_waypoints(positions: list[tuple[float, float]], spacing: float = 64.0) -> list[tuple[float, float]]:
    """Resample positions to equally-spaced waypoints along the path.

    Args:
        positions: Raw (x, y) positions from ghost
        spacing: Distance between waypoints in world units (32 = 1 tile)
    """
    if len(positions) < 2:
        return positions

    waypoints = [positions[0]]
    accumulated = 0.0

    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist = math.sqrt(dx * dx + dy * dy)
        accumulated += dist

        while accumulated >= spacing:
            # Interpolate
            overshoot = accumulated - spacing
            t = 1.0 - (overshoot / dist) if dist > 0 else 1.0
            t = max(0.0, min(1.0, t))
            wx = positions[i - 1][0] + t * dx
            wy = positions[i - 1][1] + t * dy
            waypoints.append((wx, wy))
            accumulated -= spacing

    # Always include the last position
    last = positions[-1]
    if waypoints[-1] != last:
        waypoints.append(last)

    return waypoints


def save_waypoints(waypoints: list[tuple[float, float]], output_path: str,
                   source: str = "ghost", map_name: str = ""):
    """Save waypoints in the common JSON format."""
    # Calculate total path distance
    total_dist = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]
        total_dist += math.sqrt(dx * dx + dy * dy)

    data = {
        "source": source,
        "map": map_name,
        "num_points": len(waypoints),
        "total_distance": round(total_dist, 1),
        "spacing": round(total_dist / max(len(waypoints) - 1, 1), 1),
        "points": [[round(x, 1), round(y, 1)] for x, y in waypoints],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

    print(f"Saved {len(waypoints)} waypoints to {output_path} "
          f"(total distance: {total_dist:.0f} units)", file=sys.stderr)


def save_waypoints_binary(waypoints: list[tuple[float, float]], output_path: str):
    """Save waypoints as binary file for C++ server loading.

    Format: [num_points: uint32][x0: float32][y0: float32][x1: float32][y1: float32]...
    """
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(waypoints)))
        for x, y in waypoints:
            f.write(struct.pack('<ff', x, y))
    print(f"Saved {len(waypoints)} waypoints to {output_path} (binary)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Parse DDNet ghost files and extract waypoints")
    parser.add_argument("ghost_file", help="Path to .gho ghost file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: <ghost>.waypoints.json)")
    parser.add_argument("--binary", "-b", default=None, help="Output binary file for C++ server")
    parser.add_argument("--spacing", type=float, default=64.0,
                        help="Spacing between waypoints in world units (32=1 tile, default: 64)")
    parser.add_argument("--raw", action="store_true", help="Output raw positions without resampling")
    args = parser.parse_args()

    # Extract positions
    positions = extract_positions(args.ghost_file)
    if not positions:
        print("Error: no positions found in ghost file", file=sys.stderr)
        sys.exit(1)

    # Parse map name from header
    with open(args.ghost_file, 'rb') as f:
        header = parse_ghost_header(f)

    # Resample or use raw
    if args.raw:
        waypoints = positions
    else:
        waypoints = resample_waypoints(positions, spacing=args.spacing)

    # Save JSON
    output_path = args.output or args.ghost_file.rsplit('.', 1)[0] + '.waypoints.json'
    save_waypoints(waypoints, output_path, source="ghost", map_name=header['map'])

    # Save binary if requested
    if args.binary:
        save_waypoints_binary(waypoints, args.binary)


if __name__ == "__main__":
    main()
