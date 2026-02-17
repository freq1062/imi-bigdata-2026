#!/usr/bin/env python3
"""
Utility to convert CSV files to compressed CSV.gz format.
Uses streaming to handle files of arbitrary size without loading into memory.
"""

import gzip
import shutil
import argparse
from pathlib import Path


def csv_to_gzip(input_path, output_path=None, chunk_size=1024*1024):
    """
    Convert a CSV file to gzip-compressed CSV.gz format using streaming.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to the output CSV.gz file (optional, defaults to input_path + '.gz')
        chunk_size: Size of chunks to read/write in bytes (default: 1MB)
    
    Returns:
        Path to the created .csv.gz file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if output_path is None:
        output_path = Path(str(input_path) + '.gz')
    else:
        output_path = Path(output_path)
    
    # Ensure output has .gz extension
    if not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')
    
    print(f"Compressing: {input_path}")
    print(f"Output: {output_path}")
    
    # Stream the file in chunks to avoid memory issues
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb', compresslevel=6) as f_out:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
    
    # Get file sizes
    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {ratio:.1f}%")
    print(f"Successfully created: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV files to compressed CSV.gz format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                    # Creates data.csv.gz
  %(prog)s data.csv output.csv.gz      # Creates output.csv.gz
  %(prog)s data.csv -o compressed/     # Creates compressed/data.csv.gz
        """
    )
    
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', nargs='?', help='Output CSV.gz file path (optional)')
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')
    parser.add_argument('-c', '--chunk-size', type=int, default=1024*1024,
                        help='Chunk size in bytes (default: 1MB)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = None
    
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (input_path.name + '.gz')
    
    try:
        csv_to_gzip(input_path, output_path, args.chunk_size)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
