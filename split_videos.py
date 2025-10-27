#!/usr/bin/env python3
"""
split_videos.py - Split videos into multiple directories using symbolic links
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List
import math

def get_video_files(directory: Path) -> List[Path]:
    """Get all video files from a directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    video_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)

def split_videos(source_dir: str, num_splits: int = 4, use_copy: bool = False) -> bool:
    """
    Split videos from source directory into multiple subdirectories.
    
    Args:
        source_dir: Path to source directory containing videos
        num_splits: Number of subdirectories to create
        use_copy: If True, copy files instead of creating symbolic links
    
    Returns:
        True if successful, False otherwise
    """
    
    # Validate inputs
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Error: Source directory '{source_dir}' does not exist!")
        return False
    
    if not source_path.is_dir():
        print(f"❌ Error: '{source_dir}' is not a directory!")
        return False
        
    if num_splits < 2:
        print(f"❌ Error: Number of splits must be at least 2!")
        return False
    
    # Get video files
    print(f"🔍 Scanning for video files in: {source_path}")
    video_files = get_video_files(source_path)
    
    if not video_files:
        print(f"❌ No video files found in {source_dir}")
        return False
    
    video_count = len(video_files)
    print(f"📹 Found {video_count} video files")
    
    # Calculate distribution
    videos_per_split = video_count // num_splits
    remainder = video_count % num_splits
    
    print(f"📊 Videos per split: {videos_per_split}")
    if remainder > 0:
        print(f"📊 Remainder: {remainder} (will be distributed to first {remainder} directories)")

    # Create split directories
    parent_dir = source_path.parent
    source_name = source_path.name
    split_dirs = []
    
    print(f"\n📁 Creating {num_splits} split directories...")
    for i in range(1, num_splits + 1):
        split_dir = parent_dir / f"{source_name}_{i}"
        split_dir.mkdir(exist_ok=True)
        split_dirs.append(split_dir)
        print(f"  ✓ Created: {split_dir}")
    
    # Distribute files
    print(f"\n🔄 Distributing video files...")
    file_index = 0
    
    for i, split_dir in enumerate(split_dirs, 1):
        # Calculate files for this split
        files_for_this_split = videos_per_split
        if i <= remainder:  # First 'remainder' directories get one extra file
            files_for_this_split += 1
        
        print(f"  📂 Processing {source_name}_{i}: {files_for_this_split} files")
        
        # Process files for this split
        for j in range(files_for_this_split):
            if file_index < video_count:
                video_file = video_files[file_index]
                target_path = split_dir / video_file.name
                
                try:
                    if use_copy:
                        # Copy file
                        import shutil
                        shutil.copy2(video_file, target_path)
                        print(f"    📋 Copied: {video_file.name}")
                    else:
                        # Create symbolic link
                        if target_path.exists():
                            target_path.unlink()  # Remove existing link
                        target_path.symlink_to(video_file.resolve())
                        print(f"    🔗 Linked: {video_file.name}")
                    
                    file_index += 1
                    
                except Exception as e:
                    print(f"    ❌ Error processing {video_file.name}: {e}")
                    return False
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  Original directory: {source_path} ({video_count} files) - UNCHANGED")
    print(f"  Created {num_splits} split directories:")
    
    total_distributed = 0
    for i, split_dir in enumerate(split_dirs, 1):
        actual_count = len(list(split_dir.glob('*')))
        total_distributed += actual_count
        print(f"    {split_dir}: {actual_count} files")
    
    print(f"  Total files distributed: {total_distributed}")
    print(f"✅ Video splitting completed successfully!")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Split videos from a directory into multiple subdirectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_videos.py /path/to/videos 4
  python split_videos.py /path/to/videos 3 --copy
  python split_videos.py --source /path/to/videos --splits 4
        """
    )
    
    parser.add_argument(
        'source', 
        nargs='?',
        help='Source directory containing videos'
    )
    
    parser.add_argument(
        'splits', 
        nargs='?',
        type=int,
        help='Number of splits (default: 2)'
    )
    
    parser.add_argument(
        '--source', '-s',
        dest='source_dir',
        help='Source directory (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--splits', '-n',
        type=int,
        help='Number of splits (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Copy files instead of creating symbolic links'
    )
    
    args = parser.parse_args()
    
    # FIXED: Determine source directory
    source_dir = args.source_dir or args.source
    if not source_dir:
        # Default source directory
        source_dir = "/scratch/data/bikash_rs/vivek/dataset/MELD.Raw/output_repeated_splits_test"
        print(f"Using default source directory: {source_dir}")
    
    # FIXED: Determine number of splits - prioritize any provided value
    num_splits = 4  # Default
    if args.splits is not None:  # From positional argument
        num_splits = args.splits
    elif hasattr(args, 'splits') and getattr(args, 'splits') is not None:  # From --splits flag
        num_splits = getattr(args, 'splits')
    
    print("=== Video Directory Splitter ===")
    print(f"Source: {source_dir}")
    print(f"Number of splits: {num_splits}")
    print(f"Method: {'Copy files' if args.copy else 'Symbolic links'}")
    print()
    
    # Perform the split
    success = split_videos(source_dir, num_splits, args.copy)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()