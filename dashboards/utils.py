# dashboards/utils.py
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

def get_parquet_files(base_dir: str = "data") -> List[str]:
    """Auto-detect all parquet files in the data directory."""
    parquet_files = []
    if os.path.exists(base_dir):
        # Get all .parquet files recursively
        pattern = os.path.join(base_dir, "**", "*.parquet")
        parquet_files = glob.glob(pattern, recursive=True)
        # Sort by modification time (newest first)
        parquet_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return parquet_files

def get_model_files(base_dir: str = "artifacts") -> List[Tuple[str, str]]:
    """Auto-detect all trained model files in the artifacts directory.

    Returns:
        List of tuples (display_name, file_path)
    """
    model_files = []
    if os.path.exists(base_dir):
        # Get all .zip files in artifacts subdirectories
        pattern = os.path.join(base_dir, "**", "*.zip")
        zip_files = glob.glob(pattern, recursive=True)

        for file_path in zip_files:
            # Create display name from path with algorithm detection
            rel_path = os.path.relpath(file_path, base_dir)

            # Detect algorithm from path/filename
            algo_tag = ""
            if "ppo" in file_path.lower():
                algo_tag = "[PPO] "
            elif "dqn" in file_path.lower():
                algo_tag = "[DQN] "
            elif "a2c" in file_path.lower():
                algo_tag = "[A2C] "
            else:
                algo_tag = "[?] "

            # Add file size info
            try:
                size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 1)
                size_info = f" ({size_mb} MB)"
            except:
                size_info = ""

            display_name = f"{algo_tag}{rel_path.replace(os.sep, ' / ')}{size_info}"
            model_files.append((display_name, file_path))

        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)

    return model_files

def get_file_info(file_path: str) -> Optional[dict]:
    """Get basic info about a file."""
    if not os.path.exists(file_path):
        return None

    stat = os.stat(file_path)
    return {
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": stat.st_mtime,
        "exists": True
    }

def format_file_display(file_path: str, show_size: bool = True) -> str:
    """Format file path for display in dropdown."""
    info = get_file_info(file_path)
    if not info:
        return f"{file_path} (NOT FOUND)"

    # Get relative path for cleaner display
    try:
        rel_path = os.path.relpath(file_path)
        display_name = rel_path if len(rel_path) < len(file_path) else file_path
    except ValueError:
        display_name = file_path

    if show_size:
        return f"{display_name} ({info['size_mb']} MB)"
    else:
        return display_name