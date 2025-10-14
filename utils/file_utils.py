import os
import logging
from pathlib import Path


def cleanup_temp_files(directory, patterns=None, logger=None):
    """
    Clean up temporary files in a directory.
    
    Parameters:
    -----------
    directory : str or Path
        Directory to clean up
    patterns : list of str, optional
        List of file patterns to delete (e.g., ['*.tmp', '*.temp', '*~'])
        If None, defaults to common temporary file patterns
    logger : logging.Logger, optional
        Logger instance for logging cleanup operations
    
    Returns:
    --------
    int
        Number of files deleted
    """
    if patterns is None:
        # Common temporary file patterns
        patterns = [
            '*.tmp',
            '*.temp',
            '*~',
            '*.swp',
            '*.swo',
            '.DS_Store',
            '*.cache',
            '__pycache__',
            '*.pyc',
            '*.bak'
        ]
    
    directory = Path(directory)
    if not directory.exists():
        if logger:
            logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    deleted_count = 0
    
    for pattern in patterns:
        if pattern == '__pycache__':
            # Handle __pycache__ directories
            for pycache_dir in directory.rglob('__pycache__'):
                try:
                    for file in pycache_dir.iterdir():
                        file.unlink()
                    pycache_dir.rmdir()
                    deleted_count += 1
                    if logger:
                        logger.info(f"Deleted directory: {pycache_dir}")
                except Exception as e:
                    if logger:
                        logger.error(f"Error deleting {pycache_dir}: {e}")
        else:
            # Handle file patterns
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        if logger:
                            logger.info(f"Deleted temp file: {file_path}")
                    except Exception as e:
                        if logger:
                            logger.error(f"Error deleting {file_path}: {e}")
    
    if logger:
        logger.info(f"Cleaned up {deleted_count} temporary files from {directory}")
    
    return deleted_count


def cleanup_multiple_dirs(directories, patterns=None, logger=None):
    """
    Clean up temporary files in multiple directories.
    
    Parameters:
    -----------
    directories : list of str or Path
        List of directories to clean up
    patterns : list of str, optional
        List of file patterns to delete
    logger : logging.Logger, optional
        Logger instance for logging cleanup operations
    
    Returns:
    --------
    dict
        Dictionary mapping directory paths to number of files deleted
    """
    results = {}
    for directory in directories:
        count = cleanup_temp_files(directory, patterns, logger)
        results[str(directory)] = count
    
    return results
