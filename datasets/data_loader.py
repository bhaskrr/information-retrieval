"""
Simple data loader utility for CSV files used in the information-retrieval project.

Provides a small wrapper around pandas.read_csv that resolves relative paths
against the repository's datasets directory (or a custom base_path).
"""
import pandas as pd
import os


class DataLoader:
    """
    DataLoader locates and loads CSV files.

    Parameters
    - base_path (str | None): Directory to resolve filenames against. If None,
      defaults to the directory containing this module.
    """

    def __init__(self, base_path=None):
        # If no base_path provided, use the directory where this file lives.
        if base_path is None:
            self.base_path = os.path.dirname(__file__)
        else:
            # Allow overriding the base path (useful for tests or different dataset locations)
            self.base_path = base_path

    def load(self, filename):
        """
        Load a CSV file by filename.

        - filename (str): Name of the CSV file (can include subdirectories relative to base_path).
        Returns a pandas.DataFrame.

        Example:
            loader = DataLoader()
            df = loader.load('my_dataset.csv')
        """
        # Build the full filesystem path to the requested file.
        path = os.path.join(self.base_path, filename)
        # Delegate actual CSV parsing to pandas.
        return pd.read_csv(path)