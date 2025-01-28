from typing import List, Any
from numpy.typing import NDArray
import numpy as np
import pandas as pd

class Dataloader():
    """
    Base Dataloader class that performs some basic functionalities:
        -  Read the data files as a .csv
        -  Store basic file properties given on the first line of the data files
        -  Retrieve the individual coordinate points as an array of tuples
    """
    def __init__(
        self,
        metadata: List[Any],
        raw_data: NDArray[np.float32]
    ) -> None:

        # Stores metadata of the data file
        self.metadata: List[Any] = metadata

        # Store tuples of (x, y, z) coordinates read from the file
        self.raw_data: NDArray[np.float32] = raw_data

    # Construct a Dataloader class given a data file
    @staticmethod
    def read_file(filename: str) -> 'Dataloader':
        # Read data file as CSV & construct a dataframe
        df = pd.read_csv(filename)

        # Extract metadata from dataframe column titles
        metadata = Dataloader._read_metadata(df)

        # Extract raw data (coordinates) from data file
        raw_data: NDArray[np.float32] = Dataloader._read_raw_data(df)

        return Dataloader(metadata, raw_data)

    # Reads the data file's metadata from dataframe
    @staticmethod
    def _read_metadata(df: pd.DataFrame) -> List[Any]:
        return list(df.columns.values)

    # Reads (x, y, z) coordinates from each row
    @staticmethod
    def _read_raw_data(df: pd.DataFrame) -> NDArray[np.float32]:
        raw_data = []

        # Extract all the coordinates from dataframe
        for _, row in df.iterrows():
            raw_data.append([float(coord) for coord in row.values[0].split()])

        return np.array(raw_data)