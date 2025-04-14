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
        raw_data: NDArray[np.float32],
        start_time: str = None,
        end_time: str = None
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
    
    def get_first_timestamp(self) -> str:  
        """
        Get the first timestamp from the raw data.
        """
        return self.raw_data[0, :-3]
    
    def get_last_timestamp(self) -> str:
        """
        Get the last timestamp from the raw data.
        """
        return self.raw_data[-1, :-3]
    
    def crop(self, start_time: str, end_time: str) -> 'Dataloader':
        """
        Crop the data to a specific time range.
        """
        # Convert start and end times to datetime objects
        start_datetime = pd.to_datetime(start_time, format="%Y %m %d %H %M %S")
        end_datetime = pd.to_datetime(end_time, format="%Y %m %d %H %M %S")
        print(f"Start time: {start_datetime}, End time: {end_datetime}")

        # Ensure the first 6 columns are numeric
        raw_data_numeric = self.raw_data[:, :6].astype(int)
        
        # Create timestamps from the individual time components
        timestamps = []
        for row in raw_data_numeric:
            year, month, day, hour, minute, second = row
            timestamp_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            timestamps.append(pd.to_datetime(timestamp_str))
        
        timestamps = pd.Series(timestamps)
        
        # Filter the raw data based on the time range
        # Note: We're excluding the start and end times as per your example
        mask = (timestamps > start_datetime) & (timestamps < end_datetime)
        cropped_data = self.raw_data[mask]
        
        return Dataloader(self.metadata, cropped_data)