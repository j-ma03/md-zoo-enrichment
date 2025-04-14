import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *




class Processor:
    """
    A class to process IMU data from a Zoodata Dataloader object.
    """
    def __init__(self, zoodata_dl):
        self.zoodata_dl = zoodata_dl
        # self.output_dir = output_dir
        # self.data_name = data_name
        self.result_with_counts = None
    
    def process_imu_data(self, threshold):
        imu_data = np.array(self.zoodata_dl.raw_data)
        imu_data_avg = []

        for row in imu_data:
            # avg = np.abs(np.mean(row[-3:]))
            avg = (np.mean(row[-3:]))
            imu_data_avg.append(avg)

        self.imu_data_avg = np.array(imu_data_avg)

        # ### Get average acceleration per interval of time
        time = imu_data[:, :6]
        imu_data_avg_time = np.column_stack((time, imu_data_avg))

        # Get unique time stamps and their indices
        unique_times, unique_indices = np.unique(imu_data_avg_time[:, :6], axis=0, return_inverse=True)

        # Calculate the average acceleration for each unique time stamp
        avg_acceleration = np.array([imu_data_avg_time[unique_indices == i, 6].mean() for i in range(len(unique_times))])

        # Combine unique times with their corresponding average accelerations
        result = np.column_stack((unique_times, avg_acceleration))
        print(result.shape)

        # Define threshold for significant acceleration
        threshold = threshold

        # Get time data (year, month, day, hour) and acceleration data
        time_data = imu_data[:, :6]  # Time data (year, month, day, hour, minute, second)
        accel_data = imu_data_avg    # Average acceleration data

        # Combine time data with acceleration data
        combined_data = np.column_stack((time_data, accel_data))

        # Group by hour (year, month, day, hour)
        hours = combined_data[:, :4]
        unique_hours = np.unique(hours, axis=0)

        # Count minutes with significant acceleration per hour
        minutes_per_hour = []

        for hour in unique_hours:
            # Get all data for this hour
            mask = np.all(combined_data[:, :4] == hour, axis=1)
            hour_data = combined_data[mask]
            
            # Get unique minutes in this hour
            unique_minutes = np.unique(hour_data[:, 4])
            
            # Count minutes with acceleration > threshold
            minutes_above_threshold = 0
            for minute in unique_minutes:
                minute_mask = hour_data[:, 4] == minute
                minute_data = hour_data[minute_mask]
                if np.any(minute_data[:, 6] > threshold):
                    minutes_above_threshold += 1
            
            # Store hour with count of minutes above threshold
            minutes_per_hour.append(np.append(hour, minutes_above_threshold))

        # Convert to numpy array
        result_with_counts = np.array(minutes_per_hour)

        # Save result as CSV
        # Convert result_with_counts to integers before saving
        result_with_counts = result_with_counts.astype(int)
        print(result_with_counts)
        self.result_with_counts = result_with_counts
        return result_with_counts
    
    def save_results(self, output_dir, data_name):        
        np.savetxt(f'{output_dir}/{data_name}_enrichment_data.csv', 
                self.result_with_counts, 
                delimiter=',', 
                fmt='%d,%d,%d,%d,%d',
                header='Year,Month,Day,Hour,Minutes of Interaction', 
                comments='')
        
        return f'{output_dir}/{data_name}_enrichment_data.csv'


    # ### Graph



    # plt.plot(range(len(imu_data_avg)), imu_data_avg)
    # plt.xlabel('Index')
    # plt.ylabel('Average acceleration (m/s^2)')
    # plt.title(f'Line Graph of {DATA_NAME} IMU Data')
    # plt.show()


    

    # # Create plot
    # time_labels = [f'{int(year)}-{int(month):02d}-{int(day):02d} {int(hour):02d}:00' 
    #             for year, month, day, hour in result_with_counts[:, :4]]
    # counts = result_with_counts[:, 4]

    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(counts)), counts)
    # plt.xlabel('Time')
    # plt.ylabel('Minutes with acceleration > threshold')
    # plt.title('Minutes of interaction per hour')
    # plt.xticks(range(len(counts)), time_labels, rotation=90)
    # plt.tight_layout()
    # plt.show()




    