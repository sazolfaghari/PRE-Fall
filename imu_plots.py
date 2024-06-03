######### IMUs plots######

import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to normalize a column based on min and max values
def normalize_column(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)

    return normalized_column

# Function to load and normalize CSV files
def load_normalize(csv_filepath):

    # Load the sensor data for a specific exercise (replace 'exercise1.csv' with your actual file name)
    df_sensor = pd.read_csv(csv_filepath)
    sampling_rate = int(df_sensor.sample_rate.iloc[1])
    

    # Apply normalization to each column in the DataFrame
    df_sensor.ax_1 = normalize_column(df_sensor.ax_1)
    df_sensor.ax_2 = normalize_column(df_sensor.ax_2)
    df_sensor.ay_1 = normalize_column(df_sensor.ay_1)
    df_sensor.ay_2 = normalize_column(df_sensor.ay_2) 
    df_sensor.az_1 = normalize_column(df_sensor.az_1)  
    df_sensor.az_2 = normalize_column(df_sensor.az_2)

    df_sensor.gx_1 = normalize_column(df_sensor.gx_1) 
    df_sensor.gx_2 = normalize_column(df_sensor.gx_2) 
    df_sensor.gy_1 = normalize_column(df_sensor.gy_1) 
    df_sensor.gy_2 = normalize_column(df_sensor.gy_2) 
    df_sensor.gz_1 = normalize_column(df_sensor.gz_1) 
    df_sensor.gz_2 = normalize_column(df_sensor.gz_2) 

    # Convert the default numeric index to time using
    df_sensor['Time'] = df_sensor.index / sampling_rate

    return df_sensor

def csv_to_plot(df_sensor, out_dir, plot_file_name):

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 7))
    ax1.plot(df_sensor['Time'], df_sensor['ax_1'], label='X-axis')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration')
    ax1.set_title('X-axis')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_sensor['Time'], df_sensor['ay_1'], label='Y-axis', color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Y-axis')


    # Plot the data from df3 on the third subplot
    ax3.plot(df_sensor['Time'],df_sensor['az_1'], label='Z-axis', color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration')
    ax3.set_title('Z-axis')


    # Add a common title for both axes
    fig.suptitle('Top Accelerometer data in x,y,z axes for '+ plot_file_name+'.', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'_top_acc.png')


    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 7))
    ax1.plot(df_sensor['Time'], df_sensor['gx_1'], label='X-axis')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angular Velocity')
    ax1.set_title('X-axis')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_sensor['Time'],df_sensor['gy_1'], label='Y-axis', color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity')
    ax2.set_title('Y-axis')


    # Plot the data from df3 on the third subplot
    ax3.plot(df_sensor['Time'],df_sensor['gz_1'], label='Z-axis', color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Angular Velocity')
    ax3.set_title('Z-axis')


    # Add a common title for both axes
    fig.suptitle('Top Gyroscope data in x,y,z axes for '+ plot_file_name+'.', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'_top_gyr.png')


    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 7))
    ax1.plot(df_sensor['Time'],df_sensor['ax_2'], label='X-axis')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration')
    ax1.set_title('X-axis')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_sensor['Time'],df_sensor['ay_2'], label='Y-axis', color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Y-axis')


    # Plot the data from df3 on the third subplot
    ax3.plot(df_sensor['Time'],df_sensor['az_2'], label='Z-axis', color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration')
    ax3.set_title('Z-axis')


    # Add a common title for both axes
    fig.suptitle('Bottom Accelerometer data in x,y,z axes for '+ plot_file_name+'.', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'_bot_acc.png')

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 7))
    ax1.plot(df_sensor['Time'],df_sensor['gx_2'], label='X-axis')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angular Velocity')
    ax1.set_title('X-axis')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_sensor['Time'],df_sensor['gy_2'], label='Y-axis', color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity')
    ax2.set_title('Y-axis')


    # Plot the data from df3 on the third subplot
    ax3.plot(df_sensor['Time'],df_sensor['gz_2'], label='Z-axis', color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Angular Velocity')
    ax3.set_title('Z-axis')

    # Add a common title for both axes
    fig.suptitle('Bottom Gyroscope data in x,y,z axes for '+ plot_file_name+'.', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'_bot_gyr.png')




# Specify the input folder paths
input_folder = '../data/pre_processed/IMU/'


# Loop through each folder in the input folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    output_folder = '../plots/IMU/' + folder_name + '/'
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the current item is a folder
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        file_names = os.listdir(folder_path)
        sorted_file_names = sorted(file_names)
        # Loop through each CSV file in the current folder
        counter = 0
        while counter < len(sorted_file_names):
            file_name = sorted_file_names[counter]
            split_filename = file_name.split('_')
            if file_name.endswith('.csv'):

                s_num = split_filename[1]
                s_age = split_filename[2].split('.')[0]

                if len(split_filename) == 4:
                  retest_file_name =split_filename[-1].split('.')[0]
                  file_name= split_filename[0] + '_' + s_num +'_' + s_age + '_' + retest_file_name
                  plot_file_name= split_filename[0] + '_' +s_num +'_' + s_age + '_' + retest_file_name
                else:
                  file_name= split_filename[0] + '_' + s_num +'_' + s_age
                  plot_file_name= split_filename[0] + '_' +s_num +'_' + s_age

                file_path = os.path.join(folder_path, file_name +'.csv')

                # Add your custom processing logic here
                normalized_df = load_normalize(file_path)
                csv_to_plot(normalized_df,output_folder, plot_file_name)
                counter += 1
print("Plot complete.")
