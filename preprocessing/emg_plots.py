import matplotlib.pyplot as plt
import os
import pandas as pd

# Function to normalize a column based on min and max values
def normalize_column(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)
    return normalized_column

# Function to load and normalize CSV files
def load_normalize(csv_file, sampling_rate):

    # Load the sensor data for a specific exercise (replace 'exercise1.csv' with your actual file name)
    df_sensor = pd.read_csv(csv_file, names=['Channel1', 'Channel2', 'Channel3','time_index'])

    # Convert the default numeric index to time using
    df_sensor['Time'] = df_sensor.index / sampling_rate
    return df_sensor

def csv_to_plot(df_sensor, out_dir, plot_file_name):

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15, 7))
    ax1.plot(df_sensor['Time'], df_sensor['Channel1'], label='Channel 1')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Value')
    ax1.set_title('Channel 1')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_sensor['Time'], df_sensor['Channel2'], label='Channel 2', color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Value')
    ax2.set_title('Channel 2')


    # Plot the data from df3 on the third subplot
    ax3.plot(df_sensor['Time'], df_sensor['Channel3'], label='Channel 3', color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Value')
    ax3.set_title('Channel 3')


    # Add a common title for both axes
    fig.suptitle('Plots of Channel 1, 2, and 3 of EMG for '+ plot_file_name+'.', fontsize=16)
    # Adjust layout
    plt.tight_layout()

    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'.png')

    #plt.show()



# Specify the input folder paths
input_folder = '../data/pre_processed/EMG/'
sampling_rate = 200

# Loop through each folder in the input folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    output_folder = '../plots/EMG/' + folder_name + '/'
    if not os.path.exists(output_folder):
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
                normalized_df = load_normalize(file_path, sampling_rate)
                csv_to_plot(normalized_df,output_folder, plot_file_name)
                counter += 1


print("Plot complete.")
