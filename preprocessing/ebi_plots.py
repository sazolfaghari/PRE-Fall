#### EBI plots####
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to normalize a column based on min and max values
def normalize_column(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)
    return normalized_column

# Function to load and normalize CSV files
def load_normalize(c_csv_file, q_csv_file, sampling_rate):

    # Load the sensor data for a specific exercise (replace 'exercise1.csv' with your actual file name)
    df_sensor_C = pd.read_csv(c_csv_file, names=['Value'])
    df_sensor_Q = pd.read_csv(q_csv_file, names=['Value'])

    # Apply normalization to each column in the DataFrame
    normalized_df_C = df_sensor_C.apply(normalize_column)

    # Apply normalization to each column in the DataFrame
    normalized_df_Q = df_sensor_Q.apply(normalize_column)

    # Convert the default numeric index to time using
    normalized_df_C['Time'] = normalized_df_C.index / sampling_rate
    normalized_df_Q['Time'] = normalized_df_Q.index / sampling_rate

    return normalized_df_C, normalized_df_Q

def csv_to_plot(df_c, df_q, out_dir, plot_file_name):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    ax1.plot(df_c['Time'], df_c['Value'], label='Calf')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Value')
    ax1.set_title('Calf')

    # Plot the data from df2 on the second subplot
    ax2.plot(df_q['Time'], df_q['Value'], label='Quadriceps', color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Value')
    ax2.set_title('Quadriceps')
    # Add a common title for both axes
    fig.suptitle('Plots of Calf and Quadriceps for '+ plot_file_name+'.', fontsize=16)
    # Adjust layout
    plt.tight_layout()
    # Save the figure with subplots
    plt.savefig(out_dir  + 'plot_'+plot_file_name+'.png')
    #plt.show()



# Specify the input folder paths
input_folder = '../data/pre_processed/EBI/'
sampling_rate = 390

# Loop through each folder in the input folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    output_folder = '../plots/EBI/' + folder_name + '/'
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
                s_num = split_filename[2]
                s_age = split_filename[3].split('.')[0]
                if len(split_filename) == 5:
                  retest_file_name =split_filename[4].split('.')[0]
                  file_name_C= split_filename[0] + '_C_' + s_num +'_' + s_age + '_' + retest_file_name
                  file_name_Q= split_filename[0] + '_Q_' + s_num +'_' + s_age + '_' + retest_file_name
                  plot_file_name= split_filename[0] + '_' +s_num +'_' + s_age + '_' + retest_file_name
                else:
                  file_name_C= split_filename[0] + '_C_' + s_num +'_' + s_age
                  file_name_Q= split_filename[0] + '_Q_' + s_num +'_' + s_age
                  plot_file_name= split_filename[0] + '_' +s_num +'_' + s_age

                file_path_c = os.path.join(folder_path, file_name_C +'.csv')
                file_path_q = os.path.join(folder_path, file_name_Q +'.csv')

                # Add your custom processing logic here
                normalized_C, normalized_Q = load_normalize(file_path_c, file_path_q, sampling_rate)
                csv_to_plot(normalized_C, normalized_Q,output_folder, plot_file_name)
                counter += 2


print("Plot complete.")
