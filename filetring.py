#### EBI plots####
import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy.signal import detrend

# Define function for applying wavelet filter and detrending
def apply_wavelet_filter(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    thresholded_coeffs = [pywt.threshold(c, 0.1, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(thresholded_coeffs, 'db4')
    # Truncate denoised signal to match original signal length
    original_length = len(signal)
    denoised_signal = denoised_signal[:original_length]
    return denoised_signal


def apply_detrending(signal):
    detrended_signal = detrend(signal)
    return detrended_signal


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
    #normalized_df_C = df_sensor_C.apply(normalize_column)

    # Apply normalization to each column in the DataFrame
    #normalized_df_Q = df_sensor_Q.apply(normalize_column)

    # Convert the default numeric index to time using
    df_sensor_C['Time'] = df_sensor_C.index / sampling_rate
    df_sensor_Q['Time'] = df_sensor_Q.index / sampling_rate

    return df_sensor_C, df_sensor_Q

def filtering_plot(df_c, out_dir, plot_file_name, x1,y1,y2,y3,title1,title2):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 6))

    # Plot the data from df_c on the first subplot
    sns.lineplot(data=df_c, x=x1, y=y1, ax=ax1, label='Original Signal')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    #ax1.set_title('Original Signal')
    ax1.legend(loc='upper right')

    # Plot the data from df_q on the second subplot
    sns.lineplot(data=df_c, x=x1, y=y2, color='violet', ax=ax2, label=title1)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    #ax2.set_title(title1)
    ax2.legend(loc='upper right')

    # Plot the data from df_q on the second subplot
    sns.lineplot(data=df_c, x=x1, y=y3, color='green', ax=ax3, label=title2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    #ax3.set_title(title2)
    ax3.legend(loc='upper right')

    # Add a common title for both axes
    fig.suptitle('Plots of Filtering and Normalization for ' + plot_file_name + '.', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    #plt.grid(True)
    # Save the figure with subplots
    plt.savefig(out_dir + 'plot_' + plot_file_name + '.png')

    # Show the plot
    plt.show()

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

def frame_to_time(frame_number):
    """
    Convert a video frame number to a timestamp.

    Parameters:
        frame_number (int): The frame number.
        frame_rate (float): The frame rate of the video (frames per second).

    Returns:
        float: The corresponding timestamp.
    """
    video_frame_rate = 60  # Video frame rate in frames per second
    return frame_number / video_frame_rate


def elan_sync(normalized_C,normalized_Q, sampling_rate, exercise_name):

  # Get the start and end times from ELAN annotations for the corresponding exercise
  df = pd.read_csv('./data/1005_Y_2023-09-11.csv')
  df = df[df.ANNOTATION_VALUE == exercise_name]

  # Calculate the offset
  offset = df.START_TIME.min()

  # Apply the offset to start and end times using lambda functions
  df['START_TIME'] = df['START_TIME'].apply(lambda x: x - offset)
  df['END_TIME'] = df['END_TIME'].apply(lambda x: x - offset)
  # Calculate the duration of ELAN annotations
  elan_duration = df['END_TIME'].iloc[len(df['END_TIME'])-1] - df['START_TIME'].iloc[0]
  total_csamples = len(normalized_C)
  total_qsamples = len(normalized_Q)
  elan_samples = int(elan_duration * sampling_rate)

  print(elan_samples)
  print(len(normalized_C))
  print(len(normalized_Q))

  # Initial calculation for the first block
  start_qindex = int((total_qsamples - elan_samples) // 2)
  start_cindex = int((total_csamples - elan_samples) // 2)

  normalized_C = normalized_C.iloc[start_cindex:]
  normalized_Q = normalized_Q.iloc[start_qindex:]

  stop_value = df['END_TIME'].max()
  # Find the index corresponding to the stop value
  normalized_C = normalized_C[normalized_C['Time'] <= stop_value ]
  normalized_Q = normalized_Q[normalized_Q['Time'] <= stop_value ]

  #print(elan_duration)
  #print(elan_samples)
  #print(len(normalized_C))
  #print(len(normalized_Q))

  # Select the last 'num' items from the DataFrame
  #normalized_C = normalized_C.tail(elan_samples)
  #normalized_Q = normalized_Q.tail(elan_samples)
  return normalized_C,normalized_Q
# Define function to subtract continuous representation from original signal
def subtract_continuous_representation(original_signal, wavelet_signal):
    subtracted_signal = original_signal - wavelet_signal
    return subtracted_signal


def signal_proc(c_signal_df, q_signal_df):
  # Extract bioimpedance signal values as a NumPy array
  bioimpedance_csignal = c_signal_df['Value'].values
  print(bioimpedance_csignal.shape)
  # Assuming 'bioimpedance_signal' is your raw bioimpedance signal
  denoised_csignal = apply_wavelet_filter(bioimpedance_csignal)
  detrended_csignal = apply_detrending(denoised_csignal)
  print(denoised_csignal.shape)
  #subtracted_csignal = subtract_continuous_representation(bioimpedance_csignal, detrended_csignal)

  # Add denoised and detrended signals back to the DataFrame
  c_signal_df['denoised_csignal'] = denoised_csignal
  c_signal_df['detrended_csignal'] = detrended_csignal

  # Extract bioimpedance signal values as a NumPy array
  bioimpedance_qsignal = q_signal_df['Value'].values
  # Assuming 'bioimpedance_signal' is your raw bioimpedance signal
  denoised_qsignal = apply_wavelet_filter(bioimpedance_qsignal)
  detrended_qsignal = apply_detrending(denoised_qsignal)
  #subtracted_qsignal = subtract_continuous_representation(bioimpedance_qsignal, denoised_qsignal)
  # Add denoised and detrended signals back to the DataFrame
  q_signal_df['denoised_qsignal'] = denoised_qsignal
  q_signal_df['detrended_qsignal'] = detrended_qsignal

  return c_signal_df, q_signal_df


# Function to extract features from bioimpedance signals
def extract_bioimpedance_features(c_signal, q_signal,out_dir,plot_file_name):
    print(c_signal.shape)
    print(q_signal.shape)
    # Calculate relevant features
    c_bio_signal = c_signal.detrended_csignal.values
    c_impedance_magnitude = np.abs(c_bio_signal)
    c_phase_angle = np.angle(c_bio_signal)
    c_resistance = np.real(c_bio_signal)

    q_bio_signal = q_signal.detrended_qsignal.values
    q_impedance_magnitude = np.abs(q_bio_signal)
    q_phase_angle = np.angle(q_bio_signal)
    q_resistance = np.real(q_bio_signal)


    c_signal['C_Impedance_Magnitude']= c_impedance_magnitude
    c_signal['C_Phase_Angle']= c_phase_angle
    #c_signal['C_Resistance']= c_resistance

    q_signal['Q_Impedance_Magnitude']= q_impedance_magnitude
    q_signal['Q_Phase_Angle']= q_phase_angle
    #q_signal['Q_Resistance']= q_resistance

    # Visualize the extracted features (optional)
    plt.figure(figsize=(10, 3))
    plt.plot(c_signal.Time, c_signal['C_Impedance_Magnitude'], label='Impedance Magnitude')
    plt.plot(c_signal.Time, c_signal['C_Phase_Angle'], label='Phase Angle', color ='violet')
    plt.xlabel('Time')
    plt.ylabel('Feature Value')
    plt.title('Extracted Features from Calf Bioimpedance Signals: '+ plot_file_name)
    plt.legend(loc='upper right')
    plt.grid(True)
    # Adjust layout
    plt.tight_layout()
    #plt.grid(True)
    # Save the figure with subplots
    plt.savefig(out_dir + 'plot_' + plot_file_name + '_C.png')

    # Show the plot
    plt.show()


    # Visualize the extracted features (optional)
    plt.figure(figsize=(10, 3))
    plt.plot(q_signal.Time, q_signal['Q_Impedance_Magnitude'], label='Impedance Magnitude')
    plt.plot(q_signal.Time, q_signal['Q_Phase_Angle'], label='Phase Angle', color ='violet')
    #plt.plot(q_signal.Time, q_signal['Q_Resistance'], label='Resistance')
    plt.xlabel('Time')
    plt.ylabel('Feature Value')
    plt.title('Extracted Features from Quadrecips Bioimpedance Signals: '+ plot_file_name)
    plt.legend(loc='upper right')
    plt.grid(True)
    # Adjust layout
    plt.tight_layout()
    #plt.grid(True)
    # Save the figure with subplots
    plt.savefig(out_dir + 'plot_' + plot_file_name + '_Q.png')

    # Show the plot
    plt.show()

    # Display the extracted features DataFrame
    print("Extracted Features:")
    print(c_signal.head())

    #return features_df
def statistical_features(impedance_data):

    # Assuming 'impedance_data' is a numpy array containing impedance signals

    # Mean Impedance
    mean_impedance = np.mean(impedance_data)

    # Median Impedance
    median_impedance = np.median(impedance_data)

    # Standard Deviation of Impedance
    std_deviation_impedance = np.std(impedance_data)

    # Variability Metrics
    coefficient_of_variation = std_deviation_impedance / mean_impedance
    interquartile_range = np.percentile(impedance_data, 75) - np.percentile(impedance_data, 25)
    range_impedance = np.max(impedance_data) - np.min(impedance_data)

    # Print the computed features
    print("Mean Impedance:", mean_impedance)
    print("Median Impedance:", median_impedance)
    print("Standard Deviation of Impedance:", std_deviation_impedance)
    print("Coefficient of Variation:", coefficient_of_variation)
    print("Interquartile Range:", interquartile_range)
    print("Range of Impedance:", range_impedance)


# Specify the input folder paths
input_folder = '../data/pre_processed/EBI/'
sampling_rate = 390
folder_name ='EBI'
# Loop through each folder in the input folder
output_folder = './plots/' + folder_name + '_elan/'
out_feat_dir= './plots/' + folder_name + '_features/'
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(out_feat_dir, exist_ok=True)

# Loop through each folder in the input folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    output_folder = '../plots/EBI/' + folder_name + '/'
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the current item is a folder
    if os.path.isdir(folder_path):
        print(f"Processing folder: {input_folder}")
        file_names = os.listdir(input_folder)
        sorted_file_names = sorted(file_names)
        # Loop through each CSV file in the current folder
        counter = 0
        while counter < len(sorted_file_names):
            file_name = sorted_file_names[counter]
            print(file_name)
            if file_name != '.ipynb_checkpoints':

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
                    #
                    #csv_to_plot(normalized_C, normalized_Q, output_folder, plot_file_name)
                    normalized_C,normalized_Q = elan_sync(normalized_C,normalized_Q, sampling_rate, split_filename[0])
                    csv_to_plot(normalized_C, normalized_Q, output_folder, plot_file_name)
                    #normalized_C, normalized_Q = signal_proc(normalized_C, normalized_Q)
                    #filtering_plot(normalized_C[['Time','Value','denoised_csignal','detrended_csignal']],
                    #output_folder, plot_file_name+'_C', 'Time','Value','denoised_csignal','detrended_csignal','Denoised Signal','Detrended Signal')

                    #filtering_plot(normalized_Q[['Time','Value','denoised_qsignal','detrended_qsignal']],
                    #output_folder, plot_file_name+'_Q', 'Time','Value','denoised_qsignal','detrended_qsignal','Denoised Signal','Detrended Signal')

                    # Extract features from the bioimpedance data
                    #extract_bioimpedance_features(normalized_C,normalized_Q, out_feat_dir, plot_file_name)

                    counter += 2

        print("Plot complete.")
