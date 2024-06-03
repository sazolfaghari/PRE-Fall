import os

#root_folder = '../data/pre_processed/EBI/'
root_folder = '../plots/EBI/'


# List of prefixes to filter file names
#prefixes_to_change = ['CR3-a', 'CR3-b', 'CR4-a', 'CR4-b']
#prefixes_to_change = ['Ch1', 'Ch2']
prefixes_to_change = ['plot_Ch1', 'plot_Ch2']

# Iterate over folders in the root folder
for current_folder, subfolders, files in os.walk(root_folder):
    # Check if the current folder contains files with the specified prefixes
    target_files = [filename for filename in files if any(filename.startswith(prefix) for prefix in prefixes_to_change)]

    # Add the target files to the list and rename them
    for filename in target_files:
        # Construct the new filename by removing the dash
        new_filename = filename.replace('h', 'H')

        # Full paths for old and new filenames
        old_filepath = os.path.join(current_folder, filename)
        new_filepath = os.path.join(current_folder, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)

print("File renaming completed.")

# Iterate over folders in the root folder
# for current_folder, subfolders, files in os.walk(root_folder):
#     # Iterate over files in the current folder
#     for filename in files:
#         # Check if the file name has any of the specified prefixes
#         if any(filename.startswith(prefix) for prefix in prefixes_to_change):
#             # Construct the new filename by removing the dash
#             new_filename = filename.replace('-', '')

#             # Full paths for old and new filenames
#             old_filepath = os.path.join(current_folder, filename)
#             new_filepath = os.path.join(current_folder, new_filename)

#             # Rename the file
#             os.rename(old_filepath, new_filepath)

# print("File renaming completed.")

