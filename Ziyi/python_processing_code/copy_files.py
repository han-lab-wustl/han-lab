import os
import shutil

def copy_files_with_structure(src, dst, day):
    # Extract the original folder name from the source path
    folder_name = os.path.basename(src)
    
    # Create the destination folder structure
    day_folder = os.path.join(dst, str(day))
    src_folder_structure = os.path.join(day_folder, folder_name)
    vr_folder = os.path.join(day_folder, "vr")
    
    # Check if day folder exists, and create 'vr' folder if needed
    if not os.path.exists(day_folder):
        os.makedirs(day_folder)
        
    if not os.path.exists(vr_folder):
        os.makedirs(vr_folder)  # Ensure the 'vr' folder exists in the correct location
    
    # Preserve folder structure of src inside the day folder
    if not os.path.exists(src_folder_structure):
        os.makedirs(src_folder_structure)
    
    # Copy all files from src to the preserved folder name inside day_folder
    for file_name in os.listdir(src):
        src_file = os.path.join(src, file_name)
        dst_file = os.path.join(src_folder_structure, file_name)
        
        if os.path.isfile(src_file):  # Ensure it's a file
            shutil.copy(src_file, dst_file)
    
    print(f"Files copied to {src_folder_structure}, with an empty 'vr' folder inside {day_folder}.")

# Example usage
source_folder = r"F:\ziyi\250312_ZH\250312_ZH_000_000"
destination_folder = r"E:\Ziyi\Data\GrabNE_mice\E274"
day_variable = "30"  # Example day value

copy_files_with_structure(source_folder, destination_folder, day_variable)




