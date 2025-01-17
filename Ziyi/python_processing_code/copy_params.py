import os
import shutil

def copy_params_mat_with_custom_structure(src_dir, dst_dir):
    # Get the last folder name from the source directory
    last_folder_name = os.path.basename(src_dir.rstrip("\\/"))
    target_root_dir = os.path.join(dst_dir, last_folder_name)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "params.mat":
                # Calculate the relative path from the source folder
                rel_path = os.path.relpath(root, src_dir)
                
                # Create the corresponding directory in the destination
                target_dir = os.path.join(target_root_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the params.mat file to the destination directory
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} to {dst_file}")

# Example usage
source_directory = r"E:\Ziyi\Data\250116_ZH\250116_ZH_000_001"
destination_directory = r"\\storage1.ris.wustl.edu\ebhan\Active\Ziyi\Shared_Data\VTA_mice\E277"

copy_params_mat_with_custom_structure(source_directory, destination_directory)


