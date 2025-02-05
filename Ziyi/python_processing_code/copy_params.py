import os
import shutil

def copy_params_mat_with_custom_structure(src_dir, dst_dir):
    # Get the last folder name from the source directory
    last_folder_name = os.path.basename(src_dir.rstrip("\\/"))
    target_root_dir = os.path.join(dst_dir, last_folder_name)
    print(f"Target root directory: {target_root_dir}")  # Debug: Check target directory structure
    
    # Walk through the source directory
    found_file = False
    for root, dirs, files in os.walk(src_dir):
        print(f"Visiting: {root}")  # Debug: Show current directory being scanned
        for file in files:
            if file == "params.mat":
                found_file = True
                # Calculate the relative path from the source folder
                rel_path = os.path.relpath(root, src_dir)
                print(f"Found params.mat at {root}, relative path: {rel_path}")  # Debug: Confirm finding file and path
                
                # Create the corresponding directory in the destination
                target_dir = os.path.join(target_root_dir, rel_path)
                print(f"Creating target directory: {target_dir}")  # Debug: Check target directory creation
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the params.mat file to the destination directory
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} to {dst_file}")  # Debug: Confirm file copy
    if not found_file:
        print("No params.mat file found in the source directory.")  # Debug: Alert if no file found

# Example usage
source_directory = r"E:\Ziyi\Data\250128_ZH\250128_ZH_000_001"
destination_directory = r"E:\Ziyi\Data\Halo_control\srz\e245\6"

copy_params_mat_with_custom_structure(source_directory, destination_directory)



