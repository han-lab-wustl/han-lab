
import os

def delete_tif_files_recursively(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

folder_path = r"\\storage1.ris.wustl.edu\ebhan\Active\Ziyi\Shared_Data\Halo_control"
delete_tif_files_recursively(folder_path)