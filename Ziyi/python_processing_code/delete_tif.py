
import os

def delete_tif_files_recursively(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".sbx"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

folder_path = r"E:\Ziyi\Data\E228_sparseDA"
delete_tif_files_recursively(folder_path)