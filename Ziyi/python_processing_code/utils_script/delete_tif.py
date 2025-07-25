
import os

def delete_tif_files_recursively(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

folder_path = r"E:\Ziyi\Data\VTA_mice\hrz\E277\16"
delete_tif_files_recursively(folder_path)