import os
import shutil

def delete_pert_files_and_folders(
        cache_dir="cache",
        pert_adj_storage_type="evasion_global_adj",
        pert_attr_storage_type="evasion_global_attr"
):
    # Construct file names (including .json extension)
    adj_file_path = os.path.join(cache_dir, f"{pert_adj_storage_type}.json")
    attr_file_path = os.path.join(cache_dir, f"{pert_attr_storage_type}.json")

    # Construct directory paths
    adj_dir_path = os.path.join(cache_dir, pert_adj_storage_type)
    attr_dir_path = os.path.join(cache_dir, pert_attr_storage_type)

    # List all paths we want to check
    paths_to_delete = [adj_file_path, attr_file_path, adj_dir_path, attr_dir_path]

    for path in paths_to_delete:
        if os.path.exists(path):
            if os.path.isfile(path):
                # Delete the file
                os.remove(path)
                print(f"Deleted file: {path}")
            elif os.path.isdir(path):
                # Delete the directory and its contents
                shutil.rmtree(path)
                print(f"Deleted directory: {path}")
        else:
            print(f"Not found: {path}")