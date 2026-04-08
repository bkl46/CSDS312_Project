import kagglehub

# Download directly to your preferred folder
custom_path = "./data"  # e.g., "./my_dataset" or "/home/user/data"

path = kagglehub.dataset_download(
    "brianblakely/top-100-songs-and-lyrics-from-1959-to-2019",
    output_dir=custom_path
)

print("Dataset downloaded to:", path)
