import kagglehub

path = "./raw"  

path = kagglehub.dataset_download(
    "brianblakely/top-100-songs-and-lyrics-from-1959-to-2019",
    output_dir=path
)

print("Dataset downloaded to:", path)
