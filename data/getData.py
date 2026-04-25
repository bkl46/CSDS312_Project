import kagglehub

path = "./raw"  

#path = kagglehub.dataset_download(
    #"brianblakely/top-100-songs-and-lyrics-from-1959-to-2019",
    #output_dir=path
#)

print("Dataset downloaded to:", path)




# Download latest version
path = kagglehub.dataset_download(
    "devdope/900k-spotify",
    output_dir="./spotify-dataset"
)

print("Path to full dataset files:", path)
