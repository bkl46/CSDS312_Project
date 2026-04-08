
# CSDS 312 Project: Text Mining Popular Songs

Declan Boyle, Brandon Lee, Sam Lin, Andrew Lowe, Alyssa Murphy, Garrett Pavlick, Maura Schorr


## Preparing Environment

this repo is in /mnt/vstor/course/csds312/bkl46/Proj/ which should be available to everyone

to use make sure to load the correct python version: module load Python/3.12.3-GCCcore-13.3.0

then activate virtual environmnet: source venv/bin/activate 


if running locally, after cloning repo make sure to have the right python version and install the requirements: pip install -r requirements.txt

## Prepare Data

### Get Data

enter the data/ directory and run getData.py: 

cd data/ \
python getData.py \

should add the csv's from kaggle into /data/raw

### Preproces the raw data

enter the notebooks/ directory and run data_processing.py

from root: \
cd notebooks/ \
python data_processing.py \

this should add the processed csv to data/raw

Right now the script is removing unecessary columns, normalizing years/decade, converting lyrics to tokens and calculating some preliminary features (token count, unique words, sentiment, etc.)


The script can be run with arguments, 
- --input "path_to_alternative_data", if we need to run on other data
- --skip_lang_detect <True/False>, if True, will group songs by language
- --output "path_to_alternative_output_destination"
