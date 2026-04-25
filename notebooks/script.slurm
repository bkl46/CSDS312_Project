#!/bin/bash
#SBATCH -c 3           # 2 CPUs   
#SBATCH -A csds312 -p markov_cpu # PI account and partition
#SBATCH --mem=128G       # 8 GB of RAM 
#SBATCH -o  /mnt/vstor/courses/csds312/bkl46/logs/run-%j.out



echo "hello world"

module load spark/3.3.2

pip install pandas

echo "ok then"

spark-submit --master "local[8]" --driver-memory 8g main.py
