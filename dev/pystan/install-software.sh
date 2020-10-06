#!/bin/bash

# Run this script with: sbatch submit-job.sh

#SBATCH -p 128x24      # Partition name
#SBATCH -J install-software # Job name
#SBATCH --mail-user=alui2@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -o out/slurm-job.out    # Name of stdout output file
#SBATCH -N 1         # Total number of nodes requested (128x24/Instructional only)
#SBATCH -n 8         # number of mpi tasks
#SBATCH -t 1:00:00  # Run Time (hh:mm:ss) - 1 hours (optional)
#SBATCH --mem=8G # Memory to be allocated PER NODE

date 

PYTHONPATH="" && . venv/bin/activate && pip install -r requirements.txt
PYTHONPATH="" && . venv/bin/activate && python compile_stan_model.py

date
