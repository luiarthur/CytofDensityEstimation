#!/bin/bash

# Run this script with: sbatch submit-job.sh

#SBATCH -p 128x24      # Partition name
#SBATCH -J sim-denest  # Job name
#SBATCH --mail-user=alui2@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -o out/slurm-job.out    # Name of stdout output file
#SBATCH -N 1         # Total number of nodes requested (128x24/Instructional only)
#SBATCH -t 72:00:00  # Run Time (hh:mm:ss) - 72 hours (optional)
#SBATCH --mem=42G # Memory to be allocated PER NODE


date 
echo "Starting jobs!"
gcc --version
echo $PYTHONPATH

# make compile
echo "Compile model ..."
PYTHONPATH="" && . venv/bin/activate && export PYTHONPATH="" && which python
PYTHONPATH="" && . venv/bin/activate && export PYTHONPATH="" && time venv/bin/python compile_stan_model.py
echo "Finished compiling model ..."
make sim-study

echo "Done submitting jobs."
echo "Job submission time:"
date

echo "Jobs are now running. A message will be printed and emailed when jobs are done."

wait

make send-results

echo "Jobs are completed."
echo "Job completion time:"
date
