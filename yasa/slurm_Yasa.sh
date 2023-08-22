#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=YasaNSRRScoring
#SBATCH --output=eeg_%a.out
#SBATCH --array=1-2   # Number of tasks/subjects
#SBATCH --mem 10GB

# Load the necessary modules or activate the virtual environment if required

# Change to the directory containing your Python script
# cd /mnt/home/geylon/code
cd /mnt/home/geylon/code/simons-sleep-ai/analysis/yasa

# Call your Python script and pass the subject ID as an argument
python Yasa_NSRR_py.py $SLURM_ARRAY_TASK_ID