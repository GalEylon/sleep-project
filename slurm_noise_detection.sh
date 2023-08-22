#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=noiseDetection
#SBATCH --output=./logs_noise/R-%a.out
# #SBATCH --error=./logs_noise/output.log
#SBATCH --array=1-500   # Number of tasks/subjects
#SBATCH --mem 10GB

# Load the necessary modules or activate the virtual environment if required

# Change to the directory containing your Python script
# cd /mnt/home/geylon/code
cd /mnt/home/geylon/code/simons-sleep-ai/analysis

# Call your Python script and pass the subject ID as an argument
python noise_detection.py $SLURM_ARRAY_TASK_ID