#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=myjob
#SBATCH --output=subject_%a.out
#SBATCH --array=1-5   # Number of tasks/subjects
#SBATCH --mem 1GB

# Command to process each subject
subject_id=${SLURM_ARRAY_TASK_ID}
echo "Processing subject ${subject_id}..."
# Add your specific commands here using the subject ID as needed
