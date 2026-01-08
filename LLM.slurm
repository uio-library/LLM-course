#!/bin/bash

#SBATCH --account=ec443
#SBATCH --nodes=1
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Load necessary modules and activate environment
module list
module purge
module load Python/3.12.3-GCCcore-13.3.0 Python-bundle-PyPI/2024.06-GCCcore-13.3.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load IPython/8.28.0-GCCcore-13.3.0 JupyterLab/4.2.5-GCCcore-13.3.0
module load lxml/5.3.0-GCCcore-13.3.0

module list

# get GPU info
nvidia-smi

# Run the Python script that is given as an argument
python3 ${@}

# Confirm job completion
echo "Job completed"