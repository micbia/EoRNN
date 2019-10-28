#!/bin/bash -x
#SBATCH --account=hpo22
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --output=./outputs/output_eornn_gpu4.%j
#SBATCH --error=./errors/error_eornn_gpu4.%j
#SBATCH --time=24:00:00

#SBATCH --partition=mem192

module load Python
module load scikit
module load Keras
module load TensorFlow
module load CUDA

python TrainingEoRNN.py
