#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -A rivanna-training

module purge
module load singularity/3.7.1
module load tensorflow/2.4.1

singularity run --nv $CONTAINERDIR/tensorflow-2.4.1.sif Py_easy_TensorFlow.py

