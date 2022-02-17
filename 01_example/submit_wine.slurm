#!/bin/bash
#SBATCH --nodes=1                  #total number of nodes for the job
#SBATCH --ntasks=1                 #how many copies of code to run 
#SBATCH --time=00:10:00            #amount of time for the whole job
#SBATCH --partition=standard       #the queue/partition to run on
#SBATCH --account=Rivanna-training #the account/allocation to use

module purge
module load anaconda/2020.11-py3.8 #load modules my job needs
python wine.py                     #command-line execution of my job

