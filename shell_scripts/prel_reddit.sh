#!/bin/bash
#PBS -P Project
#PBS -j oe
#PBS -N out
#PBS -q volta_gpu
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=12:00:00

cd #PBS_o_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

image="/home/svu/e0407728/SIF/edge-hpc_v0.1.sif"
singularity exec -e $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID 

python3 "/home/svu/e0407728/My_FYP/tgn/train_self_supervised.py" --use_memory --prefix tgn-attn-reddit --n_runs 3

