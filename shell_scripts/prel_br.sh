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
singularity exec -e $image bash << EOF > $PBS_JOBID.out 2> $PBS_JOBID.err 

python3 "/home/svu/e0407728/My_FYP/tgn/train_self_supervised.py" -d reddit --use_memory --embedding_module time --prefix jodie_rnn_reddit --n_runs 10
python3 "/home/svu/e0407728/My_FYP/tgn/train_self_supervised.py" -d reddit --use_memory --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn_reddit --n_runs 10

