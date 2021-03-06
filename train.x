#!/bin/bash
#PBS -N LJW
#PBS -l nodes=gnode4:ppn=4
#PBS -l walltime=1000:00:00


cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

source activate ljw
python -u ./train_exe.py 1>./exp_final_os10/output.txt 2>./exp_final_os10/error.txt
