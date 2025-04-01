#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_crowd_counting
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/train_%J.out
#BSUB -e batch_output/train_%J.err
# -- end of LSF options --

source /dtu/3d-imaging-center/courses/conda/conda_init.sh
conda activate env-02510

export CUDA_VISIBLE_DEVICES=0

python -u src/crowd_counting_with_diffusion_models/ddpm_train.py --config-name exp_1.yaml
