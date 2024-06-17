#!/bin/bash 
#SBATCH --job-name=automotive_CV

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# d-hh:mm:ss
#SBATCH --time=0-20:00:00

#SBATCH --partition=gpu_4 # single, gpu_4, gpu_8

##SBATCH --mem=4000

# GPU
##SBATCH --ntasks=40
#SBATCH --gres=gpu:2

# Info
##SBATCH --mail-user=my@email.com
##SBATCH --mail-type=END,FAIL

# output name 
#SBATCH --output="output/F3Net_%j.out"

# same srun: srun --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --partition=gpu_4 --time=0-01:00:00 --pty bash

module load devel/python/3.11.7_intel_2021.4.0
export PYTHONPATH=~/AiFakeDetection:$PYTHONPATH
cd ~/AiFakeDetection
source ~/aacv_venv/bin/activate


# test if pytorch can see the gpus
echo "CUDA is available: "
python -c "import torch; print(torch.cuda.is_available())"
echo $PYTHONPATH

# run the script
echo "Running the script with leave out: $1"
python3 models/F3Net/train.py  --extra_opts config.train.dataset.leave_out=$1