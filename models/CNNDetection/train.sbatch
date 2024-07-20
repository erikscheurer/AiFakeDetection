#!/bin/bash 
#SBATCH --job-name=ResNet50

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# d-hh:mm:ss
#SBATCH --time=15:00:00

#SBATCH --partition=gpu_4 # single, gpu_4, gpu_8

##SBATCH --mem=4000

# GPU
##SBATCH --ntasks=40
#SBATCH --gres=gpu:2

# Info
#SBATCH --mail-user=mail@stud.uni-stuttgart.de
#SBATCH --mail-type=END,FAIL

# output name 
#SBATCH --output="output/ResNet_%j.out"

# same srun: srun --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --partition=gpu_4 --time=0-01:00:00 --pty bash

module load devel/python/3.11.7_intel_2021.4.0
export PYTHONPATH=~/AiFakeDetection:$PYTHONPATH
cd ~/AiFakeDetection
source ~/aacv_venv/bin/activate
unset DISPLAY

# test if pytorch can see the gpus
echo "CUDA is available: "
# python -c "import torch; print(torch.cuda.is_available())"
echo $PYTHONPATH

# run the script
source_list=()
target_list=()
echo "Running the script with target: $1"
for i in {0..7}
do
    # if i is not $1, add to source list, else add to target
    if [ $i -ne $1 ];
    then
        echo "Adding to source list: $i"
        source_list=(${source_list[@]} $i)
    else
        echo "Adding to target list: $i"
        target_list=(${target_list[@]} $i)
    fi

done
# convert to python list
source_list=$(printf "%s," "${source_list[@]}")
source_list="[${source_list::-1}]"
target_list=$(printf "%s," "${target_list[@]}")
target_list="[${target_list::-1}]"

echo "Source list: $source_list, Target list: $target_list"
echo "train.DANN_config.source=$source_list;train.DANN_config.target=$target_list"
python3 models/CNNDetection/train.py --extra_opts "train.DANN_config.source=$source_list;train.DANN_config.target=$target_list"