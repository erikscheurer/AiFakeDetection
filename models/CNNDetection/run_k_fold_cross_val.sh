
for i in {0..7}
do  
    echo "Running fold $i"
    sbatch models/CNNDetection/train.sbatch $i
done