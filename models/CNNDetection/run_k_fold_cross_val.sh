
for i in {0..7}
do
    sbatch models/CNNDetection/train.sbatch $i
done