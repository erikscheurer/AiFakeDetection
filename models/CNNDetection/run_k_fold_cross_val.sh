
for i in {1..8}
do
    sbatch models/CNNDetection/train.sbatch $i
done