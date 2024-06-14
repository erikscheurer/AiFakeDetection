
for i in {1..8}
do
    sbatch models/F3Net-main/train.sbatch $i
done