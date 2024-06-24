
for i in {0..7}
do
    sbatch models/F3Net/train.sbatch $i
done