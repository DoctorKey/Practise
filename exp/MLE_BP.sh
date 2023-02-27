gpu_id=0

seeds=(2021 2022 2023 2024 2025)
seed=2021
# for seed in ${seeds[@]}
while [ $seed -lt 2121 ]
do
    ((seed++))
    # echo $seed
    python main.py --FT BP --num_sample 50 --epoch 1000 --seed $seed --rm_blocks layer1.1,layer1.2,layer2.1 --lr 0.0001 --gpu_id $gpu_id
done