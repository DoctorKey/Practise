gpu_id=2
seeds=(2021 2022 2023 2034 2025)

for seed in ${seeds[@]}
do
    python main.py --model mobilenet_v2 --num_sample 500 --seed $seed --practise all --rm_blocks 1 --FT MiR --gpu_id $gpu_id 
    python main.py --model mobilenet_v2 --num_sample 500 --seed $seed --practise all --rm_blocks 2 --FT MiR --gpu_id $gpu_id 
done
