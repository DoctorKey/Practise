gpu_id=6

seeds=(2021 2022 2023 2024 2025)

for seed in ${seeds[@]}
do
    python main.py --FT BP --num_sample 50   --seed $seed --rm_blocks layer1.1,layer1.2 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 100  --seed $seed --rm_blocks layer1.1,layer1.2 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 500  --seed $seed --rm_blocks layer1.1,layer1.2 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 1000 --seed $seed --rm_blocks layer1.1,layer1.2 --lr 0.0001 --gpu_id $gpu_id

    python main.py --FT BP --num_sample 50   --seed $seed --rm_blocks layer1.1,layer1.2,layer2.1 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 100  --seed $seed --rm_blocks layer1.1,layer1.2,layer2.1 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 500  --seed $seed --rm_blocks layer1.1,layer1.2,layer2.1 --lr 0.0001 --gpu_id $gpu_id
    python main.py --FT BP --num_sample 1000 --seed $seed --rm_blocks layer1.1,layer1.2,layer2.1 --lr 0.0001 --gpu_id $gpu_id
done



