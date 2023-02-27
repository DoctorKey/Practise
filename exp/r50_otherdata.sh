gpu_id=2
seed=2021
datas=(50 500 1000 5000 -1)

for data in ${datas[@]}
do
    python main.py --model resnet50 --dataset imagenet_fewshot --num_sample $data --seed $seed --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.3
    python main.py --model resnet50 --dataset ADI_fewshot      --num_sample $data --seed $seed --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.3
    python main.py --model resnet50 --dataset CUB_sub          --num_sample $data --seed $seed --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.3
    python main.py --model resnet50 --dataset place365_sub     --num_sample $data --seed $seed --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.3
done
