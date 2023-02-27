gpu_id=6
datas=(50 100 500 1000)

for data in ${datas[@]}
do
    python main.py --num_sample $data --seed 2021 --epoch 2000 --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1
    python main.py --num_sample $data --seed 2021 --epoch 2000 --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2
    python main.py --num_sample $data --seed 2021 --epoch 2000 --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.2
    python main.py --num_sample $data --seed 2021 --epoch 2000 --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.2,layer3.3
    python main.py --num_sample $data --seed 2021 --epoch 2000 --gpu_id $gpu_id --FT MiR --rm_blocks layer1.1,layer1.2,layer2.2,layer3.3,layer3.2
done