gpu_id=4

python main.py --model resnet50 --dataset ADI_fewshot --num_sample -1 --seed 2021 --epoch 16000 --gpu_id $gpu_id --rm_blocks layer1.1,layer1.2,layer2.3

python main.py --model mobilenet_v2 --dataset ADI_fewshot --num_sample -1 --seed 2021 --gpu_id $gpu_id --epoch 16000 --rm_blocks features.3,features.5

