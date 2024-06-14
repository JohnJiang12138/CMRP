
for seed in 1314
do 
    CUDA_VISIBLE_DEVICES=0  python -u ./src/enhance.py --data_path . --model GIN --dataset IMDB-BINARY \
        --lr 0.01 --gmixup True --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
        --aug_ratio 0.15 --aug_num 10  --ge USVT
done