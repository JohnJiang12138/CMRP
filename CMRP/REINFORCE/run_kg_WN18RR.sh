CUDA_VISIBLE_DEVICES=4 python -u main.py --network_type lstm --agent_optim adam --agent_lr 0.0035 \
    --agent_optim sgd --entropy_coeff 0.0001 --graph_task 'kg' --llm_task 'qa' --datasetname 'WN18RR' --batch_size 64 \
    --alpha 1.0 --beta 0.2 --gamma 0.1 --alpha1 10.0 --beta1 10 \
    --LP_model 'transh' --max_epoch 10 --train_times 500