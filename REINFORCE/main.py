import os
import random
random.seed(2024)
import sys
sys.path.append('../')
import pickle
import torch
import config
import pickle as pkl
import trainer_kg, trainer_llm #, trainer_nc, trainer_gc, trainer_nc_GAT
from joint_train import joint_train

def load_processed_triplets(file_path):
    triplets = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行首行尾的括号和换行符
            clean_line = line.strip()[1:-1]
            # 按逗号分割字符串
            triplet = tuple(clean_line.split(','))
            # 添加到列表中
            triplets.append(triplet)
    return triplets

def replace_file_content(src_file, dest_file, path):
    """
    读取源文件src_file的内容并写入目标文件dest_file，从而替换目标文件的内容。
    
    参数:
    src_file (str): 源文件的路径。
    dest_file (str): 目标文件的路径。
    """
    try:
        with open(path+src_file, 'r') as file:
            content = file.read()
        
        with open(path+dest_file, 'w') as file:
            file.write(content)

        print(f"文件 '{src_file}' 的内容已成功复制到 '{dest_file}'。")
    except Exception as e:
        print(f"发生错误: {e}")


def main(args):
    datasetname=args.datasetname
    torch.manual_seed(args.random_seed)
    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.graph_task == 'kg':
        path = f'../knowledge_graph_tasks/embedding_based/benchmarks/{datasetname}/'
        replace_file_content('org_train2id.txt', 'train2id.txt',path)
        hyper_edges = pickle.load(open(f'../preprocess/{datasetname}_new_hg_triplets.pkl', 'rb'))
        hyper_edges_ns = pickle.load(open(f'../preprocess/{datasetname}_new_hg_triplets_negative_samples.pkl', 'rb'))
        graph_trainee = trainer_kg.Trainer(args, hyper_edges, hyper_edges_ns)


    # elif args.graph_task == 'nc':
    #     # add new graph
    #     data_path = f'../DropEdge/enhanced_data/'
    #     with open(os.path.join(data_path, "ind.{}.{}".format(datasetname.lower(), f'graph-top-{args.topk}')), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             hypergraph = pkl.load(f, encoding='latin1')
    #         else:
    #             hypergraph = pkl.load(f)
    #     #nc_trainee = trainer_nc.Trainer(args, hypergraph)
    #     #nc_trainee.train()

    #     if(args.type=='GAT'):
    #         graph_trainee = trainer_nc_GAT.Trainer(args, hypergraph)
    #         # graph_trainee.train()
    #     else:
    #         graph_trainee = trainer_nc.Trainer(args, hypergraph)
    #         # graph_trainee.train()
    # elif args.llm_task == 'gc':
    #     # add new graph
    #     data_path = f'../gmixup/'
    #     with open(os.path.join(data_path, "{}_{}".format(datasetname, 'new_graph.pkl')),'rb' ) as f:
    #         hypergraph = pkl.load(f)
    #     graph_trainee = trainer_gc.Trainer(args, hypergraph)
    #     # graph_trainee.train()

    if args.llm_task == 'qa':
        llm_KG_triplets = pickle.load(open(f'../LLM_tasks/{datasetname}_all_subgraphs_dict.pkl', 'rb'))
        llm_trainee = trainer_llm.Trainer(args, llm_KG_triplets)

    joint_train(llm_trainee, graph_trainee)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    print(args)
    print(unparsed)
    main(args)
