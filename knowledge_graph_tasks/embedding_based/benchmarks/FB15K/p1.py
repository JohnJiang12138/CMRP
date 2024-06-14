import csv

# Path to the input file
input_file_path = '/home/users/jiangwenyuan/projects/Reasoning-KG-main/GraphPrompt_code/knowledge_graph_tasks/embedding_based/benchmarks/FB15K/relation2id.txt'
output_file_path = '/home/users/jiangwenyuan/projects/Reasoning-KG-main/GraphPrompt_code/knowledge_graph_tasks/embedding_based/benchmarks/FB15K/relation2text.txt'
output_file2_path = '/home/users/jiangwenyuan/projects/Reasoning-KG-main/GraphPrompt_code/knowledge_graph_tasks/embedding_based/benchmarks/FB15K/text2relationid.txt'

with open(input_file_path, mode='r', encoding='utf-8') as infile, \
     open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile, \
         open(output_file2_path, mode='w', encoding='utf-8') as outfile2:
    
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')
    writer2 = csv.writer(outfile2, delimiter='\t')

    # Skip the first line
    next(reader, None)

    for row in reader:
        # 处理关系，替换 '/' 和 '_'
        relation = row[0].replace('/', ' ').replace('_', ' ')
        id = row[1]

        # 写入第一个输出文件：原始关系 -> 处理后的关系
        writer.writerow([row[0], relation])

        # 写入第二个输出文件：处理后的关系 -> ID
        writer2.writerow([relation, id])