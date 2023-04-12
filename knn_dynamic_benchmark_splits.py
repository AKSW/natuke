import pickle5 as pickle
import time
import pandas as pd
import multiprocess
from tqdm import tqdm
from copy import deepcopy

from ge import DeepWalk
from ge import Node2Vec

from natuke_utils import metapath2vec
from natuke_utils import regularization
from natuke_utils import embedding_graph
from natuke_utils import get_knn_data
from natuke_utils import run_knn

path = 'your-data-path'
file_name = 'knn_results'

def restore_hin_split(G, cutted_df, edge_group, 
                      n_jobs=-1, k=-1, node_feature='node', neighbor_feature='neighbor', 
                      group_feature='group', embedding_feature='f'):
    # function
    def process(start, end, G, df, edge_group, return_dict, thread_id):
        value_thread = df.loc[start:(end-1)]
        restored_dict_thread = {'true': [], 'restored': [], 'edge_type': []}
        for _, row in tqdm(value_thread.iterrows(), total=value_thread.shape[0]):
            edge_to_add = edge_group.split('_')
            edge_to_add[0] = row[node_feature]
            edge_to_add = [row[node_feature] if e == G.nodes[row[node_feature]][group_feature] and row[node_feature] != edge_to_add[0] else e for e in edge_to_add]
            knn_data, knn_nodes = get_knn_data(G, row[node_feature], embedding_feature=embedding_feature)
            knn_nodes['type'] = knn_nodes[0].apply(lambda x: G.nodes[x][group_feature])
            knn_data = knn_data[knn_nodes['type'].isin(edge_to_add)]
            knn_nodes = knn_nodes[knn_nodes['type'].isin(edge_to_add)]
            edge_to_add[1] = run_knn(k, G, row, knn_data, knn_nodes, embedding_feature=embedding_feature)
            restored_dict_thread['true'].append([row[node_feature], row[neighbor_feature]])
            restored_dict_thread['restored'].append(edge_to_add)
            restored_dict_thread['edge_type'].append(edge_group)
        for key in restored_dict_thread.keys():
            _key = key + str(thread_id)
            return_dict[_key] = (restored_dict_thread[key])
    # split threads
    def split_processing(n_jobs, G, df, edge_group, return_dict):
        split_size = round(len(df) / n_jobs)
        threads = []                                                                
        for i in range(n_jobs):                                                 
            # determine the indices of the list this thread will handle             
            start = i * split_size                                                  
            # special case on the last chunk to account for uneven splits           
            end = len(df) if i+1 == n_jobs else (i+1) * split_size                
            # create the thread
            threads.append(                                                         
                multiprocess.Process(target=process, args=(start, end, G, df, edge_group, return_dict, i)))
            threads[-1].start() # start the thread we just created                  

        # wait for all threads to finish                                            
        for t in threads:
            t.join()

    if n_jobs == -1:
        n_jobs = multiprocess.cpu_count()
    restored_dict = {'true': [], 'restored': [], 'edge_type': []}
    return_dict = multiprocess.Manager().dict()

    split_processing(n_jobs, G, cutted_df, edge_group, return_dict)
    return_dict = dict(return_dict)
    for thread_key in restored_dict.keys():
        for job in range(n_jobs):
            for res in return_dict[thread_key + str(job)]:
                restored_dict[thread_key].append(res)
    return pd.DataFrame(restored_dict)

def load_graph_train_test(iteration, edge_group, evaluation_stage):
    with open(f"{path}splits/kg_{edge_group}_{iteration}_{evaluation_stage}.gpickle", "rb") as fh:
        G_found = pickle.load(fh)
    train = pd.read_csv(f'{path}splits/train_{edge_group}_{iteration}_{evaluation_stage}.csv')
    test = pd.read_csv(f'{path}splits/test_{edge_group}_{iteration}_{evaluation_stage}.csv')
    return G_found, train, test

# if a new graph is generated this function can split it according to the splits file
def new_graph_splitter(G, test, extra_cut_from='nubbe', node_from_feature='node_from', type_feature='edge_group'):
    G_disturbed = deepcopy(G)
    for _, row in test.iterrows():
        neighbors_list = list(G_disturbed.neighbors(row['node']))
        neighbors_hidden = []
        for neighbor in neighbors_list:
            if G_disturbed.nodes[neighbor][node_from_feature] == extra_cut_from:
                neighbors_hidden.append({'neighbor': neighbor, 'edge_group': G_disturbed[row['node']][neighbor][type_feature]})
                G_disturbed.remove_edge(row['node'],neighbor)
    return G_disturbed

def execution(algorithm, split, iteration, edge_group, evaluation_stages):
    # if a new graph is generated load it here
    # with open(f"{path}/your_graph_name.gpickle", "rb") as fh:
    #     G = pickle.load(fh)

    if algorithm == 'deep_walk':
        for evaluation_stage in evaluation_stages:
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{evaluation_stage}')
            G_found, train, test = load_graph_train_test(iteration, edge_group, evaluation_stage)
            # if new graph
            # G_found = new_graph_splitter(G, test)
            start_time = time.time()
            model_deep_walk = DeepWalk(G_found, walk_length=10, num_walks=80, workers=1)
            model_deep_walk.train(window_size=5, iter=3, embed_size=512)
            embeddings_deep_walk = model_deep_walk.get_embeddings()
            G_found = embedding_graph(G_found, embeddings_deep_walk)
            restored_df = restore_hin_split(G_found, test, edge_group)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{evaluation_stage},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, evaluation_stage), index=False)

    elif algorithm == 'node2vec':
        for evaluation_stage in evaluation_stages:
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{evaluation_stage}')
            G_found, train, test = load_graph_train_test(iteration, edge_group, evaluation_stage)
            # if new graph
            # G_found = new_graph_splitter(G, test)
            start_time = time.time()
            model_node2vec = Node2Vec(G_found, walk_length = 10, num_walks = 80, p = 0.5, q = 1, workers = 1)
            model_node2vec.train(window_size=5,iter=3,embed_size=512)
            embeddings_node2vec = model_node2vec.get_embeddings()
            G_found = embedding_graph(G_found, embeddings_node2vec)
            restored_df = restore_hin_split(G_found, test, edge_group)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{evaluation_stage},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, evaluation_stage), index=False)

    elif algorithm == 'metapath2vec':
        for evaluation_stage in evaluation_stages:
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{evaluation_stage}')
            G_found, train, test = load_graph_train_test(iteration, edge_group, evaluation_stage)
            # if new graph
            # G_found = new_graph_splitter(G, test)
            start_time = time.time()
            embeddings_metapath2vec = metapath2vec(G_found, dimensions=512)
            G_found = embedding_graph(G_found, embeddings_metapath2vec)
            restored_df = restore_hin_split(G_found, test, edge_group)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{evaluation_stage},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, evaluation_stage), index=False)
        
    elif algorithm == 'regularization':
        iterations = 30
        for evaluation_stage in evaluation_stages:
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{evaluation_stage}')
            G_found, train, test = load_graph_train_test(iteration, edge_group, evaluation_stage)
            # if new graph
            # G_found = new_graph_splitter(G, test)
            start_time = time.time()
            G_found = regularization(G_found, iterations=iterations, mi=0.85)
            restored_df = restore_hin_split(G_found, test, edge_group)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{evaluation_stage},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, evaluation_stage), index=False)
            iterations = 20


if __name__ == '__main__':
    # just to be compatible with execution time codes
    split = 0.8
    #edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
    edge_groups = ['doi_bioActivity']
    #algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
    algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
    evaluation_stages = ['1st', '2nd', '3rd', '4th']

    # regularization
    for iteration in range(10):
        for edge_group in edge_groups:
            for algorithm in algorithms: 
                execution(algorithm, split, iteration, edge_group, evaluation_stages)