import pickle5 as pickle
import time

from ge import DeepWalk
from ge import Node2Vec

from natuke_utils import metapath2vec
from natuke_utils import disturbed_hin
from natuke_utils import regularization
from natuke_utils import restore_hin
from natuke_utils import embedding_graph
from natuke_utils import true_restore

path = 'path-to-data-repository'
file_name = 'knn_results'

def execution(G, algorithm, split, iteration, edge_group, percentual_to_time):
    G_disturbed, train, test, hidden = disturbed_hin(G, split=split, random_state=(1 + iteration), edge_group=edge_group)
    G_found, hidden, train, test = true_restore(G_disturbed, hidden, train, test, percentual=0.0, edge_group=edge_group)
    
    if algorithm == 'deep_walk':
        for key, value in percentual_to_time.items():
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{key}')
            start_time = time.time()
            model_deep_walk = DeepWalk(G_found,walk_length=10,num_walks=80,workers=1)
            model_deep_walk.train(window_size=5,iter=3,embed_size=512)
            embeddings_deep_walk = model_deep_walk.get_embeddings()
            G_found = embedding_graph(G_found, embeddings_deep_walk)
            restored_df = restore_hin(G_found, test)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{key},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, key), index=False)
            G_found, hidden, train, test = true_restore(G_found, hidden, train, test, percentual=value, edge_group=edge_group)

    elif algorithm == 'node2vec':
        for key, value in percentual_to_time.items():
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{key}')
            start_time = time.time()
            model_node2vec = Node2Vec(G_found, walk_length = 10, num_walks = 80, p = 0.5, q = 1, workers = 1)
            model_node2vec.train(window_size=5,iter=3,embed_size=512)
            embeddings_node2vec = model_node2vec.get_embeddings()
            G_found = embedding_graph(G_found, embeddings_node2vec)
            restored_df = restore_hin(G_found, test)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{key},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, key), index=False)
            G_found, hidden, train, test = true_restore(G_found, hidden, train, test, percentual=value, edge_group=edge_group)

    elif algorithm == 'metapath2vec':
        for key, value in percentual_to_time.items():
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{key}')
            start_time = time.time()
            embeddings_metapath2vec = metapath2vec(G_found, dimensions=512)
            G_found = embedding_graph(G_found, embeddings_metapath2vec)
            restored_df = restore_hin(G_found, test)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{key},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, key), index=False)
            G_found, hidden, train, test = true_restore(G_found, hidden, train, test, percentual=value, edge_group=edge_group)
        
    elif algorithm == 'regularization':
        iterations = 30
        for key, value in percentual_to_time.items():
            print(f'Evaluation for {algorithm},{split},{iteration},{edge_group},{key}')
            start_time = time.time()
            G_found = regularization(G_found, iterations=iterations, mi=0.85)
            restored_df = restore_hin(G_found, test)
            with open("{}results/execution_time.txt".format(path), 'a') as f:
                f.write(f'{algorithm},{split},{iteration},{edge_group},{key},{(time.time() - start_time)}\n')
            restored_df.to_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, key), index=False)
            iterations = 20
            G_found, hidden, train, test = true_restore(G_found, hidden, train, test, percentual=value, edge_group=edge_group)


if __name__ == '__main__':
    network_name = "hin03-05"
    splits = [0.8]
    #edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
    edge_groups = ['doi_collectionType']
    #algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
    algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
    percentual_to_time = {'1st': 0.3, '2nd': 0.32, '3rd': 0.5, '4th': 0.0}

    with open("{}{}.gpickle".format(path, network_name), "rb") as fh:
        G = pickle.load(fh)

    # regularization
    for split in splits:
        for iteration in range(10):
            for edge_group in edge_groups:
                for algorithm in algorithms: 
                    execution(G, algorithm, split, iteration, edge_group, percentual_to_time)