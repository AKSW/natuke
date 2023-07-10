import pandas as pd
import networkx as nx
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('no GPU')

"""
*************************************
*                                   *
*                                   *
*   UTILS FOR BENCHMARK EXECUTION   *
*                                   *
*                                   *
*************************************
"""

def disturbed_hin(G, split=0.6, random_state=None, extra_cut_from='nubbe', edge_group='doi_bioActivity', node_from_feature='node_from', type_feature='edge_group', group_feature='group'):
    """
    G: hin;
    split: percentage to be cut from the hin;
    random_state: ;
    extra_cut_from: edges from the origin that needs to be cut but not restored;
    edge_group: string of type of edge to be added for restoration;
    type_feature: feature name of edge_type on your hin.
    """
    def keep_left(x, G):
        edge_split = x['type'].split('_')
        if G.nodes[x['node']][group_feature] != edge_split[0]:
            x['node'], x['neighbor'] = x['neighbor'], x['node']
        return x
    # prepare data for type counting
    edges = list(G.edges)
    edge_types = [G[edge[0]][edge[1]][type_feature] for edge in edges]
    
    edges = pd.DataFrame(edges)
    edges = edges.rename(columns={0: 'node', 1: 'neighbor'})
    edges['type'] = edge_types
    edges = edges.apply(keep_left, G=G, axis=1)
    edges_group = edges.groupby(by=['type'], as_index=False).count().reset_index(drop=True)

    # preparar arestas para eliminar
    edges = edges.sample(frac=1, random_state=random_state).reset_index(drop=True)
    edges_group = edges_group.rename(columns={'node': 'count', 'neighbor': 'to_cut_count'})
    edges_group['to_cut_count'] = edges_group['to_cut_count'].apply(lambda x:round(x * split))
    train, test = {}, {}
    for _, row in edges_group.iterrows():
        if row['type'] == edge_group:
            train[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[row['to_cut_count']:].reset_index(drop=True)
            test[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[:row['to_cut_count']-1].reset_index(drop=True)
                    
    G_disturbed = deepcopy(G)
    hidden = {'node': [], 'neighbor_group': []}
    for tc_df in test.values():
        for _, row in tc_df.iterrows():
            neighbors_list = list(G_disturbed.neighbors(row['node']))
            neighbors_hidden = []
            has_cut = False
            for neighbor in neighbors_list:
                if G_disturbed.nodes[neighbor][node_from_feature] == extra_cut_from:
                    has_cut = True
                    neighbors_hidden.append({'neighbor': neighbor, 'edge_group': G_disturbed[row['node']][neighbor][type_feature]})
                    G_disturbed.remove_edge(row['node'],neighbor)
            if has_cut:
                hidden['node'].append(row['node'])
                hidden['neighbor_group'].append(neighbors_hidden)
    return G_disturbed, train, test, pd.DataFrame(hidden)

def regularization(G, dim=512, embedding_feature: str = 'embedding', iterations=15, mi=0.85):
    nodes = []
    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        if 'f' not in G.nodes[node]:
            G.nodes[node]['f'] = np.array([0.0]*dim)
        elif embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0
        nodes.append(node)
    pbar = tqdm(range(0, iterations))
    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0
        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0
            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']
                w /= np.sqrt(G.degree[neighbor])
                f_new = f_new + w*G.nodes[neighbor]['f']
                sum_w = sum_w + w
            if sum_w == 0.0: sum_w = 1.0
            f_new /= sum_w
            G.nodes[node]['f'] = f_new*1.0
            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)
            energy = energy + np.linalg.norm(f_new-f_old)
        iteration = iteration + 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)
    return G

def get_knn_data(G, node, embedding_feature: str = 'f'):
    knn_data, knn_nodes = [], []
    for node in nx.non_neighbors(G, node):
        if embedding_feature in G.nodes[node]:
            knn_data.append(G.nodes[node][embedding_feature])
            knn_nodes.append(node)
    return pd.DataFrame(knn_data), pd.DataFrame(knn_nodes)

def run_knn(k, G_restored, row, knn_data, knn_nodes, node_feature='node', embedding_feature='f'):
    if k == -1:
        k = knn_data.shape[0]
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(knn_data)
    indice = knn.kneighbors(G_restored.nodes[row[node_feature]][embedding_feature].reshape(-1, 512), return_distance=False)
    return [knn_nodes[0].iloc[indice[0][i]] for i in range(k)]

import multiprocess
def restore_hin(G, cutted_dict, n_jobs=-1, k=-1, node_feature='node', neighbor_feature='neighbor', group_feature='group', embedding_feature='f'):
    # function
    def process(start, end, G, key, value, return_dict, thread_id):
        value_thread = value.loc[start:(end-1)]
        restored_dict_thread = {'true': [], 'restored': [], 'edge_type': []}
        for _, row in tqdm(value_thread.iterrows(), total=value_thread.shape[0]):
            edge_to_add = key.split('_')
            edge_to_add[0] = row[node_feature]
            edge_to_add = [row[node_feature] if e == G.nodes[row[node_feature]][group_feature] and row[node_feature] != edge_to_add[0] else e for e in edge_to_add]
            knn_data, knn_nodes = get_knn_data(G, row[node_feature], embedding_feature=embedding_feature)
            knn_nodes['type'] = knn_nodes[0].apply(lambda x: G.nodes[x][group_feature])
            knn_data = knn_data[knn_nodes['type'].isin(edge_to_add)]
            knn_nodes = knn_nodes[knn_nodes['type'].isin(edge_to_add)]
            edge_to_add[1] = run_knn(k, G, row, knn_data, knn_nodes, embedding_feature=embedding_feature)
            restored_dict_thread['true'].append([row[node_feature], row[neighbor_feature]])
            restored_dict_thread['restored'].append(edge_to_add)
            restored_dict_thread['edge_type'].append(key)
        for key in restored_dict_thread.keys():
            _key = key + str(thread_id)
            return_dict[_key] = (restored_dict_thread[key])
    # split threads
    def split_processing(n_jobs, G, key, value, return_dict):
        split_size = round(len(value) / n_jobs)
        threads = []                                                                
        for i in range(n_jobs):                                                 
            # determine the indices of the list this thread will handle             
            start = i * split_size                                                  
            # special case on the last chunk to account for uneven splits           
            end = len(value) if i+1 == n_jobs else (i+1) * split_size                
            # create the thread
            threads.append(                                                         
                multiprocess.Process(target=process, args=(start, end, G, key, value, return_dict, i)))
            threads[-1].start() # start the thread we just created                  

        # wait for all threads to finish                                            
        for t in threads:
            t.join()

    if n_jobs == -1:
        n_jobs = multiprocess.cpu_count()
    restored_dict = {'true': [], 'restored': [], 'edge_type': []}
    return_dict = multiprocess.Manager().dict()

    for key, value in cutted_dict.items():
        split_processing(n_jobs, G, key, value, return_dict)
        return_dict = dict(return_dict)
        for thread_key in restored_dict.keys():
            for job in range(n_jobs):
                for res in return_dict[thread_key + str(job)]:
                    restored_dict[thread_key].append(res)
    return pd.DataFrame(restored_dict)

def ml_restore_hin(G, train, test, edge_group='doi_bioActivity', neighbor_feature='neighbor', node_feature='node', embedding_feature='f', min_delta=0.00001, patience=10, epochs=1000, embedding_size=512):
    def getX(G, train, test, edge_group='doi_bioActivity', node_feature='node', embedding_feature='f'):
        X_train = []
        for _, row in train[edge_group][node_feature].iteritems():
            X_train.append(G.nodes[row][embedding_feature])
        X_test = []
        for _, row in test[edge_group][node_feature].iteritems():
            X_test.append(G.nodes[row][embedding_feature])
        return np.array(X_train), np.array(X_test)

    def getY(train, test, neighbor_feature='neighbor'):
        classes = pd.Series(train[edge_group][neighbor_feature].to_list() + test[edge_group][neighbor_feature].to_list()).unique()
        classes_codes = {}
        for index, class_name in enumerate(classes):
            classes_codes[class_name] = index
        train[edge_group]['class_code'] = train[edge_group][neighbor_feature].apply(lambda x: classes_codes[x])
        test[edge_group]['class_code'] = test[edge_group][neighbor_feature].apply(lambda x: classes_codes[x])
        num_classes = len(classes)
        y_train = to_categorical(train[edge_group]['class_code'], num_classes=num_classes)
        y_test = to_categorical(test[edge_group]['class_code'], num_classes=num_classes)
        return y_train, y_test, num_classes, classes

    def get_mlp(dimX, dimY):
        model = Sequential()
        model.add(Dense(dimX, activation='relu', input_shape=(dimX,)))
        model.add(Dense(dimX, activation='relu'))
        model.add(Dense(dimX, activation='relu'))
        model.add(Dense(dimY, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    X_train, X_test = getX(G, train, test, edge_group=edge_group, node_feature=node_feature, embedding_feature=embedding_feature)
    print(X_train.shape)
    y_train, _, num_classes, classes = getY(train, test, neighbor_feature=neighbor_feature)

    K.clear_session()
    model = get_mlp(embedding_size, num_classes)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=min_delta)
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, callbacks=[callback])
    y_pred = model.predict(X_test)
    restored_df = {'true': [], 'restored': [], 'edge_type': []}
    for index, row in test[edge_group].iterrows():
        zip_pred_classes = sorted(zip(y_pred[index], classes), reverse=True)
        restored_df['true'].append([row[node_feature], row[neighbor_feature]])
        restored_df['restored'].append([row[node_feature], [class_name for _, class_name in zip_pred_classes]])
        restored_df['edge_type'].append(edge_group)            
    return pd.DataFrame(restored_df)

def embedding_graph(G, embeddings, embedding_feature='f'):
    for key, value in embeddings.items():
        G.nodes[key][embedding_feature] = value
    return G

def true_restore(G, hidden, train, test, percentual=1.0, edge_group='doi_bioActivity', node_feature='node', neighbor_group_feature='neighbor_group', neighbor_feature='neighbor', edge_group_feature='edge_group'):
    G_found = deepcopy(G)
    adding_df = hidden.loc[0:round(hidden.shape[0] * percentual)-1]
    remaining_df = hidden.loc[round(hidden.shape[0] * percentual):hidden.shape[0]-1]
    df_train, df_test = train[edge_group], test[edge_group]
    for index, row in adding_df.iterrows():
        df_train = pd.concat([df_train, df_test[df_test[node_feature] == row[node_feature]]])
        df_test = df_test.drop(df_test[df_test[node_feature] == row[node_feature]].index)
        for to_add in row[neighbor_group_feature]:
            G_found.add_edge(row[node_feature], to_add[neighbor_feature], edge_type=to_add[edge_group_feature])
    
    train[edge_group], test[edge_group] = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    return G_found, remaining_df.reset_index(drop=True), train, test


from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph

# 'bioActivity', 'molType', 'collectionSpecie', 'collectionSite', 'collectionType', 'molecularMass', 'monoisotropicMass', 'cLogP', 'tpsa', 
# 'numberOfLipinskiViolations', 'numberOfH_bondAcceptors', 'numberOfH_bondDonors', 'numberOfRotableBonds', 'molecularVolume', 'smile'

def metapath2vec(graph, dimensions = 512, num_walks = 1, walk_length = 100, context_window_size = 10, 
                           num_iter = 1, workers = 1, node_type='group', edge_type='edge_group',
                           user_metapaths=[
                                   ['doi', 'name', 'doi'], ['doi','bioActivity','doi'],['doi','molType','doi'],['doi','collectionSpecie','doi'],
                                   ['doi','collectionSite','doi'],['doi','collectionType','doi'],['doi','molecularMass','doi'],
                                   ['doi','monoisotropicMass','doi'],['doi','cLogP','doi'],['doi','tpsa','doi'],
                                   ['doi','numberOfLipinskiViolations','doi'],['doi','numberOfH_bondAcceptors','doi'],['doi','numberOfH_bondDonors','doi'],
                                   ['doi','numberOfRotableBonds','doi'],['doi','molecularVolume','doi'],['doi','smile','doi'],
                               ]
                           ):
    s_graph = StellarGraph.from_networkx(graph, node_type_attr=node_type, edge_type_attr=edge_type)
    rw = UniformRandomMetaPathWalk(s_graph)
    walks = rw.run(
        s_graph.nodes(), n=num_walks, length=walk_length, metapaths=user_metapaths
    )
    
    print(f"Number of random walks: {len(walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
    )
    
    def get_embeddings(model, graph):
        if model is None:
            print("model not train")
            return {}

        _embeddings = {}
        for word in graph.nodes():
            try:
                _embeddings[word] = model.wv[word]
            except:
                _embeddings[word] = np.zeros(dimensions)

        return _embeddings
    return get_embeddings(model, graph)

#BFS2Vec

from stellargraph.data import SampledBreadthFirstWalk

def BFS2vec(graph, n_size=[5,5,5], n=5, seed=125, weighted=False, dimensions = 512,    context_window_size = 10, 
                           num_iter=1, workers = 1, node_type='group', edge_type='edge_group',
                           ):
    s_graph = StellarGraph.from_networkx(graph, node_type_attr=node_type, edge_type_attr=edge_type)
    rw = SampledBreadthFirstWalk(s_graph)
    walks = rw.run(s_graph.nodes(), n_size=n_size, n=n, seed=seed, weighted=weighted)
 
    print(f"Number of random walks: {len(walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
    )

    def get_embeddings(model, graph):
        if model is None:
            print("model not train")
            return {}

        _embeddings = {}
        for word in graph.nodes():
            try:
                _embeddings[word] = model.wv[word]
            except:
                _embeddings[word] = np.zeros(dimensions)

        return _embeddings
    return get_embeddings(model, graph)

"""
*************************************
*                                   *
*                                   *
*   UTILS FOR BENCHMARK EVALUATION  *
*                                   *
*                                   *
*************************************
"""

def hits_at(k, true, list_pred):
    hits = []
    for index_t, t in enumerate(true):
        hit = False
        # get the list of predicteds that's on the second argument
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if index_lp >= k:
                break
            if t[1] == lp:
                hits.append(1)
                hit = True
                break
        if not(hit):
            hits.append(0)
    return np.mean(hits)

def mrr(true, list_pred):
    # using the first list pred to get how many there will be
    rrs = []
    for index_t, t in enumerate(true):
        # get the list of predicteds that's on the second argument
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if t[1] == lp:
                rrs.append(1/(index_lp + 1))
                break
    return np.mean(rrs)
