import pandas as pd
from ast import literal_eval

from natuke_utils import hits_at
from natuke_utils import mrr

path = 'path-to-data-repository'
file_name = "knn_results"
splits = [0.8]
#edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
edge_group = 'doi_collectionType'
#algorithms = ['bert', 'deep_walk', 'node2vec', 'metapath2vec', 'regularization']
algorithms = ['deep_walk', 'node2vec', 'metapath2vec', 'regularization']
k_at = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dynamic_stages = ['1st', '2nd', '3rd', '4th']

# hits@k
hitsatk_df = {'k': [], 'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': []}
for algorithm in algorithms:
    for k in k_at:
        for split in splits:
            for iteration in range(10):
                for dynamic_stage in dynamic_stages:
                    restored_df = pd.read_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, dynamic_stage))
                    restored_df['true'] = restored_df['true'].apply(literal_eval)
                    restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                    hitsatk_df['k'].append(k)
                    hitsatk_df['algorithm'].append(algorithm)
                    hitsatk_df['split'].append(split)
                    hitsatk_df['edge_group'].append(edge_group)
                    hitsatk_df['dynamic_stage'].append(dynamic_stage)
                    hitsatk_df['value'].append(hits_at(k, restored_df.true.to_list(), restored_df.restored.to_list()))
                        
hitsatk_df = pd.DataFrame(hitsatk_df)
hitsatk_df.to_csv('{}metric_results/full_dynamic_hits@k_{}_{}.csv'.format(path, edge_group, file_name), index=False)
hitsatk_df_mean = hitsatk_df.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage'], as_index=False).mean()
hitsatk_df_std = hitsatk_df.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage'], as_index=False).std()
hitsatk_df_mean['std'] = hitsatk_df_std['value']
print(hitsatk_df_mean)

# mrr
mrr_df = {'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': []}
for algorithm in algorithms:
    for split in splits:
        for iteration in range(10):
            for dynamic_stage in dynamic_stages:
                restored_df = pd.read_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, dynamic_stage))
                restored_df['true'] = restored_df['true'].apply(literal_eval)
                restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                mrr_df['algorithm'].append(algorithm)
                mrr_df['split'].append(split)
                mrr_df['edge_group'].append(edge_group)
                mrr_df['dynamic_stage'].append(dynamic_stage)
                mrr_df['value'].append(mrr(restored_df.true.to_list(), restored_df.restored.to_list()))
                        
mrr_df = pd.DataFrame(mrr_df)
mrr_df.to_csv('{}metric_results/full_dynamic_mrr_{}_{}.csv'.format(path, edge_group, file_name), index=False)
mrr_df_mean = mrr_df.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage'], as_index=False).mean()
mrr_df_std = mrr_df.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage'], as_index=False).std()
mrr_df_mean['std'] = mrr_df_std['value']
print(mrr_df_mean)

# saving files
hitsatk_df_mean.to_csv('{}metric_results/dynamic_hits@k_{}_{}.csv'.format(path, edge_group, file_name), index=False)
mrr_df_mean.to_csv('{}metric_results/dynamic_mrr_{}_{}.csv'.format(path, edge_group, file_name), index=False)
