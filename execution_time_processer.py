import pandas as pd
path = 'path-to-data-repository'

with open('{}results/execution_time.txt'.format(path)) as f:
  lines = f.readlines()
execution_times = [line.split(',') for line in lines]

execution_times_dict = {'algorithm': [], 'split': [], 'edge_group': [], 'dynamic_stage': [], 'execution_time': []}
for et in execution_times:
  execution_times_dict['algorithm'].append(et[0])
  execution_times_dict['split'].append(et[1])
  execution_times_dict['edge_group'].append(et[3])
  execution_times_dict['dynamic_stage'].append(et[4])
  execution_times_dict['execution_time'].append(float(et[5].replace('\n','')))

df = pd.DataFrame(execution_times_dict)
df.to_csv('{}metric_results/execution_times.csv'.format(path), index=False)