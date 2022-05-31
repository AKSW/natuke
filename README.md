# Natural Product Knowledge Extraction Benchmark
NaPKE source code for running and evaluating experiments

## GraphEmbeddings
GraphEmbeddings submodule based on https://github.com/shenweichen/GraphEmbedding but the used algorithms works with tf 2.x
### install
inside GraphEmbeddings directory from this repository run
```
python setup.py install
```

## metapath2vec
metapath2vec submodule based on https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/metapath2vec-link-prediction.html

## enviroments compatibility
for a better user experience we recommend setting up two virtual environments for running biologist: 
requirements.txt for all the codes, except topic_distribution.ipynb; topic_generation.ipynb; and hin_generation.ipynb;
requirements_topic.txt for topic_distribution.ipynb; topic_generation.ipynb; and hin_generation.ipynb (BERTopic requires a different numpy version for numba).

## wip
wiki page with further info as well as other usability and reproducibility features will be made available in this repository.
