# %%
import pandas as pd

path = 'path-to-data-repository'

# %%
texts_df = pd.read_parquet('{}texts_query03-05.parquet'.format(path))
# %%
texts_df['text'] = texts_df['text'].apply(lambda x: x.replace('\n', ' '))

# %%
def phrases_slicing(x):
    phrases = []
    slices = round(len(x)/512)
    for i in range(slices):
        if i + 1 == slices:
            phrases.append(x[i*512:])
        else:
            phrases.append(x[i*512:(i+1)*512])
    return phrases

# %%
texts_df['phrases'] = texts_df['text'].apply(phrases_slicing)
print(texts_df)

# %%
texts_df.to_parquet("{}pdf-phrases03-05.parquet".format(path))


