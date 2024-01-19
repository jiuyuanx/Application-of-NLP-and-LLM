#%%
import pandas as pd
import numpy as np
import os

#%%
# os.listdir('business')
#%%
def load_data():
    all_docs = []
    topic_list = ['business','entertainment','politics','sport','tech']
    for topic in topic_list:
        docs=os.listdir(topic)
        for doc in docs:
            path = topic+'/'+doc
            file = open(path, "r")
            content = file.read()
            row = [topic, content]
            all_docs.append(row)
    df = pd.DataFrame(all_docs, columns = ['topic','content'])
    return df
#%%
# load_data()
#%%