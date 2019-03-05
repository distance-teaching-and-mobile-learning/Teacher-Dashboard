import zipfile
import pandas as pd
import matplotlib as plt
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize
get_ipython().run_line_magic('matplotlib', 'inline')



# POS frequency
df = pd.read_csv('alldata.csv')

count = df.groupby(['stemmed_word','pos_tag']).size().reset_index(name='counts')

total = count.groupby(['stemmed_word'])['counts'].agg(sum).reset_index(name='total')
final = pd.merge(count,total, on = 'stemmed_word')
final['precentage'] = final['counts']/final['total']
final.to_csv('POS frequency.csv')


#  TF Median

regex = r'\b\w+\b'
files = os.listdir('Corpus/')

df_data = pd.read_csv('data.csv')

df_data.rename(columns = {'TF_Happy Fee':'TF_Happy Feet','TF_Beauty and the Beas':'TF_Beauty and the Beast',
                         'TF_Cat in the ha':'TF_Cat in the hat','TF_Fantastic MrFo':'TF_Fantastic MrFox',
                         'TF_InsideOu':'TF_InsideOut'}, inplace =True)

data_col_name = [c.split('_')[1] for c in df_data.columns if 'TF_' in c]
a = [f.split('.')[0] for f in files]


print([x for x in a if x not in data_col_name])
print([x for x in data_col_name if x not in a])


lemma = df_data['Lemma']
tf_overall = df_data['TF']
output = [['document', 'sentence', 'words','tf_median', 'tf_overall']]
for file in files:
    col = 'TF_%s'%file.split('.')[0]
    tf_col = df_data[col]
    with open('Corpus/%s'%file, encoding='utf-8', errors='ignore') as f:
        chucks = f.read().splitlines()
        doc = ' '.join(chucks)
        sentences = sent_tokenize(doc)
        for s in sentences:
            sentence = list(set([x.lower() for x in word_tokenize(s) if s is not None]))
            if len(sentence) > 0:
                tf_median = tf_col[lemma.isin(sentence)].median()
                tf_all = tf_overall[lemma.isin(sentence)].median()
                output.append([file, s, sentence,tf_median, tf_all])

output_df = pd.DataFrame(output[1:], columns = output[0])

output_df.sort_values(['tf_overall'], ascending=False, inplace = True)

output_df.to_csv('tf_median_updated.csv', index = False)



# Extra

converted = []
regex = r'\b\w+\b'
files = os.listdir('Corpus/')

for file in files:
    with open('Corpus/%s'%file, encoding='utf-8', errors='ignore') as f:
        text = f.read().splitlines()
        sentences = ['_'.join([y.lower() for y in re.findall(regex,x)]) for x in text if x is not '']           
        converted.append(' '.join(sentences))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

max_features  = 5000
cv = CountVectorizer(max_features=max_features)
X = cv.fit_transform(converted)
tfidf = TfidfTransformer(norm=u'l2')
TfIdf_mat = tfidf.fit_transform(X)

TfIdf_mat

import scipy.sparse
scipy.sparse.save_npz('tfidf_matrix.npz', TfIdf_mat)
#sparse_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')


df1 = pd.read_csv('processed.csv', names = ['col_%s'%(x) for x in range(7)])

df1.drop(columns = ['col_6'], inplace = True )

df2 = pd.read_csv('adjacency_matrix92.csv')

df2.drop(columns = ['Unnamed: 0'], inplace = True )

new_cols = [x for x in df2['words'] if x in set(df1['col_1'])]

# Histograms
pd.Series([x if x<=40 else 41 for x in df_data.TF]).hist()
plt.title('TF Histogram')
plt.xticks(rotation=90)
plt.show()

df_data.TF.hist(bins = 20)
plt.title('TF Histogram')
plt.show()

df_alldata = pd.read_csv('alldata.csv')

df_alldata.pos_tag.hist()
plt.title('Pos Tag Histogram')
plt.xticks(rotation=90)
plt.show()

df_conted = pd.read_csv('connected.csv')

dist_10 = (round(df_conted[df_conted.value >= 0]['value']/10)).value_counts()

df_conted[df_conted.value >= 0]['value'].hist()
plt.title('Distance Histogram')
#plt.xticks(rotation=90)
plt.show()

dist = df_conted[df_conted.value >= 0]['value'].value_counts()

plt.plot(dist_10)
plt.title('Distance Histogram')
plt.xlabel('x10')
plt.show()

df = df2[df2['words'].isin(new_cols)][['words'] + new_cols]
df_melt = pd.melt(df, id_vars= ['words'] )
df_melt['connected'] = [int(x > 0) for x in df_melt['value']]
df_melt.to_csv('connected.csv')


pt_df = df_melt.pivot(index = 'words', columns = 'variable', values = 'connected')

# Graph Analysis

import numpy as np
from scipy.sparse import csr_matrix

csr = csr_matrix(pt_df.values)

print(list(zip(*csr.nonzero())))
pairs = list(zip(*csr.nonzero()))

import networkx as nx
graph = nx.from_edgelist(pairs)

l = set(list(nx.connected_components(graph))[0])


nx.draw_networkx(graph)

degree = nx.degree_centrality(graph)

import matplotlib.pyplot as plt

pd.Series([round(x,3) for x in list(degree.values())]).hist()
plt.title('Centrality')
plt.show()

centrality = nx.eigenvector_centrality(graph)

pd.Series([round(x,3) for x in list(centrality.values())]).hist()
plt.title('Eigenvector Centrality')
plt.show()

index = [list(df.columns[1:])[i] for i in degree.keys()]

pd.DataFrame({'degree':list(degree.values()),'eigenvector':list(centrality.values())}, index = index).to_csv('centrality.csv')

words = pt_df.columns.tolist()

nodes = list(graph.node())
result = [['node1','node2','n_step','path']]
l = len(nodes)
for i in range(l):
    for j in range(i+1,l):
        x = nodes[i]
        y = nodes[j]
        path = nx.bshortest_path(graph,source=x,target=y)
        result.append([words[x],words[y],len(path),','.join([words[z] for z in path])])

sh_df = pd.read_csv('shortest_path.csv')

sh_df.n_step.hist()
plt.title('Shorest Path')
plt.show()

shortest_df = pd.DataFrame(result[1:], columns = result[0])
shortest_df.to_csv('shortest_path.csv', index = False)

# histogram of POS

freq = {}
his = {}

with open('POS_frequency.csv') as f:
    for line in f:
        splits = line.split(',')
        if splits[1] not in freq:
            freq[splits[1]] = 1
        else:
            freq[splits[1]] += 1
            
for word in freq:
    if freq[word] not in his:
        his[freq[word]] = 1
    else:
        his[freq[word]] += 1

for num in his:
    print(str(num) + "," + str(his[num]))
 