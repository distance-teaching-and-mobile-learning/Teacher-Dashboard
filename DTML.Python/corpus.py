import os
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pandas as pd
import re
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
files = os.listdir("Corpus")
df = defaultdict(dict)
num_of_file = len(files)
for file in files:
    word_freq = dict()
    lines = open('Corpus/{}'.format(file), encoding='utf-8', errors='ignore').readlines()
    for line in lines:
        words = [re.search(r'[a-z]*', item.lower()).group() for item in line.split() if re.search(r'[a-z]*', item.lower()).group() ]
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    for word in word_freq:
        if word not in df:
            df[word]['Stem'] = ps.stem(word)
            wo, df[word]['Pos'] = nltk.pos_tag([word])[0]
            df[word]['TF_{}'.format(file.strip('.txt'))] = word_freq[word]
        else:
            df[word]['TF_{}'.format(file.strip('.txt'))] = word_freq[word]


for word in df:
    df[word]['IDF'] = (len(df[word])-2)/num_of_file
    df[word]['TF'] = sum([df[word][key] for key in df[word].keys() if (key not in ['Stem', 'Pos', 'IDF'])])


data = pd.DataFrame.from_dict(df, orient='index')
data.index.name = 'Lemma'
data.fillna(0, inplace=True)
data['Rank'] = data['TF'].rank(ascending=0, method='dense')
data.sort_values('Rank', inplace=True)
data.to_csv('data.csv')
data.groupby('Stem')