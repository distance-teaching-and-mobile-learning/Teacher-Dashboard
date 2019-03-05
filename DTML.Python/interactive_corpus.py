from nltk.stem import PorterStemmer
import os
import pandas as pd
import json
from collections import OrderedDict

ps = PorterStemmer()
word = input("Please Enter Word: ")
word = ps.stem(word)
pos = input("Please Enter pos: ")
top_number = int(input("Please Enter top number: "))

files = sorted(os.listdir('pos_tag_files'))
df = pd.read_csv('alldata.csv')


def before_occurrence(pos_tag_file, word, pos):
    content = open('pos_tag_files/{}'.format(pos_tag_file), errors='ignore').readlines()
    for line in content:
        if word in line:
            dicts = json.loads(line, object_pairs_hook=OrderedDict)
            key_list = list(dicts.keys())
            try:
                index = key_list.index(word)
                for i in range(index - 1, -1, -1):
                    if dicts[key_list[i]] == pos:
                        counts = len(df.index[df['stemmed_word'] == key_list[i]].tolist())
                        return (key_list[i], counts)
            except ValueError:
                continue
        else:
            continue
    return None


def after_occurrence(pos_tag_file, word, pos):
    content = open('pos_tag_files/{}'.format(pos_tag_file), errors='ignore').readlines()
    for line in content:
        if word in line:
            dicts = json.loads(line, object_pairs_hook=OrderedDict)
            key_list = list(dicts.keys())
            try:
                index = key_list.index(word)
                for j in range(index + 1, len(dicts)):
                    if dicts[key_list[j]] == pos:
                        counts = len(df.index[df['stemmed_word'] == key_list[j]].tolist())
                        return (key_list[j], counts)
            except ValueError:
                continue
        else:
            continue
    return None


result_dict = dict()
for file in files:
    if before_occurrence(file, word, pos):
        word1, freq1 = before_occurrence(file, word, pos)
        if word1 not in result_dict:
            result_dict[word1] = freq1
    if after_occurrence(file, word, pos):
        word2, freq2 = after_occurrence(file, word, pos)
        if word2 not in result_dict:
            result_dict[word2] = freq2

print(pd.DataFrame(list(result_dict.items()), columns=['word', 'freq']).sort_values('freq', ascending=0).iloc[0:top_number,])


