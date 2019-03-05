from nltk.stem import PorterStemmer
from collections import OrderedDict
import os
import csv
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import names
import nltk
import enchant
eng_dictionary = enchant.Dict("en_US")


names = names.words('male.txt') + names.words('female.txt')
ps = PorterStemmer()
files = os.listdir("Corpus/")
for file in sorted(files):
    with open('alldata.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['stemmed_word', 'pos_tag'])
        lines = open('Corpus/{}'.format(file), encoding='utf-8', errors='ignore').readlines()
        for line in lines:
            words = word_tokenize(line.lower())
            for word in words:
                if word.capitalize() in names:
                    word_pos_tag = 'Name'
                elif not eng_dictionary.check(word):
                    word_pos_tag = 'NEW'
                else:
                    word_pos_tag = nltk.pos_tag([word])[0][1]
                csv_writer.writerow([ps.stem(word), word_pos_tag])



for file in sorted(files):
    lines = open('Corpus/{}'.format(file), encoding='utf-8', errors='ignore').readlines()
    for line in lines:
        words = word_tokenize(line.lower())
        dump_dict = OrderedDict()
        for word in words:
            if word.capitalize() in names:
                word_pos_tag = 'Name'
            elif not eng_dictionary.check(word):
                word_pos_tag = 'NEW'
            else:
                word_pos_tag = nltk.pos_tag([word])[0][1]
            dump_dict[word] = word_pos_tag
        if dump_dict:
            json.dump(dump_dict, open('pos_tag_files/{}'.format(file),'a'))
            open('pos_tag_files/{}'.format(file), 'a').write('\n')





