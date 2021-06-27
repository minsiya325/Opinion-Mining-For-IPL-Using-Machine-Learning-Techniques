import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import pickle


np.random.seed(500)

Corpus1 = pd.read_csv("IPL2020_Tweets.csv",encoding='latin-1')
Corpus1 = Corpus1[:100]

Corpus1['text'].dropna(inplace=True)
Corpus1['text'] = [entry.lower() for entry in Corpus1['text']]
Corpus1['text']= [word_tokenize(entry) for entry in Corpus1['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus1['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus1.loc[index,'text_final'] = str(Final_words)


Test_X = Corpus1['text_final']

Tfidf_vect = pickle.load(open('tfidf.sav', 'rb'))

Test_X_Tfidf = Tfidf_vect.transform(Test_X)
Test_X_Tfidf=Test_X_Tfidf.toarray()

clf_rf = pickle.load(open('model_rf.sav', 'rb'))

predictions_rf = clf_rf.predict(Test_X_Tfidf)

#print(predictions_rf)
Encoder = pickle.load(open('label.sav', 'rb'))
out = Encoder.inverse_transform(predictions_rf)

count_0 = 0
count_1 = 0

for i in predictions_rf:
	if i == 0:
		count_0 += 1
	elif i == 1:
		count_1 += 1

if count_0 > count_1:
	perce = count_0/(count_0 + count_1)
	print('{}% negative opinion'.format(perce * 100))
elif count_0 < count_1:
	perce = count_1/(count_0 + count_1)
	print('{}% positive opinion'.format(perce * 100))
else:
	print('tie opinion')
