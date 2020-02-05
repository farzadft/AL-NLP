#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re 
import string 
import nltk 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from nltk.stem import WordNetLemmatizer
import itertools
from tqdm import tqdm 
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy
from nltk.tokenize import word_tokenize 
from spacy import displacy
from country_list import *

get_ipython().system('pip install spacy && python -m spacy download en')
get_ipython().system('python spacy_ner_custom_entities.py ')
from nltk.corpus import stopwords
nlp = spacy.load("en")
lemmatizering = WordNetLemmatizer()

data=pd.read_csv('Dumy-Data.csv')

def sex(text):
    doc= nlp(text)
    female_list=['female','Female','Girl','girl','Woman','woman','her','Her','She','she']
    for i in doc :
        if i.text in ['male','Male','boy','Boy','man','his','His','He','he'] and i.pos_ in ['NOUN','ADJ','PRON']:
            return 'Male'
        if i.text in female_list and i.pos_ in ['NOUN','ADJ','PRON'] :
            return 'Female'

        
def age(text):
    text=text.replace('.','')
    text=text.replace('�','')
    text=text.replace('-',' ')
    new_df=pd.DataFrame()
    year_old=[]
    list1=[]
    list2=[]
    l=[]
    list3=[] 
    for i in range(100000):
        list2.append(str(i))
       
    tokenize= word_tokenize(text)

#taking out times
    for words in tokenize:
        for i in words:
            if i in [':',',',]:
                words=words.replace(words,'')
         
        list1.append(words)
        list1.append(' ')
    text_time=' '.join([lemmatizering.lemmatize(w) for w in list1 ])
    
    doc= nlp(text_time)
    for i in range(len(doc)):
        if doc[i].pos_=='NUM':
                if str(doc[i].text) in list2:
                    if int(doc[i].text)<120:
                        try:
                            year_old.append((str(doc[i-2].text)+str(doc[i-1].text)+str(doc[i].text)+str(doc[i+1].text)+str(doc[i+2].text)))
           
        
                        except:
                            year_old.append(str(doc[i-2].text)+str(doc[i-1].text)+str(doc[i].text))
                else:
                    try:
                        year_old.append(str(doc[i-1].text)+str(doc[i].text)+str(doc[i+1].text)+str(doc[i+2].text))
                    except:
                        year_old.append(str(doc[i-1].text)+str(doc[i].text))
                        
    names=[str(i).split() for i in name(text)]
    names1=[l for i in names for l in i ]
    for i in range(len(year_old)):
        split1=year_old[i].split()
        for m in range(len(split1)):
            try:
                if split1[m] in ['He','he','she','She'] or split1[m] in names1 or split1[m] in ['late','early']:
                    list3.append(str(split1[m+1])+' '+'years'+' '+'old')
            except:
                list3.append('')
            if split1[m] in ['year','month','yrs']:
                list3.append(str(split1[m-1])+' '+ str(split1[m])+' '+'old')
    return list3
        
      
        
def name(text):
    text=text.replace('�','')
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
    
        
    return [list1[i-1] for i in range(len(list1)) if list1[i]=='PERSON']

def town_adress(text):
    text=text.replace('�','')
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
        
    for e in range(len(list1)):
            return [list1[e-1] for e in range(len(list1)) if list1[e] in ['GPE','LOC','FAC']]
            if list1[e] in countries:
                return list1[e]+[list1[i-1] for i in range(len(list1)) if list1[i] in ['GPE','LOC','FAC']]
            
    else:
        return None
def date(text):
    text=text.replace('.','')
    text=text.replace('�','')
    list2=[]
    for i in range(100000):
        list2.append(str(i))
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
    return [list1[i-1] for i in range(len(list1)) if list1[i] in ['DATE','TIME'] and list1[i-1] not in list2]
           
                
def main():
    cleaned_data=pd.DataFrame()
    cleaned_data['Narrative']= data['Narrative']
    cleaned_data['sex']= data['Narrative'].apply(sex)
    cleaned_data['age']=data['Narrative'].apply(age)
    cleaned_data['name']=data['Narrative'].apply(name)
    cleaned_data['town']=data['Narrative'].apply(town_adress)
    cleaned_data['date']=data['Narrative'].apply(date)
    return cleaned_data


# In[27]:


main().head(50)


# In[27]:


data=pd.read_csv('Dumy-Data.csv')
df=pd.DataFrame()
df['Nar']=data['Narrative'].map(lambda x: re.sub('[,\.!?]', '', x))
df['Nar'] = df['Nar'].map(lambda x: x.lower())
df['Nar'].head()


# In[33]:


get_ipython().run_cell_magic('time', '', "import gensim\nfrom gensim.utils import simple_preprocess\ndef sent_to_words(sentences):\n    for sentence in sentences:\n        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations\ndata1 = df['Nar'].values.tolist()\ndata_words = list(sent_to_words(data1))\nprint(data_words[0])")


# In[34]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[39]:


# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[40]:


import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])


# In[41]:


import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])


# In[42]:


lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)


# In[43]:


from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[44]:


from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[45]:


def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


# In[51]:


import numpy as np
import tqdm
grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
               corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
if 1 == 1:
    pbar = tqdm.tqdm(total=540)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    


# In[4]:


for i in nlp('AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins.'):
    print(i.text,i.pos_)


# In[6]:


data


# In[22]:


get_ipython().system('pip install spacy && python -m spacy download en')
get_ipython().system('python spacy_ner_custom_entities.py ')
nlp = spacy.load("en")
def noun(text):
    doc= nlp(text)
    list1= [i.text for i in doc if i.pos_=='NOUN']
    return list1
    


# In[23]:


data['Narrative'].apply(noun)


# In[17]:





# In[ ]:




