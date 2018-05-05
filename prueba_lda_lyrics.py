from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
en_stop = en_stop + get_stop_words('es')
palabras = ['si', 'pa', 'yeh', 'letra', 'ft', 'de', 'wuh', 'woa', 'oh', 'yeah', 'se']
en_stop = en_stop + palabras
#print(en_stop)

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer('spanish')

# Read Lyrics file and save text
filename_1 = 'Lyrics_BadBunny.txt'
fin=open(filename_1,'r')
texto=fin.read()
raw = texto.lower()

# separate songs
doc_set=raw.split("letra de")
tam=len(doc_set)

texts = []

for i in doc_set:
    tokens = tokenizer.tokenize(i)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens=[]

    for i in stopped_tokens:

        pstem=p_stemmer.stem(i)
        sstem=s_stemmer.stem(i)
        if pstem==sstem:
            stemmed_tokens.append(sstem)
        else:
            if pstem != i:
                stemmed_tokens.append(pstem)
            else:
                stemmed_tokens.append(sstem)

        #stemmed_tokens = [s_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=tam, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=tam, num_words=10))
