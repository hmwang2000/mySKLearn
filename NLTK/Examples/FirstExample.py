import nltk
#nltk.download()

from nltk.tokenize import sent_tokenize
mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
print(sent_tokenize(mytext))

from nltk.corpus import wordnet
synonyms = []
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

from nltk.corpus import wordnet
antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('increases'))

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('working'))

'''
*** Introductory Examples for the NLTK Book ***
Loading text1, ..., text9 and sent1, ..., sent9
Type the name of the text or sentence to view it.
Type: 'texts()' or 'sents()' to list the materials.
text1: Moby Dick by Herman Melville 1851
text2: Sense and Sensibility by Jane Austen 1811
text3: The Book of Genesis
text4: Inaugural Address Corpus
text5: Chat Corpus
text6: Monty Python and the Holy Grail
text7: Wall Street Journal
text8: Personals Corpus
text9: The Man Who Was Thursday by G . K . Chesterton 1908
Moby Dick by Herman Melville 1851
'''
from nltk.book import *
print(text1.name)#书名
print(text1.concordance(word="love"))#上下文
print(text1.similar(word="very"))#相似上下文场景
print(text1.common_contexts(words=["pretty","very"]))#相似上下文
text4.dispersion_plot(words=['citizens','freedom','democracy'])#美国总统就职演说词汇分布图
print(text1.collocations())#搭配
print(type(text1))
print(len(text1))#文本长度
print(len(set(text1)))#词汇长度
fword=FreqDist(text1)
print(text1.name)#书名
print(fword)
voc=fword.most_common(50)#频率最高的50个字符
fword.plot(50,cumulative=True)#绘出波形图
print(fword.hapaxes())#低频词
