from nltk.corpus import stopwords

#stop = set(stopwords.words('english'))
#print(stop)

sentence = "This is a apple"
filter_sentence= [w for w in sentence.split(' ') if w not in stopwords.words('english')]
print(filter_sentence)

#词干提取(stemming)和词型还原(lemmatization)是英文文本预处理的特色
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english") # 选择语言
print(stemmer.stem("leaves")) # 词干化单词

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('leaves'))