from gensim.models import word2vec

sentences = word2vec.Text8Corpus('data.txt')
model = word2vec.Word2Vec(sentences, sg=1, size=100, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)
'''
参数解释：

1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。

2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。

3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。

4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。

5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。

6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。

7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。
'''

model.save(fname)
model = Word2Vec.load(fname)

model.most_similar(positive=['woman', 'king'], negative=['man'])
# 输出[('queen', 0.50882536), ...]

model.doesnt_match("breakfast cereal dinner lunch".split())
# 输出'cereal'

model.similarity('woman', 'man')
# 输出0.73723527

model['computer']  # raw numpy vector of a word
# 输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)