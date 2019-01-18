import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9, -4.3]])

print(data)

#标准化（均值移除）
data_standardized = preprocessing.scale(data)
print("\nMean =", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))

#归一化
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data =", data_scaled)

#中心化
data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data =", data_normalized)

#二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data =", data_binarized)

#独热编码
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded vector =", encoded_vector)
'''
分析：
4个特征：
第一个特征（即为第一列）为[0,1,2,1] ，其中三类特征值[0,1,2]，因此One-Hot Code可将[0,1,2]表示为:[100,010,001]
同理第二个特征列可将两类特征值[2,3]表示为[10,01]
第三个特征将4类特征值[1,2,4,5]表示为[1000,0100,0010,0001]
第四个特征将2类特征值[3,12]表示为[10,01]
因此最后可将[2,3,5,3]表示为[0,0,1,0,1,0,0,0,1,1,0]
'''

#标记编码
label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels =", labels)
print("Encoded labels =", list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_labels))