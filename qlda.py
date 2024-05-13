import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#funtion to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.pink):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
df = pd.read_csv("data.csv", header=None, dtype=str)
df['label'] = df[df.shape[1]-1]
df.drop([df.shape[1]-2], axis=1, inplace=True)

X = np.array(df.drop(['label'], axis = 1), dtype='<U13')
y = np.array(df['label'])
X = X[1:,2:]
y = y[1:]
print(X)
print(y)


kf = KFold(n_splits=2)
kf.get_n_splits(X)
# y = list(map(int, y))
y_converted = list(map(lambda x: -1 if x == 0 else x, y))
X_train, X_test, y_train, y_test = train_test_split(X, y_converted,test_size=0.25, random_state= 100)
print("\ndữ liệu train sau khi đã chia\n")
print(X_train)
print(y_train)


X_train = list(X_train)
new_X_train = []
for i in X_train:
    for j in i:
        new_X_train.append(j)

X_test = list(X_test)
new_X_test = []
for i in X_test:
    for j in i:
        new_X_test.append(j)

print("\ndữ liệu train sau khi chuyển từ array sang list\n")
print(X_train)


count_vect = CountVectorizer(lowercase=False)
X_train_counts = count_vect.fit_transform(new_X_train)
X_test_counts = count_vect.transform(new_X_test)
df_counts = pd.DataFrame(X_train_counts.A, columns=count_vect.get_feature_names_out())
print(df_counts)
print("\n dữ liệu train_counts\n",X_train_counts)
print("\n dữ liệu train_counts\n",X_test_counts)



#Passive Aggressive Classifier with count vectorizer
pac = PassiveAggressiveClassifier(max_iter=500)
pac.fit(X_train_counts,y_train)



row = X[5]

# row = [""]
print(row)
new_document_counts=count_vect.transform(row)
# Dự đoán với mô hình đã huấn luyện
pred_y = pac.predict(new_document_counts)  # Dự đoán nhãn cho văn bản mới
# In kết quả dự đoán
print("Dự đoán cho văn bản mới:")
print(pred_y)  # In nhãn dự đoán


pred_y_pac = pac.predict(X_test_counts)
d= accuracy_score(y_test,pred_y_pac)
print ("Accuracy", float("{0:.2f}".format(d*100)))
# Thay đổi tùy chọn in để không bị cắt giảm
np.set_printoptions(threshold=np.inf)  # Vô hạn, in ra toàn bộ mảng
np.set_printoptions(linewidth=120)  # Tăng chiều rộng dòng để tránh xuống dòng
print("Dự đoán cho tập kiểm tra:")
print(pred_y_pac)

cm_pac = confusion_matrix(y_test,pred_y_pac)
print(cm_pac)
plot_confusion_matrix(cm_pac, classes=['FAKE', 'REAL'])