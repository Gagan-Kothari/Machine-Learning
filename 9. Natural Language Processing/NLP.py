import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# in a tsv file, instead of words being separated by a comma they are separated by a tab-space (use delimiter = '\t'  [default = ','] )

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\9. Natural Language Processing\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# quoting = 3 means "NO QUOTES" or "IGNORE QUOTES"

#################### Cleaning the text

import re
import nltk                # Will help to clean-up stop words (irrelevant words: the, a, an, he, she ....) i.e. don't give hint to what the output should be


# nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer          # applies stemming to data, converts word to the root of it: eg- loved -> love

corpus = []                                  # empty list that will contain all the data after cleaning

for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ' , data.iloc[:,0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus)

import numpy as np
from scipy.sparse import csr_matrix

x = csr_matrix(x).toarray()    # csr_matrix() converts nd array to sparse matrix
y = data.iloc[:,-1].values


# Naive Bayes Prediction

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(x_train,y_train)

from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred = y_pred.reshape(-1,1)

y_test = y_test.reshape(-1,1)

print(np.concatenate((y_pred,y_test),1))

from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))