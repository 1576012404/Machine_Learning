sample = ["Machine learning is fascinating, it is wonderful"
          ,"Machine learning is a sensational techonology"
          ,"Elsa is a popular character"]

from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
x=vec.fit_transform(sample)

print("features",vec.get_feature_names())

# print("x",x.shape,type(x),x)
# print("x",x.toarray())
# import pandas as pd
# vcresult=pd.DataFrame(x.toarray(),columns=vec.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
vec=TFIDF()
x=vec.fit_transform(sample)
print("x",x)





