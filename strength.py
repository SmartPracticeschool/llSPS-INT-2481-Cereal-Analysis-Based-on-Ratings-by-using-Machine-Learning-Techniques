# -*- coding: utf-8 -*-

#import required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('cereal.csv')
df.head()
df.info()
df.shape
df.isnull().any()
df.isnull().sum()
df.columns
df['type'].unique()

df['mfr'].unique()

# visualization
plt.figure(figsize=(20, 4))

#Sugar info
plt.subplot(1, 5, 1)
plt.scatter(df["calories"],df["sugars"], color="c")
plt.xlabel("Calories (per serving)")
plt.ylabel('Sugar/gm')
plt.title('Cereals: Sugar & Nutrition Content')

#Carb info
plt.subplot(1, 5, 2)
plt.scatter(df["calories"],df["carbo"], color="m")
plt.xlabel("Calories (per serving)")
plt.ylabel('Carbohydrates/gm')
plt.title('Cereals: Carb & Nutrition Content')

#Fat info
plt.subplot(1, 5, 3)
plt.scatter(df["calories"],df["fat"], color="orange")
plt.xlabel("Calories (per serving)")
plt.ylabel('Fat/gm')
plt.title('Cereals: Fat & Nutrition Content')

#Fiber info
plt.subplot(1, 5, 4)
plt.scatter(df["calories"],df["fiber"], color="g")
plt.xlabel("Calories (per serving)")
plt.ylabel('Fiber/gm')
plt.title('Cereals: Fiber & Nutrition Content')

plt.scatter(df["mfr"],df["shelf"], marker="s", s=200, c="orangered")
plt.xlabel("Manufacturing")
plt.ylabel('Shelf Locations')
plt.xticks(['N','Q','K','R','G','P','A'], ['Nabisco','Quaker','Kellog','Ralston Purina','General Mills','Post','American Home Foods'], rotation = 70)
plt.title('Shelf Location of Cereal Manufacturers')

plt.show()
from sklearn.model_selection import train_test_split
X = df[['calories', 'protein', 'fat', 'sodium', 'fiber','carbo',  'potass', 'vitamins']]
print(X)

y= df.iloc[:,15]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
print("Accuracy on training set: {}".format(linear_reg.score(X_train,y_train)))
print("Accuracy on test set: {}".format(linear_reg.score(X_test,y_test)))
prediction = linear_reg.predict(X_test)

print(prediction)

import pickle
pickle.dump(mr,open('strength.pkl','wb'))
model=pickle.load(open('strength.pkl','rb'))











