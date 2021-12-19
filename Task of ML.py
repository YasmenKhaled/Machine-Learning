# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:53:38 2021

@author: Tweety
"""
#1.	Read the dataset, convert it to DataFrame and display some from it.
#2.	Display structure and summary of the data.

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
dataset=pd.read_csv("Wuzzuf_Jobs.csv")
dataset.describe()

#3.	Clean the data ( duplications)

dataset.drop_duplicates(subset =["Title","Company","Location","Type","Level","YearsExp","Country","Skills"],keep = "first", inplace = True)

#3.	Clean the data (NULL)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = imp.fit(dataset)
transformed_data = imp.transform(dataset)

# EX  miss=(dataset=='null Yrs of Exp').sum()
#4.	Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)

x=dataset['Company'].value_counts()

#5.	Show step 4 in a pie chart

import matplotlib.pyplot as plt
plt.pie(x)
plt.title("jobs in each company")
plt.show()
 

#6.	Find out what are the most popular job titles.

y=dataset['Title'].value_counts()

#7.	Show step 6 in bar chart
y=y[0:10] #minimize the graph
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20,5))
f=y.keys()
plt.bar(f,y,color='black',width=0.1)


#8.	Find out the most popular areas

z=dataset['Location'].value_counts()

#9.	Show step 8 in bar chart
z=z[0:10] #minimize the graph
F=plt.figure(figsize=(10,5))
f=y.keys()
plt.bar(f,y,color='maroon',width=0.3)


#10.	Print skills one by one and how many each repeated and order the output to find out the most important skills required
#print skills one by one
skills=dataset['Skills']
#count skills> how many repeated
L=dataset['Skills'].value_counts()










