import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Importing the training set and initialize the X_train and y_train
#We drop the features that will not be very useful to our model
training_dataset=pd.read_csv('BallonDorData.csv').drop(columns=['Player','Year','AVG goals','AVG assists','Value'])
training_dataset=training_dataset.sample(frac=1).reset_index(drop=True)
X_train=training_dataset.drop(columns=['BallonDor']).values
y_train=training_dataset.iloc[:,-1].values

#Initialization of X_test.
test_dataset=pd.read_csv('BallonDorTest.csv').drop(columns=['Player','Year','AVG goals','AVG assists','Value'])
X_test=test_dataset.values

#This is an auxiliary dataset for the final results.We use it for the Player names
results_dataset=pd.read_csv('BallonDorTest.csv').drop(columns=['AVG goals','Year','AVG assists','Value'])

#Printing the datasets to check
print(training_dataset.head())
print(test_dataset.head())
print(results_dataset.head())

#Get the index of the categorical features of the training set to apply Column Transformer and OneHotEncoder later...
categ_indexes_train=[]
for col in training_dataset.columns:
  if training_dataset[col].dtype=='object':
    categ_indexes_train.append(training_dataset.columns.get_loc(col))
print(categ_indexes_train)

#Get the index of the categorical features of the test set to apply Column Transformer and OneHotEncoder later...
categ_indexes_test=[]
for col in test_dataset.columns:
  if test_dataset[col].dtype=='object':
    categ_indexes_test.append(test_dataset.columns.get_loc(col))
print(categ_indexes_test)

#Turns out the index is [10] at both situations

#Actually applying Column Transformer
CT=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[10])],
                     remainder='passthrough')
X_train=np.array(CT.fit_transform(X_train))
X_test=np.array(CT.transform(X_test))

#Applying Feature Scaling so our model does not get affected by higher values
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#We check again. . .
print(X_train)
print(X_test)

#Here is the creation of the model.In this situation I prefer Logistic Regression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#Predicting the results of the test set (Future outcomes)
y_pred=classifier.predict_proba(X_test)[:,1]

#Creating a data frame that will help us print the output
results=pd.DataFrame({
    'Player':results_dataset['Player'].str.upper(),
    'Prediction':y_pred
})

#We sort the players based on their possibility to win the Ballon D'or
results=results.sort_values(by='Prediction',ascending=False).reset_index(drop=True)

#We will print the top 6 contenders of the Ballon D'or
top_6=results.iloc[:6].copy()

#Creating a 'Final Percentage' column, so the SUM of the top 6 predictions are 85.0%
top_6['Final Percentage']=(top_6['Prediction']/(top_6['Prediction'].sum()))*85
top_6['Names']=results_dataset['Player'].iloc[:6]
#Rounding the % to be more clear and understanding
top_6['Final Percentage']=top_6['Final Percentage'].round(0).astype(int)

#Printing the results :
print("\n2026 BALLON DOR PREDICTIONS\n")
print('==========TOP 5==========')

for index, row in top_6.iterrows():
    print(f"{index + 1}. {row['Player']:<15} : {row['Final Percentage']:.1f}%")

print('OTHERS : 15.0%')

#Visualizing the results

label = top_6['Player'].tolist()
sizes = top_6['Final Percentage'].tolist()
label.append("Others")
sizes.append(15)

#Visualization with Pie Chart
fig,ax=plt.subplots()
ax.pie(sizes,labels=label,autopct='%1.1f%%')
plt.show()
