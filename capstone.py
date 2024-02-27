import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

data = pd.read_csv('CAR DETAILS.csv')
data['brand'] = data.name.apply(lambda x : ' '.join(x.split(' ')[:1]))
data['model'] = data.name.apply(lambda x : ' '.join(x.split(' ')[1:]))
data.drop('name' ,axis = 1, inplace = True)
def visualize(data,column):

  print('countplot visualization')
  print('\n')
  plt.figure(figsize = (10,7))
  sns.countplot(data = data , x = column, palette= 'CMRmap' )
  plt.xticks(rotation = 45)
  plt.xlabel(column, fontsize = 12, color = 'b')
  plt.ylabel('count',fontsize = 12, color = 'b')
  plt.title(f'distrubution {column} is Year by', color = 'blue', fontsize = 12)
  plt.legend()
  plt.axis('equal')
  plt.show()
  print('\n')

  print('pie chart visualization')
  print('\n')
  labels = data[column].value_counts().index
  size = data[column].value_counts()

  plt.figure(figsize = (8,8))
  plt.pie(size, labels = labels , autopct = '%1.1f%%',shadow = True, startangle = 45, rotatelabels=False)
  plt.title(f'distrubution {column} is year by', fontsize = 10, color = 'g')
  plt.axis('equal')
  plt.show()

  print('\n')
  print('visualize using histplot')
  print('\n')

  plt.figure(figsize = (8,4))
  sns.histplot(data = data[column], bins = 20, kde = True)
  plt.title(f'distrubution of {column}')
  plt.show()
  # making copy of orignal dataset

new_df = data.copy()
# outlier
columns = ['year','selling_price','km_driven']

fig, axes = plt.subplots(nrows=1, ncols=len(columns),figsize = (12,3) )

for i, column in enumerate(columns):
  sns.boxplot(data = new_df[column], ax = axes[i])
  axes[i].set_title(columns)

plt.tight_layout()
plt.show()
# remove outliers
def remove_outliers(data,columns):

  q1,q3 = data[columns].quantile([.25,.75])

  # find iqr
  iqr = q3 - q1

  lower_bound = q1 - 1.5*iqr
  upper_bound = q3 + 1.5*iqr

  # capping outliers
  outliers = data[(data[columns]> lower_bound) & (data[columns]<upper_bound)]

  return outliers
# remove outliers
for column in columns:
  new_df = remove_outliers(new_df, column)

# checking outliers after remove outliers
fig, axes = plt.subplots(nrows=1, ncols=len(columns),figsize = (12,3) )
for i, column in enumerate(columns):
  sns.boxplot(data = new_df[column], ax = axes[i])
  axes[i].set_title(columns)

plt.tight_layout()
plt.show()

# first  make a copy of the dataset
df = new_df.copy()
numeric_column = ['year','selling_price','km_driven']
categorical_column = ['fuel','seller_type','transmission','owner','brand','model']

brand_names = data['brand'].tolist()
model_names = data['model'].tolist()

pickle.dump(brand_names,open('brand_name.pkl','wb'))
pickle.dump(model_names,open('model_name.pkl','wb'))


# encoding categorical columns
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for column in categorical_column:
  df[column] =  lb.fit_transform(df[column])
  
from sklearn.preprocessing import StandardScaler

st = StandardScaler()
df[numeric_column] = st.fit_transform(df[numeric_column])

df.to_csv('cars_modify.csv', index = False)

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score ,classification_report
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR




def compare_models(x_train, y_train, x_test, y_test, models):
    results = {}

    for model_name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[model_name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R-squared': r2_score(y_test, y_pred)
        }

    return results

def evaluate_model(data,x, y):
    x = data.drop('selling_price',axis = 1)
    y = data['selling_price']
    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.3, random_state=42)

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor()),
        ("Support Vector Regressor", SVR()),
        ("Bagging Regressor", BaggingRegressor()),
        ("Voting Regressor", VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('svr', SVR())]))
    ]

    best_model = None
    best_metrics = {
        'Model': None,
        'MAE': float('inf'),  # Initialize with infinity for comparison
        'MSE': float('inf'),
        'RMSE': float('inf'),
        'R-squared': -float('inf')  # Initialize with negative infinity for comparison
    }

    for model_name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        if mae < best_metrics['MAE']:
            best_model = model
            best_metrics.update({
                'Model': model_name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R-squared': r2
            })

    return best_model  # Return only the best model objectdef evaluate_model(df):
    

df = pd.read_csv('cars_modify.csv')

x = df.drop('selling_price', axis = 1)
y = df['selling_price']

best_model = evaluate_model(df,x,y)

import pickle
# Save the model
file = 'best_model.pkl'
with open(file, 'wb') as f:
    pickle.dump(best_model, f)



# Load the car details dataset
cars_modify = pd.read_csv('cars_modify.csv')

import random 
# Pick 20 random data points
random_data = random.sample(list(cars_modify.index), 20)

# Load the saved model
loaded_model = pickle.load(open('best_model.pkl', 'rb'))

# Remove the target variable 'selling_price' if present
new_data = cars_modify.drop('selling_price', axis=1, errors='ignore')

# Apply the model to the random data
predictions = loaded_model.predict(new_data.loc[random_data])

# Print the predictions
print(predictions)
