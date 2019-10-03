import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#'exec(%matplotlib inline)'
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv('AB_NYC_2019.csv')
print(data.isnull().sum())

def data_cleaning(data):
    #categorical:
    #name (16 null) N, host_name (21) N, neighbourhood_group Y, neighbourhood N, room_type Y, last_review (10K null) N,

    #numerical:
    #id N, host_id N, latitude, longitude, price, minimum_nights, number_of_reviews,
    #reviews per month (10K null) , calculated_host_listings_count, availability_365

    #drop unnecessary columns
    data.drop(['name', 'host_name', 'neighbourhood','last_review', 'id','host_id'], axis = 1, inplace = True)

    #we dont want data points that are not going to be rented %35 of overall data
    data = data[data['availability_365']!=0]

    #we only want to predict those that have a price lower than 500
    data = data[data.price<=500]

    #set all those reviews per month that are null to 0
    data.fillna({'reviews_per_month':0}, inplace = True)

    #process categorical features into numerical
    data['neighbourhood_group'] = data['neighbourhood_group'].astype('category').cat.codes
    data['room_type'] = data['room_type'].astype('category').cat.codes
    return data

data = data_cleaning(data)
#define data and target variable
y = data['price']
X = data[['neighbourhood_group', 'longitude', 'latitude','room_type', 'reviews_per_month','availability_365', 'calculated_host_listings_count']]

#separate it into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fit and transform xtrain data using minmax scaler.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

#create the model and fit (Train it) x_train (after scaling) and y_train (no scaling)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

model_2 = DecisionTreeRegressor()
model_2.fit(X_train_scaled, y_train)

#transform x_test
X_test_scaled = scaler.transform(X_test)

#predict using xtest transformed
y_pred = model.predict(X_test_scaled)
y_pred_2 = model_2.predict(X_test_scaled)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_pred.shape)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

rmse_2= (np.sqrt(mean_squared_error(y_test, y_pred_2)))
r2_2 = r2_score(y_test, y_pred_2)

error_frame = pd.DataFrame({'Actual': np.array(y_test).flatten(), 'Predicted': y_pred.flatten()})
error_frame_2= pd.DataFrame({'Actual': np.array(y_test).flatten(), 'Predicted': y_pred_2.flatten()})
print(error_frame.head(10))
print(error_frame_2.head(10))

print('rmse 1 is :' +str(rmse))
print('r2 1 is :' + str(r2))

print('rmse 2 is :' +str(rmse_2))
print('r2 2 is :' + str(r2_2))
