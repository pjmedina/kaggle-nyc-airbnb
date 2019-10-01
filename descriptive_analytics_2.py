import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#'exec(%matplotlib inline)'
import seaborn as sns

data = pd.read_csv('AB_NYC_2019.csv')

def data_quality_check(data):
    #check type of columns
    print(data.dtypes)
    print(data.head(10))

    #find out how which columns have null values
    print(data.isnull().sum())

    #name and host_name have 16 and 21 nulls, but these are not relevant
    #last_review, if there are no reviews then these are NaN
    #reviews_per_month if there are no reviews, these should be set to 0.0

def data_preparation(data):
    data.drop(['id','host_name','last_review'], axis =1 , inplace = True)

    #replace all NaN values with 0 for reviews per reviews_per_month
    data.fillna({'reviews_per_month':0}, inplace = True)
    #print('total nulls in reviews per month'+ str(data.reviews_per_month.isnull().sum()))
    #print(data.head(5))
    return data

def visualize_price_neighgroup(data):
    #we saw extreme values of 10,000 in brooklyn, manhatan, queens
    #create new df with no extreme values i.e only price <500
    sub_6 = data[data.price<500]
    #using violinplot for density and distribution of prices
    viz_2 = sns.violinplot(data = sub_6, x='neighbourhood_group', y ='price', scale = 'count')
    viz_2.set_title('Density and distribution of prices for each neighbourhood_group')

    print(plt.show(viz_2))

def visualize_price_hometype_neigh(data):
    sub_6 = data[data.price<500]
    sub_7= sub_6.loc[sub_6['room_type']== 'Private room']
    # sub_8= sub_6.loc[sub_6['room_type']== 'Shared room']
    # sub_9= sub_6.loc[sub_6['room_type']== 'Entire home/apt']

    # viz_3 = sns.violinplot(data = sub_8, x='neighbourhood_group', y ='price', scale = 'count')
    # viz_3.set_title('Density and distribution of prices of Shared rooms for each neighbourhood_group')
    #print(plt.show(viz_3))

    viz_4 = sns.catplot(x="neighbourhood_group", y="price", hue="room_type", data=sub_6, kind="violin", scale ='count')
    print(plt.show(viz_4))

def visualize_correlation(data):
    print(data.head(5))
    data.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'reviews_per_month'], axis=1, inplace=True)

    #convert categorical features into numerical ones
    data['room_type'] = data['room_type'].astype('category').cat.codes
    data['neighbourhood_group'] = data['neighbourhood_group'].astype('category').cat.codes

    sns.set(rc={'figure.figsize':(10,10)})
    viz_5 = sns.heatmap(data.corr().round(3), annot=True)
    print(plt.show(viz_5))

def visualize_roomtype_price(data):
    sub_10 = data[data.price<1000]

    viz_3 = sns.violinplot(data = sub_10, x='room_type', y ='price', scale = 'count')
    viz_3.set_title('Density and distribution of prices for each room type')
    print(plt.show(viz_3))


#data_quality_check(data)

# number_of_neighs = len(data.neighbourhood.value_counts().index.tolist())
# print(number_of_neighs)


#visualize_price_hometype_neigh(data)
#data = data_preparation(data)
#visualize_correlation(data)
visualize_roomtype_price(data)
#i want to see the distribution of price per roomtyp for each neighbourhood neig_group_list
