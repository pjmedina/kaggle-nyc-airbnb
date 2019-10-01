#import necessary libraries
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

def categorical_features(data):
    neig_roup_array = data.neighbourhood_group.unique()
    neig_array = data.neighbourhood.unique()
    room_type_array = data.room_type.unique()


def top_hosts(data):
    #number of listings per host
    top_10_hosts = data.host_id.value_counts().head(10)
    #sanity check with calculated host listing counts
    max_host_list_counts = data.calculated_host_listings_count.max()

    print(top_10_hosts)
    print(max_host_list_counts)
    #set figure size
    sns.set(rc={'figure.figsize':(10,8)})
    viz_1 = top_10_hosts.plot(kind = 'bar')
    viz_1.set_title('Hosts with the most listings in NYC')
    viz_1.set_ylabel('Count of Listings')
    viz_1.set_xlabel('Host IDs')
    viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation = 45)

    print(plt.show(viz_1))

def price_neighgroup(data):
    # brooklyn, manhattan, queens, staten island, bronx

    sub_1= data.loc[data['neighbourhood_group']== 'Brooklyn']
    price_sub1= sub_1[['price']]

    sub_2= data.loc[data['neighbourhood_group']== 'Manhattan']
    price_sub2= sub_2[['price']]

    sub_3= data.loc[data['neighbourhood_group']== 'Queens']
    price_sub3= sub_3[['price']]

    sub_4= data.loc[data['neighbourhood_group']== 'Staten Island']
    price_sub4= sub_4[['price']]

    sub_5= data.loc[data['neighbourhood_group']== 'Bronx']
    price_sub5= sub_5[['price']]

    #puttin all price dataframes in one listing
    price_list_by_neigh = [price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]

    #empty list for price distribution
    price_list_by_neigh_2 = []

    # list of unique neighbourhood_group
    neig_group_list = neig_roup_array = data.neighbourhood_group.unique().tolist()

    #statistics of price ranges
    #for every price of each neighbourhood_group
    for x in price_list_by_neigh:
        #this gives us the count, mean, std, min, 25%,50%,75% and max
        i = x.describe(percentiles=[0.25, 0.50, 0.75])
        #we are grabbing from the min to max values here
        i = i.iloc[3:]
        i.reset_index(inplace=True)
        #create a table index/Stats/price
        i.rename(columns={'index':'Stats'}, inplace= True)
        price_list_by_neigh_2.append(i)

    #rename price to neighbourhood
    for i in range(len(neig_group_list)):
        price_list_by_neigh_2[i].rename(columns={'price':neig_group_list[i]}, inplace=True)


    price_stats_df = price_list_by_neigh_2

    #setting all df to have Stats as their index, so they all have the same rows
    price_stats_df = [df.set_index('Stats') for df in price_stats_df]
    price_stats_df = price_stats_df[0].join(price_stats_df[1:])
    return price_stats_df

def visualize_price_neighgroup(data):
    #we saw extreme values of 10,000 in brooklyn, manhatan, queens
    #create new df with no extreme values i.e only price <500
    sub_6 = data[data.price<500]
    #using violinplot for density and distribution of prices
    viz_2 = sns.violinplot(data = sub_6, x='neighbourhood_group', y ='price', scale = 'count')
    viz_2.set_title('Density and distribution of prices for each neighbourhood_group')

    print(plt.show(viz_2))

def visualize_neighs(data):
    #lets only use the top 10 neigs with the most Listings
    top_10_neighs = data.neighbourhood.value_counts().head(10).index.tolist()
    print(top_10_neighs)
    sub_7 = data.loc[data['neighbourhood'].isin(top_10_neighs)]
    #use catplot to represent attributes together and a count
    viz_3= sns.catplot(x='neighbourhood',hue='neighbourhood_group', col='room_type', data= sub_7, kind = 'count')
    viz_3.set_xticklabels(rotation=90)

    print(plt.show(viz_3))

def visualize_longitude_latitude_price(data):
     sub_6 = data[data.price<500]
     viz_4 = sub_6.plot(kind = 'scatter', x='longitude', y='latitude', label='availability_365', c='price', cmap = plt.get_cmap('jet'), colorbar = True, alpha = 0.4, figsize=(10,8))
     viz_4.legend()
     print(plt.show(viz_4))


def visualize_names(data):
    names = []

    for name in data.name:
        names.append(name)

    def split_name(name):
        spl = str(name).split()
        return spl

    names_for_count =[]

    for x in names:
        for word in split_name(x):
            word = word.lower()
            names_for_count.append(word)

    #we are going to use counter
    from collections import Counter
    #let's see top 25 used words by host to name their listing
    _top_25_w=Counter(names_for_count).most_common()
    _top_25_w=_top_25_w[0:25]

    sub_w=pd.DataFrame(_top_25_w)
    sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)
    print(sub_w)

    viz_5=sns.barplot(x='Words', y='Count', data= sub_w)
    viz_5.set_title('Counts of the top 25 used words for listing names')
    viz_5.set_ylabel('Count of words')
    viz_5.set_xlabel('Words')
    viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)
    print(plt.show(viz_5))

def reviews(data):
    #Return the first n rows ordered by columns in descending order.
    top_10_reviewed= data.nlargest(10, 'number_of_reviews')
    #value counts is like a hashmap that keeps track of unique values and their count
    top_10_rev = data.number_of_reviews.value_counts().head(10)

    df_10_rev = data.sort_values(by= ['number_of_reviews'], ascending = False).head(10)

    average_price = df_10_rev.price.mean()
    #print(top_10_reviewed)
    print(df_10_rev)
    print('Average price per night: {}'.format(average_price))
#data_quality_check(data)
data_preparation(data)
#ca
#top_hosts(data)
# stats_df = price_neighgroup(data)
# print(stats_df)
# visualize_price_neighgroup(data)
#visualize_neighs(data)
#visualize_longitude_latitude_price(data)
#visualize_names(data)
reviews(data)
