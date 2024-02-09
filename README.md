# CLTV-PREDICTION
CLTV Forecast with BGNBD &amp; GG and Sending Results to Remote Server
# BUSINESS P R O B L E M
# For an e-commerce site customer actions /behavior  forward according to the CLTV values of their customers 
# 1 month or 6 months with the dataset we have its the most revenue-generating within the periods 
#Dataset: The dataset, Online Retail II, contains the sales of a UK-based online retail store between 01/12/2009 and 09/12/2011
# The product catalog of this company includes souvenirs, Promotion can also be considered as products,There is also information that most of its customers are wholesalers


#Task 1 : 6 months  CLTV Prediction
############################################
# CUSTOMER LIFETIME VALUE
############################################

# 1. Data Preparation
# Average Order Value (average_order_value = total_price / total_transaction)
# Purchase Frequency (total_transaction / total_number_of_customers)
# Repeat Rate & Churn Rate (number of customers making more than one purchase / all customers)
# Profit Margin (profit_margin = total_price * 0.10)
# Customer Value (customer_value = average_order_value * purchase_frequency)
# Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creation of Segments

##################################################
# 1. Data Preparation
#Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#show all columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#reading data
df_ = pd.read_excel("/Users/mvahit/Desktop/DSMLBC4/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

cltv_c.head()
#################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################


cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']


##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10


##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"]) / churn_rate

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c['cltv'] = cltv_c['customer_value'] * cltv_c['profit_margin']

cltv_c.head()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_c[["cltv"]])
cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])

cltv_c.sort_values(by="scaled_cltv", ascending=False).head(50)

##################################################
# 8. Segmentation 
##################################################

# cutomers in 4 groups 
cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.head()


cltv_c[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].sort_values(by="scaled_cltv",
                                                                                              ascending=False).head()


cltv_c.groupby("segment")[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].agg(
    {"count", "mean", "sum"})
#
# cltv_segment rfm_segment
# B loyal_customers 245
# champions 161
# potential_loyalists 90
# at_risk 74
# need_attention 37
# about_to_sleep 14
# cant_loose 13
# hibernating 8

# Segment A customers, also in RFM segmentation, are mostly loyal_customers and
The # champions segment consists of users. For this segment:

# ▪ Organise loyalty programs.
# ▪ Provide free shipping etc. opportunities.
# ▪ Do not make suggestions in areas you know they are not interested in.
# ▪ Loyalist customers can be turned into "brand advocates". These users
# mouth to mouth marketing as they are already extremely loyal to the brand and products
# They are open to attracting new users and advocating products rather than the company.




##################################################
# BONUS: 
##################################################

def create_cltv_c(dataframe, profit=0.10):

    # data preparation 
  dataframe.dropna(inplace=True)
   dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe = dataframe[(dataframe['Price'] > 0)]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

  cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
   cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

    # avg_order_value
  cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

    # purchase_frequency
  cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

    # repeat rate & churn rate
   repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
  cltv_c['profit_margin'] = cltv_c['total_price'] * profit

    # Customer Value
  cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])

    # Customer Lifetime Value
   cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

   scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_c[["cltv"]])
    cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])

    # Segment
  cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])

   return cltv_c


df = df_.copy()
df.head()

cc = create_cltv_c(df)
