import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
import matplotlib
matplotlib.use("Qt5Agg")
# !pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

retail = pd.read_csv("free_work/online_retail_II/online_retail_II.csv")
df = retail.copy()

#--------------------------------- Data Preprocessing ----------------------------------------------

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T


# ---------------------------------- EDA(Exploratory Data Analysis) -----------------------------------
df.columns = [col.replace(" ", "_").upper() for col in df.columns]

df.dropna(inplace=True)

# ------------------------ Handling Outlier Values ------------------------------------------
df.describe().T
# There is negative value when we look at quantity, so let's examine and deal with it first
# QUANTITY

qtt = df.loc[df["QUANTITY"]<0,"QUANTITY"].count()
inv = df["INVOICE"].str.contains("C",na=False).sum()

print(f"The number of negative QUANTITY values: {qtt}\nThe number of INVOICES containing 'C' : {inv}")

# There was a problem with the cancellation process because the number of INVOICES containing "C" and the negative QUANTITY values were the same. When the transaction was canceled, the system entered a negative amount. Therefore, if we get rid of invoices containing "C" the problem will be solved

df.drop(df[df["INVOICE"].str.contains("C")].index,inplace=True)

qtt = df.loc[df["QUANTITY"]<0,"QUANTITY"].count()
inv = df["INVOICE"].str.contains("C",na=False).sum()
print(f"The number of negative QUANTITY values: {qtt}\nThe number of INVOICES containing 'C' : {inv}")

# they both look good

df.describe().T

# Negative price values also improved

# secondly, lets deal with outlier values

# INTERQUARTILE RANGE
# What is interquantile range?

def handling_outlier(data,variable):
    quartile1 = data[variable].quantile(0.01) # Range (%1-%99)
    quartile3 = data[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    data.loc[data[variable] < low_limit, variable] = low_limit
    data.loc[data[variable] > up_limit, variable] = up_limit

handling_outlier(df,"QUANTITY")
handling_outlier(df,"PRICE")

df.describe().T
# they look good

# Selecting some countries from the data set
list_cntry = ["Greece","Singapore","Netherlands","Switzerland","Cyprus","France","Korea","Canada"]
for number,country in enumerate(list_cntry):
    list_cntry[number] = df[df['COUNTRY'] == country]

del df
df = pd.concat(list_cntry,axis=0)
df = df.sort_index()

df = df.reset_index(drop=True)
df.shape
# ---------------------------------- Data Analysis & Visualization -------------------------------
# Top 10 best selling products

product_count = df.groupby("DESCRIPTION")["QUANTITY"].sum().nlargest(10)
product_count=product_count.reset_index()

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=product_count,y="DESCRIPTION",x="QUANTITY",palette="icefire")
for i in ax.containers:
    ax.bar_label(i,)
ax.set_title("Top 10 Best Selling Products")
plt.xlabel("Total Quantity")
plt.ylabel("Products")
plt.tight_layout()
plt.show()


# Price of any product by country

list_country, list_price = [], []
for col in df["COUNTRY"].unique():
    price = df.loc[(df["COUNTRY"] == col) & (df["DESCRIPTION"] == "WHITE HANGING HEART T-LIGHT HOLDER"), "PRICE"].mean()
    print(f" COUNTRY: {col} {price}")

    list_country.append(col)
    list_price.append(round(price,3))

df_price = pd.DataFrame(columns=["COUNTRY"],data=list_price,index=list_country)
df_price.dropna(inplace=True)
df_price = df_price.sort_values(by="COUNTRY",ascending=False)


plt.figure(figsize=(12, 8))
ax = sns.barplot(data=df_price,y=df_price.index,x="COUNTRY",palette="rocket_r")
for i in ax.containers:
    ax.bar_label(i,)
ax.set_title("Price of 'White Hanging Heart t-light Holder' by Country")
plt.xlim(0, 4)
plt.xlabel("Unit Price")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# Total amount of the first 10 products

df["TOTAL_AMOUNT"] = df["QUANTITY"] * df["PRICE"]
df.head()

total_amount = df.groupby("DESCRIPTION")["TOTAL_AMOUNT"].sum().nlargest(10)
total_amount=total_amount.reset_index()

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=total_amount,y="DESCRIPTION",x="TOTAL_AMOUNT",palette="viridis")
for i in ax.containers:
    ax.bar_label(i,)
ax.set_title("Total Amount of the First 10 Products")
# plt.xlim(0, 4)
plt.xlabel("Total Amount")
plt.ylabel("Products")
plt.tight_layout()
plt.show()



# ------------------- Preparing the ARL Data Structure (Invoice-Product Matrix) -------------------

# Setting it so that there are invoices in the rows and products in the columns. If there are products, 1, otherwise 0.

# Reaching the product quantities in each invoice.
df.groupby(["INVOICE","DESCRIPTION"])["QUANTITY"].sum().head(20)
# if you want you can use this code, it gives same result
df.groupby(["INVOICE","DESCRIPTION"]).agg({"QUANTITY":"sum"}).head(20)

# Sorting descriptions by columns
df.groupby(["INVOICE", "DESCRIPTION"]).agg({"QUANTITY": "sum"}).unstack().iloc[0:5, 0:5]


# Filling nan values with zero
df.groupby(['INVOICE', 'DESCRIPTION']).agg({"QUANTITY": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]


# 0,0 is converted to 0, if there is a value then it is 1
df.groupby(['INVOICE', 'DESCRIPTION']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# Changing product names with stock code
df.groupby(['INVOICE', 'STOCKCODE']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# It ready !!
df_arl = df.groupby(['INVOICE', 'STOCKCODE']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def prdct_name_finder(data,stckcde):
    product_name = data[data["STOCKCODE"] == stckcde][["DESCRIPTION"]].values[0].tolist()
    #print(product_name)
    return product_name

prdct_name_finder(df,"85014A") # it works

# ---------------------------------- Association Rule Analysis ----------------------------------
# what is Association Rule ?

frequent_itemsets = apriori(df_arl,min_support=0.01,use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

# association_rules(support degerleri, metric degeri="support", min_threshold = support degeri girme)
rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)

# Filtering
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

# Filtering by confidence
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

# --------------------------------------- Application ---------------------------------------
# For example, a member purchased a product with stock code 85123A...

def arl_recommender(rules_df, product_id, rec_count):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    recommendation_list_name = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j[1] == product_id:
                for k in list(sorted_rules.iloc[i]["consequents"]):
                    if k[1] not in recommendation_list:
                        recommendation_list.append(k[1])
    added_product = prdct_name_finder(df, product_id)
    print(f"Added to Cart:           {added_product[0]}\n\n")
    print(f"Members Who Bought This Also Bought:\n\n")
    for i in range(0, rec_count):
        recommendation_list_name.append(prdct_name_finder(df, recommendation_list[i]))
        print(f"                         {recommendation_list_name[i][0]}\n")


arl_recommender(rules, "85123A", 3)