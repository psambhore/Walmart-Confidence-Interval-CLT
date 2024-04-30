#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries - 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from scipy.stats import kstest
import statsmodels.api as sm
import warnings 
warnings.filterwarnings("ignore")
import copy


# ## 1.Defining Problem Statement and Analyzing basic metrics - 

# ### About Walmart -
# Walmart is an American multinational retail corporation that operates a chain of supercenters, discount departmental stores, and 
# grocery stores from the United States. Walmart founded in 1962 by Sam Walton. Walmart has more than 100 million customers worldwide.
# Beyond retail, Walmart has also ventured into emcommerce, technology innovation, and sustainability initiatives, further solidyfying its position as key player in the modern reatail ecosystem.
# 
# ### Business Problem - 
# The Management team at Walmart Inc. wants to analyze the customer purchase behavior (specifically, purchase amount) against the customers's gender and the various other factors to help the business make better decisions. They want to understand if the spending habits differ between male and female customers: Do women spend more on Black Friday than men? 
# 
# ### Objective - 
# The objective of this project is to conduct a comprehensive analysis of customers purchase behavior, with a specific focus on pruchase amounts, in relation to customer gender during the Black Friday sales event at Walmart Inc. This study aims to provide valuable insights that can assist the management team at Walmart Inc. in making data - driven decisons.
# 

# 

# 
# 

# In[34]:


data = pd.read_csv("walmart_data.csv")


# In[35]:


data


# In[36]:


data.shape


# #### Dataset contains 550068  rows and 10 columns.

# In[37]:


data.info()


# 1. The colums User_id, Occupation, Marital_Status, Product_Category and Purchase are in integer datatype, rest of the columns are in object dataype.
# 2. There are no null values.

# In[38]:


data.nunique()


# The columns gender, age, city_category, stay_in_current_city_years and marital status can be converted into category datatype.

# In[39]:


# Converting some column into categorical datatype - 

dt = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years", "Marital_Status"]
for i in dt:
    data[i] = data[i].astype("category")


# In[40]:


data.info()


# ### Statistical Summary -

# In[41]:


data.describe()


# 1. Range of purchase amount is 12 dollars to 23961 dollars.
# 2. Mean purchase amount is 9264 dollars.
# 3. Median purchase amount is 8047 dollars.
# 4. Standard deviation of purchase amount is 5023 dollars.
# 5. Inter quartile range of purchase amount is 5823 to 12054 dollars.

# ## 2.Non-Graphical Analysis - 

# ### Value Counts and Unique Attributes -

# In[42]:


# Value counts for first 5 columns - 

for i in data.columns[:5]:
    print("Value counts for column", i,"-")
    print(data[i].value_counts())
   
    


# In[43]:


for i in data.columns[5:]:
    print("Value counts for column", i, "-")
    print(data[i].value_counts())


# 1. Product categories 5,1 and 8 sell more than other categories.
# 2. Product categories 17 and 9 sell the least.
# 3. Most frequent users have made almost 1000 purchases.
# 4. Least frequent users have made 6 to 7 purchases.
# 5. The most sold product is "P00265242".
# 6. People in age group 26-35 has more purchases than other age group.
# 7. People of city category B make more purchases than other city category.
# 8. Male users - 414259 and Female users - 135809
# 9. Unmarried people has more purchases than married people.

# ## 3. Missing Value & Outlier Detection -

# In[44]:


# Finding outliers using IQR method -

for i in ["Purchase"]:
    outliers = []
    p25 = np.percentile(data[i], 25)
    p75 = np.percentile(data[i], 75)
    
    iqr = p75 - p25
    max_cut = p75 + iqr*1.5
    min_cut = max(0, p25 - iqr*1.5)
    
    outliers = data.loc[(data[i]<min_cut) | (data[i]>max_cut), i]
    
    print("Outliers for the column", i, "-")
    print(outliers)
    print("Number of outliers - ", len(outliers))
    print("Percentage of outliers = ", round((len(outliers)/len(data[i]))*100,2), "%")


# 1. Purchase columns contains 2677 outliers.
# 2. This is 0.49% of total number of entries.

# In[45]:


# Missing value - 

data.isna().sum()


# There is no missing value in any column.

# ## 4. Visual Analysis - Univariate & Bivariate-

# In[46]:


# Purchase Analysis - Box Plot

plt.figure(figsize = (8,4))
sns.boxplot(data = data, x = "Purchase")
plt.title("Purchase Analysis")
plt.show()


# 1. Median purchase is about 8000 dollars.
# 2. Inter Quartile Range is 6000 to 12000 dollars.
# 3. There are many outliers.

# In[47]:


# Distribution plots - 

dt = ["Purchase", "Product_Category", "Occupation"]
plt.figure(figsize = (20,6))
for i in range(len(dt)):
    plt.subplot(1,3, i+1)
    sns.distplot(data[dt[i]])
    plt.title("Distribution of {feature}".format(feature = dt[i]))

plt.show()


# 1. Purchase amounts range between 0 and 25000.
# 2. Purchase amounts between 5000 to 10000 are maximum.
# 3. Product categories 1, 5 and 8 are most frequnetly purchased.
# 4. There are 20 product categories.
# 5. There are 21 different occupations.
# 6. People with occupations 0, 4 and 7 make more purchase than other occupations.

# In[48]:


# Creating bins of purchase amount - 
data["purchase_bins"] = pd.cut(data["Purchase"], bins = [0,5000,10000,15000,20000,25000])


# In[49]:


# Categorical variables - Count Plots

plt.figure(figsize =(20,16))

dt = ["Gender","Age","City_Category","Stay_In_Current_City_Years", "Marital_Status", "Product_Category", "purchase_bins", "Occupation"]

for i in range(len(dt)):
    plt.subplot(3,3, i+1)
    sns.countplot(x = data[dt[i]])
    plt.title("Count of different {dt_i} (s)".format(dt_i = dt[i]))
    if dt[i] in ["purchase_bins"]:
        plt.xticks(rotation = 90)
plt.show()


# 1. More purchases made by males than females.
# 2. People of age group 26-35 have made the maximum purchases.
# 3. People in cities of category B have made maximun number of purchases.
# 4. Unmarried people made more puchases than married people.
# 5. Products of category 1, 5 and 8 sold most frequently.
# 6. Purchases of amount (5000, 10000) were maximum in number.
# 7. People of having occupation 0,4 and 7 made more purchases than other occupations.

# In[50]:


# Purchase vs Categorical-Variable  - bi-variate boxplots 

plt.figure(figsize = (15,25))

dt = ["Gender","Age","City_Category","Stay_In_Current_City_Years", "Marital_Status", "Product_Category", "Occupation"]
for i in range(len(dt)):
    plt.subplot(4,2,i+1)
    sns.boxplot(x = data[dt[i]], y = data["Purchase"])
    plt.title("Purchase made by differnet {dt_i} (s)".format(dt_i = dt[i]))
    plt.ylabel("Purchase Amount")
    plt.grid()
plt.show()


# 1. Median purchase amounts of males and females are similar.
# 2. Median purchase amounts of all age group are similar.
# 3. Median purchase amounts of city of category C is slightly higher than other city categories.
# 4. Median purchase amounts of product category 10 is highest, category 19 is lowest.
# 5. Median purchase amounts of occupations 12 and 17 are slighlty higher than other occupations.

# In[51]:


# Converting Gender, Age and City_Category to numeric data - 

data_copy = data.copy()
data_copy["Gender"].replace({"F" : 1, "M" : 0}, inplace = True)
data_copy["Age"].replace({"0-17" : 1, "18-25" : 2, "26-35" : 3, "36-45" : 4, "46-50": 5, "51-55" : 6, "55+" : 7 },inplace = True)
data_copy["City_Category"].replace({"A" : 1, "B" : 2, "C" : 3}, inplace = True)


# In[52]:


# Finding Pearson Correlation between all the numerical features -

data_copy.corr()


# In[53]:


# Heatmap for the above correlation values - 

plt.figure(figsize = (8,4))
sns.heatmap(data_copy.corr().round(2), annot = True)
plt.title("Correlation between features", fontsize = 12)
plt.show()


# 1. There is no strong correlation of purchase with any feature.
# 2. There's a weak negative correlation between product category and purchase amount.
# 

# In[54]:


# Pair plot - 
# Blue - Male , Orange - Female

sns.pairplot(data_copy.sample(5000).iloc[:,1:], hue = "Gender")
plt.show()


# 1. Most popular product category among males is 1.
# 2. Most popular product category among female is 5. 
# 3. Both males and females of city category B make more purchase compared to city categories A and C.
# 4. Females purchase products of category 4, 11, 15, 17 and 18 less often.
# 5. Females with occupation 0-10 made more puchases than females with occupations 11-20.

# ## Sampling -  

# In[55]:


# Creating kde plots to check if it appears gaussian - 

plt.figure(figsize = (6,10))
x = 1
for j in ["M", "F"]:
    means = []
    for i in range(1000):
        dt = data.loc[data["Gender"]==j, "Purchase"].sample(1000)
        avg = dt.mean()
        means.append(avg)
        
    plt.subplot(2,1,x)
    sns.kdeplot(x = means)
    
    if j == "M":
        gen = "male"
        means_m = means
    else:
        gen = "female"
        means_f = means
    
    plt.title("Distribution of mean expenses for {g} users".format(g = gen), fontsize = 12)
    plt.xlabel("Mean")
    x += 1
    plt.show()


# The distributions appear to be gaussian.

# In[56]:


# Finding differnet Confidence Intervals for Males and Females - 

for i in ["males", "females"]:
    print("For {g}-".format(g = i))
    if i == "males":
        means = means_m
        gen = "M"
    else:
        means = means_f
        gen = "F"
        
    print("Mean of sample means =", np.mean(means))
    print("Population mean =", np.mean(data.loc[data["Gender"]== gen, "Purchase"]))
    print("Standard deviation of means (Standard Error) =", np.std(means))
    print("Standard deviation of population = ", data.loc[data["Gender"] == gen, "Purchase"].std())
    print("99% CONFIDENCE INTERVAL for mean expense by {g} users -".format(g = i))
    print((np.percentile(means, 0.5).round(2), np.percentile(means, 99.5).round(2)))
    print("95% CONFIDENCE INTERVAL for mean expense by {g} users -".format(g = i))
    print((np.percentile(means, 2.5).round(2), np.percentile(means, 97.5).round(2)))
    print("90% CONFIDENCE INTERVAL for mean expense by {g} users -".format(g = i))
    print((np.percentile (means, 5).round(2), np.percentile(means, 95).round(2)))
    print("-" * 80)
  


# 1. Mean purchase amount for Males = 9437.52
# 2. Mean purchase amount for Females = 8734.56
# 3. 95% Confidence Interval for purchase amount of females is less than males.
# 4. With 95% Confidence Interval we can say females spend less than males.

# ### Married and Unmarried users - 

# In[57]:


# Taking samples of 1000 entries, creating kde plots to check if it appears gaussian - 

plt.figure(figsize = (6,10))
x = 1
for j in [1,0]:
    means = []
    for i in range(100):
        dt = data.loc[data["Marital_Status"] == j, "Purchase"].sample(1000)
        avg = dt.mean()
        means.append(avg)
    
    plt.subplot(2,1,x)
    sns.kdeplot(x = means)
    
    if j == 0:
        ms = "unmarried"
        means_mr = means
    else:
        ms = "married"
        means_umr = means
    
    plt.title("Distribution of mean expenses for {m} users".format(m = ms), fontsize = 12)
    plt.xlabel("Mean")
    x += 1

plt.show()


# In[58]:


# Different Confidence Interval for mean expense by married and unmarried customers - 

for i in ["married", "unmarried"]:
    print("For {m}-".format(m = i))
    if i == "married":
        means = means_mr
        ms = 1
    else:
        means = means_umr
        ms = 0
    
    print("Mean of sample means =", np.mean(means))
    print("Population mean =", np.mean(data.loc[data["Marital_Status"] == ms, "Purchase"]))
    print("Standard deviation of means (Standard Error) =", np.std(means))
    print("Standard deviation of population =", data.loc[data["Marital_Status"] == ms, "Purchase"].std())
    print("99% CONFIDENCE INTERVALL for mean expense by {m} users-".format(m = i))
    print((np.percentile(means, 0.5).round(2), np.percentile(means, 99.5).round(2)))
    print("95% CONFIDENCE INTERVAL for mean expense by {m} users-".format(m = i))
    print((np.percentile(means, 2.5).round(2), np.percentile(means, 97.5).round(2)))
    print("90% CONFIDENCE INTERVAL for mean expense by {m} users-".format(m = i))
    print((np.percentile(means, 5).round(2), np.percentile(means, 95).round(2)))
    print("-" * 80)
    


# 1. Mean expense by married customers is 9261.17
# 2. Mean expense by unmarried customers is 9265.90
# 3. There is overlap between 90%, 95% and 99% Confidence Intervals for both.

# ### For different age groups-

# In[59]:


# Plotting KDE plots to see if their distribution looks gaussian - 

plt.figure(figsize = (20,18))
x = 1
for j in ["0-17", "55+", "26-35", "46-50", "51-55", "36-45", "18-25"]:
    means = []
    for i in range(100):
        dt = data.loc[data["Age"] == j, "Purchase"].sample(1000)
        avg = dt.mean()
        means.append(avg)
    
    plt.subplot(3,3,x)
    sns.kdeplot(x = means)
    
    if j == "0-17":
        means_0 = means
    elif j == "55+":
        means_55 = means
    elif j == "26-35":
        means_26 = means
    elif j == "46-50":
        means_46 = means
    elif j == "51-55":
        means_51 = means
    elif j == "36-45":
        means_36 = means
    else:
        means_18 = means
        
    plt.title("Distribution of mean expenses for users of age group {a}".format(a = j), fontsize = 10)
    plt.xlabel("Mean")
    x += 1

plt.show()



# In[60]:


# Finding Confidence Intervals for mean purchase for each age group - 

for i in ['0-17', '55+', '26-35', '46-50', '51-55', '36-45', '18-25']:
    print("For {m}-".format(m = i))
    if i == '0-17':
        means = means_0
    elif i == '55+':
        means = means_55
    elif i == '26-35':
        means = means_26
    elif i == '46-50':
        means = means_46
    elif i == '51-55':
        means = means_51
    elif i == '36-45':
        means = means_36
    else:
        means = means_18
    
    print("Mean of samples means =", np.mean(means))
    print("Population mean =", np.mean(data.loc[data["Age"]==i, "Purchase"]))
    print("Standard deviation of means (Standard Error) =", np.std(means))
    print("Standard deviation of population =", data.loc[data["Age"]==i, "Purchase"].std())
    print("99% CONFIDENCE INTERVAL for mean expense by users of age group {a}-".format(a = i))
    print((np.percentile(means, 0.5).round(2), np.percentile(means, 99.5).round(2)))
    print("95% CONFIDENCE INTERVAL for mean expense by users of age group {a}-".format(a = i))
    print((np.percentile(means, 2.5).round(2), np.percentile(means, 97.5).round(2)))
    print("90% CONFIDENCE INTERVAL for mean expense by users of age group {a}-".format(a = i))
    print((np.percentile(means, 5).round(2), np.percentile(means, 95).round(2)))
    print("-" * 50)


# 1. 99% Confidence Interval for 0-17 is less than 51-55 without overlap.
# 2. With  99% confidence we can say that expense of 0-17 is less compared to expense of 51-55 ages.

# ## Insights - 

# #### 1. Sample Size - 
# 
# The analysis highlights the importance of sample size in estimating population parameters. It suggests that as the sample size increases, the confidence intervals become narrower and more precise. In business, this implies that larger sample sizes can provide more reliable insights and estimates.
# 
# #### 2. Confidence Intervals -
# 
# From the above analysis, we can see that except for the Sample Size of 100, the confidence interval do not overlap as the sample size increases. This means that there is a statistically significant difference between the average spending per transaction for men and women within the given samples.
# 
# #### 3. Population Average -
# 
# We are 95% confident that the true population average for males falls between $9,393 and $9,483, and for females, it falls between $8,692 and $8,777.
# 
# #### 4. Women spend less -
# 
# Men tend to spend more money per transaction on average than women, as the upper bounds of the confidence intervals for men are consistently higher than those for women across different sample sizes.
# 
# 5. Products of categories 1, 5 and 5 can be kept in inventory as well as made easily visible in the stores.
# 6. Offers/ rewards can be given on purchases above 12000 dollarss to nudge customers to make more purchases.
# 7. More popular products among people with occupations 0, 4 and 7 can be kept in store.
# 8. Walmart products like P00265242 and P00025442 are selling a lot. Products like P00056342 and P00350742 are selling very less.
# 9. Ads can be targeted towards unmarried people because of sell.
# 10. Ads for slightly expensive products can be targetted towards people with occupation 12 and 17.
# 11. Ads can be targeted towards people who have spent between 1 to 2 years in their cities.

# ## Recommendations - 
# 
# #### 1.Target Male Shoppers -
# 
# Since male customers account for a significant portion of Black Friday sales and tend to spend more per transaction on average, Walmart should tailor its marketing strategies and product offerings to incentivize higher spending among male customers while ensuring competitive pricing for female-oriented products.
# 
# #### 2. Focus on 26 - 45 Age Group -
# 
# With the age group between 26 and 45 contributing to the majority of sales, Walmart should specifically cater to the preferences and needs of this demographic.This could include offering exclusive deals on products that are popular among this age group.
# 
# #### 3. Engage Younger Shoppers -
# 
# Knowing that customers in the 0 - 17 age group have the lowest spending per transaction, Walmart can try to increase their spending per transaction by offering them more attractive discounts, coupons, or rewards programs. It's essential to start building brand loyalty among younger consumers.
# 
# #### 4. Post-Black Friday Engagement -
# 
# After Black Friday, walmart should engage with customers who made purchases by sending follow-up emails or offers for related products. This can help increase customer retention and encourage repeat business throughout the holiday season and beyond.
# 
# 5. Give offers on purchases above 12000 to nudge customers to spend more.
# 6. Target ads towards people of city category B.
# 7. Target ads for product which cost between 9151 and 9790 towards males.
# 8. Target ads for product which cost between 8507 and 9051 towards females.
# 9. Recognizing that customers in the 51 - 55 age group have the highest spending per transaction, Walmart can explore opportunities to enhance the shopping experience for this demographic. 
# 10. Age group between 26 - 45 accounts to almost 60% of the total sales suggesting that Walmart's Black Friday sales are most popular among these age groups.
# 

# In[ ]:




