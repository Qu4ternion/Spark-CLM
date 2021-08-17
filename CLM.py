# -*- coding: utf-8 -*-
"""
@author: Reda Gossony
"""

import pandas as pd
import numpy  as np
import scipy.stats as stats
import math
import statistics
import statsmodels.api as sm

import sklearn.preprocessing
from   sklearn.linear_model  import LogisticRegression, Ridge
from   sklearn.cluster       import KMeans

import matplotlib.pyplot     as plt
from   bokeh.plotting        import figure, output_file, show 
from   ridgeplot             import ridgeplot
import seaborn               as sb
import shap


'''
###############
Data wrangling
###############
'''

# import dataframe:
df = pd.read_excel(r'C:\Users\Acer\Desktop\CLM project\data\CLM.xlsx')

# Check columns:
df.columns

# Check the first few observations:
df.head()

# drop the 2 redundant index columns:
df = df.drop(['Unnamed: 0', 'index', 'client_id',
              'churn_date', 'subscription_date'], 1)

# Set all discrete columns to Pandas 'category' type:
categoricals = ['churned', 'sale/coupon', 'add_shown', 'product_bought',
                'called_customer_service','engaged_last_30', 'credit/debit',
                'subscription_type']

for column in categoricals:
    df[column] = df[column].astype('category')

# Show unique levels of each categorical variable:
for var in categoricals:    
    print(var, list(df[var].unique()), '\n')
    
# Group by month for later:
dt = df
dt.index = pd.DatetimeIndex(df.transaction_date)
data_by_month = list(dt.groupby(by=dt.index.month))

# Save "products bought" and "subscription type" for later:
products = df.product_bought
subscription = df.subscription_type

# One-hot encoding of nominal variables: (save levels first for next)
nominal = ['product_bought', 'subscription_type']
levels = list(df[nominal[0]].unique()) + list(df[nominal[1]].unique())

for var in nominal:
    df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var])], axis=1)

# Set newly encoded levels to Categorical:
for level in levels:
    df[level] = df[level].astype('category')


'''
####################
Exploratory analysis
####################
'''

# Count missing values:
df.isnull().count()        # no missing values

# Descriptive statistics:
df.describe()

# Histogram:
df.hist()

# Scatterplot matrix on the numerical data:
numeric_data = df._get_numeric_data()
pd.plotting.scatter_matrix(numeric_data)

# Show only variables of interest:
df.plot.scatter('retention_time', 'cLTV')
df.plot.scatter('client_age', 'transaction_value')

# Enhanced version of the scatter matrix above:
df = df._get_numeric_data()
a_min = df['cLTV'].min()
a_max = df['cLTV'].max()
bins = np.linspace(a_min, a_max, 10)  # Divide between a_min and vcc_max into 10
df['a_range'] = pd.cut(df['cLTV'], bins=bins)

sb.set_style('white')
sb.set()
g = sb.pairplot(df, hue='a_range', hue_order=df['a_range'].cat.categories, palette='YlGnBu', aspect=1.5)

       
handles = g._legend_data.values()    # Change legends location
labels = g._legend_data.keys()
g._legend.remove()
g.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=3)
plt.show()


# Version 3:
sb.set_theme(style='ticks')
g = sb.pairplot(df._get_numeric_data(), hue = 'churned')

handles = g._legend_data.values()   # Change legends location
labels = g._legend_data.keys()
g._legend.remove()
g.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=3)
plt.show()


# Pie charts:
subscription_prop = subscription.value_counts()/len(products)*100
labels = ['Basic', 'Platinum', 'Gold']
plt.pie(subscription_prop, labels=labels,
        shadow=True, autopct='%1.1f%%',
        explode=(0.1, 0, 0)
        )

# Ridge plot of transaction value distribution per month:
distribution = []
for month in range(12):
    distribution.append(data_by_month[month][1].visits_number)

fig = ridgeplot(samples = distribution,
                labels = ['January', 'February', 'March', 'April', 'May',
                          'June', 'July', 'August', 'September', 'October',
                          'November', 'December'])

fig.update_layout(title= '<b>How average clicks per month change over the year</b>',
                  yaxis_title = 'Click density',
                  xaxis_title = 'Average clicks per day')

# Export to HTML (must be visualized using HTTP server)
fig.write_html(r'C:\Users\Acer\Desktop\ridge.html')

# Correlation matrix:
corr = numeric_data.corr()

# Correlation heatmap:
heat_map = sb.heatmap(corr)


'''Hexagon plot'''
# file to save the model 
output_file(r"C:\Users\Acer\Desktop\CLM project\img\hex.html") 
       
# instantiating the figure object 
graph = figure(title = "Hexagraph of Transaction Value and Client Age") 

# x-axis label
graph.xaxis.axis_label = "Transaction value"
       
# y-axis label
graph.yaxis.axis_label = "Client Age"
    
# Plot the 2 most correlated variables:
x = df.transaction_value
y = df.client_age
  
# plotting the graph 
graph.hexbin(x, y, size = 0.2) 
     
# displaying the model 
show(graph)

'''
After executing 'show(graph)', it will be stored in the "output_file" path.
Bokeh shows graphs in HTML. So start an HTTP server over CMD using:
    
    "python -m http.server"
    
And use the browser to go to "localhost:PORT" by using the port shown in CMD.
Then browse to "hex.html" and open it to see the graph.
'''

'''
###############
Data processing
###############
'''
# Select only numeric features:
dt = df._get_numeric_data()

# Define scaler:
scaler = sklearn.preprocessing.StandardScaler()
#scaler = sklearn.preprocessing.Normalizer()

# Fit the scaler:
scaler.fit(dt)
scaled_columns = scaler.transform(dt)
scaled_data = pd.DataFrame(scaled_columns)

# re-assign the original column names:
scaled_data.columns = df._get_numeric_data().columns

# Show scatter plots again but this time on normalized data:
pd.plotting.scatter_matrix(scaled_data)    # shows better
                                           # relationships
# New NUMERIC scaled data:
scaled_data

'''
Before scaling the data the relationships weren't clear.
After scaling it we can see some relationships spring up.
'''

# Check columns in scaled_data NOT IN df:
diff = list(set(df.columns) - set(scaled_data.columns))

# Unscaled data (dropping the dates because we don't need them)
unscaled = df[diff]

# Concatenate the two frames:
new = pd.concat([scaled_data, unscaled], axis = 1).drop(columns = 'churned')

logit = LogisticRegression()
logit.fit(new, df.churned)

# Show coefficients:
logit.coef_

# Pack every coefficient with its variable:
x = list(new.columns)
y = list(logit.coef_)[0]
z = zip(y,x)

# Show zipped:
z = list(z)
z

# Sorted low to high:
z = sorted(z)

# We will plot the Feature Importance chart
coefs = pd.DataFrame(z)
coefs['rank'] = range(len(coefs))

# Initiate empty figure and axes:
fig, ax = plt.subplots()
ax.barh(coefs['rank'], coefs[0])
ax.set_yticks(coefs['rank'])
ax.set_yticklabels(coefs[1])
ax.set_xlabel('Feature importance')
ax.set_title('Effect magnitude of each variable on customer Churn')

# SHAP (shapley) values:
numeric_data = new._get_numeric_data()
shap_logit = LogisticRegression().fit(numeric_data, df.churned)
explainer = shap.LinearExplainer(shap_logit, numeric_data)
shap.summary_plot(explainer.shap_values(numeric_data), numeric_data)

# SHAP Feature importance
explainer = shap.Explainer(logit, new)
shap_values = explainer(new)
shap.plots.bar(shap_values, max_display = 20)

# Shap of transaction value colored by retention time:
explainer2 =  shap.Explainer(shap_logit, numeric_data)
shap_values2 = explainer2(numeric_data)
shap.plots.scatter(shap_values2[:,'transaction_value'],
                   color=shap_values2, x_jitter= 0)

# Waterfall dependence:
shap.plots.waterfall(shap_values[0], max_display=20)


'''
#######################
cLTV optimization model
#######################
'''

# Fit linear regression model on cLTV as target:
reg_model = Ridge(alpha=1.5)
x = new.drop(columns='cLTV')
y = new.cLTV
reg_model.fit(x,y)

# Show estimated coefficients:
reg_model.coef_

# Feature importance graph:
explainer = shap.LinearExplainer(reg_model, x)
shap_values = explainer(x)

# Bee swarm graph:
shap.plots.beeswarm(shap_values)

# Organize coefficients:
y = list(reg_model.coef_)
z = zip(y,x)
z = list(z)
z = sorted(z)

# We will plot the Feature Importance chart
coefs = pd.DataFrame(z)
coefs['rank'] = range(len(coefs))

fig, ax = plt.subplots()
ax.barh(coefs['rank'], coefs[0])
ax.set_yticks(coefs['rank'])
ax.set_yticklabels(coefs[1])
ax.set_xlabel('Feature importance')
ax.set_title('Effect magnitude of each variable on customer Churn')


'''
###################
Client segmentation
###################
'''

# K-means clustering algorithm:
km = KMeans(n_clusters=2)
data = new[['client_age', 'transaction_value']]
km.fit(data)

# Predict clusters:
predicted = km.fit_predict(data)

# Visualize clusters by color:
fig, ax = plt.subplots()
for i in range(len(predicted)):
    if predicted[i] == 1:
        ax.scatter(data.client_age[i], data.transaction_value[i], c='red',
                   marker = 'v', edgecolor='black')
    else:
        ax.scatter(data.client_age[i], data.transaction_value[i], c='blue',
                   marker = '*', edgecolor='black')

ax.set_ylabel('Scaled transaction values')
ax.set_xlabel('Scaled ages')
ax.set_title('K-mean clusters of Age & Transaction Value')
plt.grid()
plt.show()


# See the "d3.js" file for the rest of Client Segmentation using Decision trees.

'''
#############################
A/B testing of Advertisements
#############################
'''

# Customers that were shown the ad:
shown = df.transaction_value[df.add_shown == 1]

# Customers that weren't shown the ad:
not_shown = df.transaction_value[df.add_shown == 0]

# Make sure everything adds up (we didn't miss any observation):
assert(len(shown) + len(not_shown) == len(df))

# Merge the two columns:
a_b = pd.concat( [shown.reset_index(drop= True),
                  not_shown.reset_index(drop= True)],
                  axis = 1, ignore_index = True)

# Rename:
a_b.columns = ['Shown', 'Not shown']

# Student's t-test for difference in mean:
t_test = stats.ttest_ind(shown, not_shown)
t_test

'''
The variances of the 2 samples are NOT EQUAL. So, theoretically, we shouldn't
be using a t-test: since it assumes equal variances. We should use Welch's
test which relaxes this assumption.

However, the t-test and Welch test are asymptotically equivalent: meaning the
bigger the sample, the closer their two results will be.

We can see this empirically when we use Welch's test.
'''

# The test shows that there's a significant difference in transaction value
# between groups that were shown the ad, and those that weren't => the ads
# are efficient.
welch_test = stats.ttest_ind(shown, not_shown, equal_var = False)

# Visualize the difference:
fig, ax = plt.subplots()
plt.boxplot([shown, not_shown])
ax.set_xticklabels(['Shown', 'Not shown'])
plt.title('Boxplot of the two samples')
plt.ylabel('Transaction value')

# Difference in mean between the two samples (statistically significant)
np.mean(shown) - np.mean(not_shown)

# Histograms of the two samples:
fig, ax = plt.subplots()
plt.hist(shown, bins=10, alpha=0.5, label="Shown", density=True)
plt.hist(not_shown, bins=10, alpha=0.5, label="Not shown", density=True)

mu_shown = statistics.mean(shown)    # Plot normal distribution for 'shown'
variance_shown = statistics.variance(shown)
sigma_shown = math.sqrt(variance_shown)
x = np.linspace(mu_shown - 3*sigma_shown, mu_shown + 3*sigma_shown, 100)
plt.plot(x, stats.norm.pdf(x, mu_shown, sigma_shown), color='blue')

mu = statistics.mean(not_shown)     # Plot normal distribution for 'not shown'
variance = statistics.variance(not_shown)
sigma = math.sqrt(variance)
y = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(y, stats.norm.pdf(y, mu, sigma), color='red')

# Show all plots with a legend:
plt.legend(loc='upper right')
plt.title("Difference in transaction value between Shown and Not shown")
plt.xlabel('Transaction value')
plt.ylabel('Density')
plt.show()

# Show the two samples in a Ridge Plot via Kernel Density Estimation:
data = [shown, not_shown ]

fig = ridgeplot(samples= data, labels = ['Shown', 'Not shown'])
fig.update_layout(title= '<b>Ridge plot of Shown vs Not shown</b>',
                  yaxis_title = 'Shown advertisement or not?')
fig.write_html(r'C:\Users\Acer\Desktop\ridge.html')

# Q-Q plot:
reshaped = np.array(shown).reshape(1,-1)
sm.qqplot(reshaped, line = '45')


