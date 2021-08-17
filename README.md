# Spark-CLM
Revenue Optimization based on Churn Prevention, advertisement A/B testing and Customer Segmentation using Machine Learning (XGBoost, Logistic Regression, etc.) and Data Visualization techniques with Hyperparameter optimization executed on a distributed Spark cluster.

## Goals:
▪ Optimize revenue and cLTV (Customer Life-time Value).  
▪ Minimize customer churn (i.e. customers leaving the company).  
▪ Develop a churn prediction model for pre-emptive intervention.  
▪ Segment customers and detect the most profitable profiles to target.   
▪ Perform A/B testing to assess advertisement effectiveness.  

## Technologies :
▪ Spark, Python, scikit-learn, Pandas, numpy, Scipy, seaborn, matplotlib, bokeh, d3.js, http.server

## Methods :
▪ XGBoost, Ridge Regression, SHAP, GridSearch hyperparameter tuning, Kernel Density Estimation, Logistic regression, K-means, Cross Validation.  
▪ Student’s t-test, Welch test, precision/recall, f1-score, regularization.

## Data :
▪ 100,000 observations of a company's customer transactions, each comprising 20 variables collected from their app over the 2000 – 2021 period.   
▪ Pre-processing: data cleaning/wrangling, normalization, standardization, One-hot encoding.    
  


### Exploratory analysis:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/plot1.png?raw=true)

Just by comparing the different numerical variables, we can already begin to discern some interesting relationships:

▪ On the diagonal density graphs, we see that these are mixed distributions: we therefore have subpopulations within each variable that may require idiosyncratic treatments. 

▪ There is an apparent positive correlation between the age of the client and the amount paid in each transaction : the older the client, the higher the amount paid. This customer profile is therefore more profitable per transaction. 

▪ Presence of an exponential relationship between the length of time a customer is retained and his cLTV (i.e. long-term profitability of the customer): this indicates that the more a customer is retained, the more the amount he is willing to pay increases more than proportionally.  

We could re-draw this same figure, but this time coloring the scatterplot according to whether the customer has left the company or not (i.e. by churn):
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/plot2.png?raw=true)

This new configuration gives rise to even more interesting relationships:

▪ Once again, on the diagonal densities, we see that there are now 2 mixed distributions: one in orange, for customers who left the company, and the other in blue for those who stayed.

▪ Note that there are now sub-populations only for the variables "Transaction value" and "Age of customer". While the rest of the variables have homogeneous populations.

▪ We can also detect other very relevant relationships: we note that the higher the transaction value, the less likely a customer is to leave, and the higher his cLTV and profit margin: the company should thus incentivize clients to pay more per transaction (e.g. through upselling).
Let’s now analyze the the association between the variables through a correlation heatmap :

![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/heatmap.png?raw=true)

As previously discerned:
▪ There is a strong relationship between customer age and transaction value, 
▪ A strong association between profit margin and cLTV, 
▪ A moderate association between retention time and cLTV, 
▪ The rest of the variables are slightly associated.

We now analyze the range of distribution of the two most correlated variables, the age of the client and the amount disbursed (i.e. transaction value):

![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/hexagraph.png?raw=true)

We now turn to the variable "Visits per day" which gives us the number of times the user had visited the application on the day of the transaction. When we analyze this variable normally, we find that it does not present us with any useful information. However, an interesting pattern emerges when we break down this variable by month of the year:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/clicks_distribution.png?raw=true)

### Segmentation:

#### C&RT:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/tree_graph.PNG?raw=true)

#### K-means:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/k-means.png?raw=true)

### Feature importance & SHAP:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/feature_importance.png?raw=true)
  
  _________________
  
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/importance.png?raw=true)
  _________________

![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/shap.png?raw=true)
  _________________

### A/B test:
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/t-test.png?raw=true)
![alt text](https://github.com/Qu4ternion/Spark-CLM/blob/main/img/box.png?raw=true)
