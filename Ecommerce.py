#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore")


# # Get the Data

# In[2]:


customers=pd.read_csv('Ecommerce Customers.csv')


# # Check the head of customers, and check out its info() and describe() methods

# In[3]:


customers.head()


# In[4]:


customers.info()


# In[5]:


customers.describe()


# # Exploratory Data Analysis

# In[6]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers,color="#7D3C98")


# In[7]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers,color='#28B463')


# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot 

# In[8]:


sns.pairplot(customers)


#  Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership

# In[9]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers,color="#CB4335")


# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership

# In[10]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers,markers="+")


# Construct a heatmap of these correlations

# In[12]:


sns.heatmap(customers.corr(),cmap='YlOrRd',annot=True)


# # Training and Testing Data

# In[13]:


x=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[14]:


y=customers['Yearly Amount Spent']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# # Training and Testing Data

# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lm=LinearRegression()


# In[23]:


lm.fit(x_train,y_train)


# # Print out the coefficients of the model

# In[24]:


print("Coefficients: \n",lm.coef_)
cdf=pd.DataFrame(lm.coef_,x.columns,columns=['Ceoff'])
cdf


# In[25]:


new_cdf=cdf.reset_index()


# In[26]:


new_cdf=pd.DataFrame(new_cdf)
new_cdf.columns=(["Parameter","Coefficients"])
new_cdf


# In[27]:


print("Intreceft :\n",lm.intercept_)


# In[28]:


sns.barplot(x="Parameter",y="Coefficients",data=new_cdf,hue="Parameter")


# # Predicting Test Data

# In[29]:


predict=lm.predict(x_test)


# In[30]:


plt.scatter(y_test,predict)
plt.xlabel("Y test")
plt.ylabel("Y predict")


# # Evaluating the Model

# In[31]:


from sklearn import metrics


# In[32]:


print("MAS :",metrics.mean_absolute_error(y_test,predict))
print("MSE :",metrics.mean_squared_error(y_test,predict))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,predict)))


# # Residuals
# Plot a histogram of the residuals and make sure it looks normally distributed

# In[33]:


sns.distplot((y_test-predict),bins=75,color="#2ECC71")


# # Conclusion

# In[34]:


print(new_cdf)


# # Interpreting the coefficients:

# 
# 
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# 

# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# 

# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# 

# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent

# In[ ]:




