#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data

# In[5]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
data_scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_14d", "price_change_percentage_30d", "price_change_percentage_60d", "price_change_percentage_200d", "price_change_percentage_1y"]])


# In[6]:


# Get a snapshot of the dataset
data_scaled[0:10]


# In[7]:


# Create a DataFrame with the scaled data
data_scaled_df = pd.DataFrame(data_scaled,columns=["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_14d", "price_change_percentage_30d", "price_change_percentage_60d", "price_change_percentage_200d", "price_change_percentage_1y"])
  


# Copy the crypto names from the original data
data_scaled_df["coin_id"] = df_market_data.index


# Set the coinid column as index
data_scaled_df = data_scaled_df.set_index("coin_id")


# Display sample data
data_scaled_df.head()


# ---

# ### Find the Best Value for k Using the Original Data.

# In[8]:


# Create a list with the number of k-values from 1 to 11
# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))


# In[9]:


# Create an empty list to store the inertia values
inertia = []


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=1)
    model.fit(data_scaled_df)
    inertia.append(model.inertia_)


# In[10]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}


# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)
df_elbow.head()


# In[11]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_1 = df_elbow.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve",
    xticks=k
)
elbow_1


# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** 

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data

# In[12]:



# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)


# In[13]:


# Fit the K-Means model using the scaled data
model.fit(data_scaled_df)


# In[14]:


# Predict the clusters to group the cryptocurrencies using the scaled data
kmeans_predictions = model.predict(data_scaled_df)


# Print the resulting array of cluster values.
print(kmeans_predictions)


# In[15]:


# Create a copy of the DataFrame
crypto_predictions_df = data_scaled_df.copy()


# In[16]:


# Add a new column to the DataFrame with the predicted clusters
crypto_predictions_df["predictions"] = kmeans_predictions


# Display sample data
crypto_predictions_df.head()


# In[17]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
cluster_1 = crypto_predictions_df.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    title="KMeans Clustering using original data",
    by="predictions",
    hover_cols="coin_id"
    
    
).opts(yformatter="%.0f")
cluster_1


# ---

# ### Optimize Clusters with Principal Component Analysis.

# In[18]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[19]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
pca_scaled = pca.fit_transform(data_scaled_df)



# View the first five rows of the DataFrame.
pca_scaled[:5]


# In[20]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_


# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** About 89% of the total variance is condensed into the 3 PCA variables

# In[21]:


# Create a new DataFrame with the PCA data.
# Creating a DataFrame with the PCA data
pca_df = pd.DataFrame(pca_scaled,columns=["PCA1", "PCA2", "PCA3"])

# Copy the crypto names from the original data
pca_df["coin_id"] = data_scaled_df.index

# Set the coinid column as index
pca_df = pca_df.set_index("coin_id")


# Display sample data
pca_df.head(10)


# ---

# ### Find the Best Value for k Using the PCA Data

# In[22]:


# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))


# In[23]:


# Create an empty list to store the inertia values
inertia2 = []


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=1)
    model.fit(pca_df)
    inertia2.append(model.inertia_)


# In[24]:



# Create a dictionary with the data to plot the Elbow curve
elbow2_data = {"k":k, "inertia":inertia2}

# Create a DataFrame with the data to plot the Elbow curve
elbow_2_df = pd.DataFrame(elbow2_data)
elbow_2_df.head()


# In[25]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
pca_elbow = elbow_2_df.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve 2",
    xticks=k
)
pca_elbow


# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** The Best value of k is 4
# No. The value of k is the same as the original data
# Cluster Cryptocurrencies with K-means Using the PCA Data

# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# In[26]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)


# In[27]:


# Fit the K-Means model using the PCA data
model.fit(pca_df)


# In[28]:


# Predict the clusters to group the cryptocurrencies using the PCA data
pca_predictions = model.predict(pca_df)

# Print the resulting array of cluster values.
print(pca_predictions)


# In[29]:


# Create a copy of the DataFrame with the PCA data
pca_predictions_df = pca_df.copy()


# Add a new column to the DataFrame with the predicted clusters
pca_predictions_df ["pca_predictions"] = pca_predictions


# Display sample data
pca_predictions_df.head()


# In[30]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
pca_cluster = pca_predictions_df.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="pca_predictions",
    title="Clustering using PCA",
    hovercols="coin_id"
)
pca_cluster


# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# In[31]:



# Composite plot to contrast the Elbow curves
elbow_composite = elbow_1 + pca_elbow
elbow_composite


# In[32]:


# Composite plot to contrast the clusters
cluster_composite = cluster_1 + pca_cluster
cluster_composite


# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** The results are the same for both techniques. However, clustering using PCA is more grouped and clear when using data like cryptocurrency. The inertia for the PCA data is smaller because we are using PCA which reduces the dimensions of a dataset. This is why the inertia is low.

# In[ ]:




