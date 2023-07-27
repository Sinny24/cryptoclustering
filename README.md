# cryptoclustering
Data Preparation:

To normalize the data from the CSV file, we'll use the StandardScaler() module provided by scikit-learn. After that, we'll create a DataFrame with the scaled data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

Finding the Best Value for k Using the Original Scaled DataFrame:

To determine the optimal value for k, we'll use the elbow method, which involves the following steps:
1. Creating a list with the number of k values ranging from 1 to 11.
2. Initializing an empty list to store the inertia values.
3. Using a for loop to compute the inertia with each possible value of k.
4. Creating a dictionary with the data to plot the elbow curve.
5. Plotting a line chart with all the inertia values computed for different values of k to visually identify the best value for k.

The question we'll answer in the notebook is: What is the best value for k?

Clustering Cryptocurrencies with K-means Using the Original Scaled Data:

Once we determine the best value for k, we'll proceed to cluster the cryptocurrencies using the original scaled data by following these steps:
1. Initializing the K-means model with the best value for k.
2. Fitting the K-means model using the original scaled DataFrame.
3. Predicting the clusters to group the cryptocurrencies using the original scaled DataFrame.
4. Creating a copy of the original data and adding a new column to store the predicted clusters.
5. Creating a scatter plot using hvPlot with "PC1" on the x-axis and "PC2" on the y-axis, where the graph points will be colored based on the labels obtained from K-means. Additionally, we'll add the "coin_id" column in the hover_cols parameter to identify each cryptocurrency represented by the data points.

Optimizing Clusters with Principal Component Analysis (PCA):

To reduce the features and optimize the clusters, we'll perform a PCA on the original scaled DataFrame to reduce the features to three principal components. We'll retrieve the explained variance to determine how much information each principal component contributes. The question we'll answer in the notebook is: What is the total explained variance of the three principal components?

Next, we'll create a new DataFrame with the PCA data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

Finding the Best Value for k Using the PCA Data:

Similar to the previous step, we'll use the elbow method on the PCA data to find the best value for k. The steps include:
1. Creating a list with the number of k values ranging from 1 to 11.
2. Initializing an empty list to store the inertia values.
3. Using a for loop to compute the inertia with each possible value of k.
4. Creating a dictionary with the data to plot the elbow curve.
5. Plotting a line chart with all the inertia values computed for different values of k to visually identify the best value for k.

The question we'll answer in the notebook is: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

Clustering Cryptocurrencies with K-means Using the PCA Data:

Once we determine the best value for k using the PCA data, we'll proceed to cluster the cryptocurrencies using the PCA data. The steps will be similar to clustering with the original data, but now we'll use the reduced PCA features. We'll create a scatter plot using hvPlot, with "price_change_percentage_24h" on the x-axis and "price_change_percentage_7d" on the y-axis, where the graph points will be colored based on the labels obtained from K-means. Additionally, we'll add the "coin_id" column in the hover_cols parameter to identify each cryptocurrency represented by the data point.

The final question to answer is: What is the impact of using fewer features to cluster the data using K-Means?
