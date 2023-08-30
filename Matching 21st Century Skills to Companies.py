# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:32:58 2023

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Read the CSV file
df = pd.read_csv('organizations.csv')

# Convert the "Description" field to lowercase for easier keyword matching
df['Full Description'] = df['Full Description'].str.lower()

# Remove possible null values (NaN)
df = df.dropna(subset=['Full Description'])

# Remove duplicate values
df = df.drop_duplicates(subset=['Organization Name'])

# Additional data cleaning operations can be performed here, such as removing unnecessary columns, filling in missing values, etc.

# Save the cleaned data
df.to_csv('cleaned_organizations.csv', index=False)

print("Data cleaning completed!")
    

# Read the CSV file
df = pd.read_csv('organizations.csv')

# Convert the "Description" field to lowercase for easier keyword matching
df['Full Description'] = df['Full Description'].str.lower()

# Remove possible null values (NaN)
df = df.dropna(subset=['Full Description'])

# Remove duplicate values
df = df.drop_duplicates(subset=['Organization Name'])

# Additional data cleaning operations can be performed here, such as removing unnecessary columns, filling in missing values, etc.

# Save the cleaned data
df.to_csv('cleaned_organizations.csv', index=False)

print("Data cleaning completed!")

# Read the cleaned data
df = pd.read_csv('cleaned_organizations.csv')

# Define a list of keywords related to 21st-century skills
skills_keywords = ['scientific','critical', 'collaboration', 'literacy', 'numeracy',
                  'creativity', 'problem solving', 'communication', 
                  'social skills', 'adaptability','curiosity','project-based',
                 'initiative','persistence','leadership','cultural','experiential',
                 'social','character qualities','employability','understanding','appreciating']

# Initialize a dictionary to record the match of each company with the keywords
matches = {key: [] for key in skills_keywords}

# Check for keywords in each company description
for keyword in skills_keywords:
    df[keyword] = df['Full Description'].apply(lambda x: 1 if keyword in x else 0)

# Now, each keyword related to 21st-century skills is added to df as a new column
# The column value is 1 if the description contains the keyword, otherwise 0

# Save the results
df.to_csv('matched_organizations.csv', index=False)

print("Matching completed!")




# Create a new column 'sum_keywords', whose value is the sum of all keyword columns
df['sum_keywords'] = df[skills_keywords].sum(axis=1)

# Check the number of companies with at least one keyword
has_keywords = len(df[df['sum_keywords'] > 0])

# Calculate the number of companies without any keywords
no_keywords = len(df) - has_keywords

# Draw a pie chart
labels = ['Has Keywords', 'No Keywords']
sizes = [has_keywords, no_keywords]
colors = ['green', 'orange']
explode = (0.1, 0)  # Slightly protrude the "Has Keywords" section

plt.figure(figsize=(10, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Percentage of Companies With vs. Without Keywords")
plt.show()



data = pd.read_csv('matched_organizations.csv')


skills_columns = ['scientific','critical', 'collaboration', 'literacy', 'numeracy',
                  'creativity', 'problem solving', 'communication', 
                  'social skills', 'adaptability','curiosity','project-based',
                 'initiative','persistence','leadership','cultural','experiential',
                 'social','character qualities','employability','understanding','appreciating']

# Calculate the total count for each keyword
skills_counts = data[skills_columns].sum().sort_values(ascending=False)

# Plot the bar chart
plt.figure(figsize=(12,8))
sns.barplot(x=skills_counts.values, y=skills_counts.index, palette="viridis")
plt.xlabel('Number of Companies Mentioned')
plt.ylabel('21st Century Skills')
plt.title('Distribution of 21st Century Skills in EdTech Companies')
plt.tight_layout()
plt.show()

# Calculate the correlation between skills
correlation_matrix = data[skills_columns].corr()

# Plot heatmap to showcase the correlation between skills
plt.figure(figsize=(14,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Between 21st Century Skills')
plt.tight_layout()
plt.show()


X = data[skills_columns]
y = data['CB Rank (Organization)'] 

# Train decision tree model
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Get feature importances
feature_importances = tree.feature_importances_

# Match feature importances with feature names
feature_importance_map = dict(zip(skills_columns, feature_importances))

# Sort features by importance
sorted_features = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)

# Select the top N most important features (here, I chose the top 10, but you can modify as needed)
top_features = [feature[0] for feature in sorted_features[:10]]

# Create a new dataset using these features
reduced_data = data[top_features]

# Now, the reduced_data DataFrame contains the selected most important features

numeric_data = data.select_dtypes(include=[np.number])

data_scaled = StandardScaler().fit_transform(numeric_data)

# Standardize data
data_scaled_reduced = StandardScaler().fit_transform(reduced_data)

# PCA
pca_reduced = PCA(n_components=2)  # Visualize using 2 principal components
principal_components_reduced = pca_reduced.fit_transform(data_scaled_reduced)

# Determine number of clusters using Dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(principal_components_reduced, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# Choose an appropriate number of clusters from the Dendrogram, e.g., n_clusters=3
n_clusters = 3
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
cluster_labels_reduced = agg_cluster.fit_predict(principal_components_reduced)

# Visualize clustering results
plt.figure(figsize=(8, 6))
plt.scatter(principal_components_reduced[:, 0], principal_components_reduced[:, 1], c=cluster_labels_reduced, cmap='rainbow')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Hierarchical Clustering with PCA Components')
plt.show()

