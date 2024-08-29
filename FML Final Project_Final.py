#%%
# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.outliers import Winsorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from clusteval import clusteval
from sklearn import metrics
#%%

# Importing & Loading the dataset
acidental_data = pd.read_csv("/Users/home/Desktop/Rdata/FML/Final Project/Road Accident Data.csv")
acidental_data1 = acidental_data.sample(n=35000, random_state=1)
#%%

# Checking for duplicate values
acidental_data1.duplicated().value_counts()

# Checking for nan / NA values 
acidental_data1.isna().sum()


# As there was a spelling mistake in dataset, we replaced the Value "Fetal" with Value 'Fatal" in Accidental Severity Column.
acidental_data1["Accident_Severity"] = acidental_data1.Accident_Severity.replace({"Fetal":"Fatal"})

# Dropping unwanted columns/Features
# As Accident_Index is used for reference purpose and Carriageway_Hazards has 97% of missing values. Also Accident_Index,Longitude and Latitude are the Variables which we cant work on for now. 
acidental_data1 = acidental_data1.drop(['Accident_Index', 'Carriageway_Hazards','Latitude','Longitude'], axis = 1)

# Creating the Boxplot to check for outliers in the dataset
num_col = acidental_data1.select_dtypes(exclude = ["object"])
num_columns = acidental_data1.select_dtypes(exclude = ["object"]).columns
#%%

# Box Plot Visualization
num_col.plot(kind = "box", sharey = False, subplots = True)
plt.show()
#%%

# Visualization (Bar plot)
acidental_data1.columns
acidental_data1.Accident_Severity.value_counts().plot(kind = "bar", color="red")
plt.xlabel("Severity of Accident")
plt.ylabel("Casualities")
plt.show()
#%%

# Multiple scatter plot explaining relationship between all variables
sns.pairplot(acidental_data1)

# Checking for Skewness
skew(acidental_data1.Number_of_Casualties)

# Visualizing the Skewness of Casualties column
sns.kdeplot(acidental_data1['Number_of_Casualties'], bw_method = 0.7 ,fill = True)
#%%



# Analysis the plot: 
# Bar plot : Almost 29000 accidents had slight Severity whereas serious were approximately 4000 and there were almost no fatal accidents
# Multiple scatter plot : Almost all the columns have no collinarity among each other as per the dataset.
# Analysis of Skewness: 
# In a positively skewed distribution:
# The tail on the right-hand side of the distribution is longer or fatter than the left-hand side.


#%%

# Data Preprocessing

# Dropping the Duplicate values
acidental_data1 = acidental_data1.drop_duplicates()

# Treating outliers using Winsorization method

for i in num_columns:
    # Create a Winsorizer object for the current column
    winsorize = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=i)
    winsorize.fit(acidental_data1)
    winsorized_values = winsorize.transform(acidental_data1)
    acidental_data1[i] = winsorized_values[i]
## Remmoved the outliers

# Feature Engineering

# Concatinating date and time column
acidental_data1['DateTime'] = pd.to_datetime(acidental_data1['Accident Date'] + ' ' + acidental_data1['Time'])
acidental_data1["month"] = acidental_data1['DateTime'].dt.month
acidental_data1["year"] = acidental_data1['DateTime'].dt.year


acidental_data1['Time_Category'] = pd.cut(acidental_data1['DateTime'].dt.hour,
                               bins=[0, 6, 12, 20, 24],
                               labels=['midnight', 'morning', 'evening', 'night'],
                               right=False,
                               include_lowest=True)
#%%

# Parameter selection for bin:
# We define the bins parameter to represent the boundaries of the time intervals:
# 0, 6: Midnight to 5:59 AM (midnight).
# 6, 12: 6:00 AM to 11:59 AM (morning).
# 12, 20: 12:00 PM to 7:59 PM (evening).
# 20, 24: 8:00 PM to 11:59 PM (night).


# As we have extracted all the meaning information from date time column such as month 
# such as which part of day event happened, month and year we can drop date time column as it can not provide us
# Information fruthermore

# Dropping date time column as all meaningful data was extracted from it
acidental_data1.drop(['DateTime'], axis = 1, inplace = True)
acidental_data1.drop(['Accident Date'], axis = 1, inplace = True)
acidental_data1.drop(['Time'], axis = 1, inplace = True)


# As the data type of 'Time_Category' is category we will convert it to 'object
acidental_data1['Time_Category'] = acidental_data1['Time_Category'].astype('object')


# Dropping the missing values
acidental_data1.dropna(inplace=True)
acidental_data1.isna().sum()


# Select categorical columns
categorical_columns = acidental_data1.select_dtypes(include=['object']).columns.tolist()
#%%

# Encoding the categorical columns
ordinal_encoder = OrdinalEncoder()
acidental_data_encoded = ordinal_encoder.fit_transform(acidental_data1[categorical_columns])
encoded_column_names = ordinal_encoder.get_feature_names_out(categorical_columns)
acidental_data_encoded_df = pd.DataFrame(acidental_data_encoded, columns=encoded_column_names)

# Reseting the index for column
acidental_data_encoded_df.reset_index(drop = True, inplace = True)


# Combining Categorical data and Numerical data to get all the columns of dataset
numerical_columns = pd.DataFrame(acidental_data1.select_dtypes(exclude = ['object']))
numerical_columns.reset_index(drop = True, inplace = True)

acidental_data_combined = pd.concat([acidental_data_encoded_df, numerical_columns], axis=1)
# acidental_data_combined = acidental_data_combined.drop(['index'], axis=1)
#%%

# Scaling Numerical Features
scale = StandardScaler()
acidental_data_scaled = pd.DataFrame(scale.fit_transform(acidental_data_combined),columns=acidental_data_combined.columns)

# Heatmap ( Checking for correlation)
correlations = acidental_data_scaled.corr()
sns.heatmap(correlations, annot = True, cmap = 'coolwarm')

# Outcome:
# The heighest colinearity is 0.6 which is moderate and both columns Weather_Conditions and Road_Surface_Conditions
# are important for our Analysis, so we will keep both the columns


#%%

# Applying PCA transformation to check for Correlation
from sklearn.decomposition import PCA

pca = PCA(n_components=18)
pca_fit = pca.fit(acidental_data_combined)

pca_res = pd.DataFrame(pca.transform(acidental_data_combined))
pca_res.columns = ['pc0','pc1','pc2','pc3','pc4','pc5','pc6',
                      'pc7','pc8','pc9','pc10','pc11','pc12',
                      'pc13','pc14','pc15','pc16','pc17']

var_ratio = pca_fit.explained_variance_ratio_


final = pd.DataFrame(pca_res.iloc[:,:2]).rename(columns = {0:"Component0",1:"Component1"})
# Observation:
# There is maximum change in 2 so we will consider first 2 columns of PCA dataset
# also the data first 2 columns are providing is 99.70456526.
# PCA (Principal Component Analysis) involves dimension reduction and captures the variance in the data


# Checking cummilative varience for all the columns
var = np.cumsum(pca_fit.explained_variance_ratio_)*100
var

# Plotting an elbow curve
plt.plot(var,color='red')
plt.xticks(range(0, 18, 1))



# Create the scree plot for PCA
plt.figure(figsize=(8, 6))  # Adjust figure size as desired
plt.plot(range(1, len(var_ratio) + 1), var_ratio, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot for PCA')
plt.grid(True)
plt.show()

# Performing Bi-Plot with all the variables in the Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(acidental_data_scaled)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

def biplot(score, coeff, labels=None):
    plt.figure(figsize=(8, 8))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    

# Reshape principal components for biplot
pc1 = principal_components[:, 0].reshape(-1, 1)
pc2 = principal_components[:, 1].reshape(-1, 1)
pcs = np.hstack((pc1, pc2))

# Call the biplot function
biplot(pcs, np.transpose(pca.components_), labels=acidental_data_scaled.columns)
plt.show()

print(acidental_data_scaled.columns)

# Removing variables from Dataset after seeing the above BI-PLOT. 'Day_of_Week', 'Junction_Control', 'Junction_Detail',
# 'Local_Authority_(District)',Police_Force', 'year'
acidental_data_scaled2 = acidental_data_scaled.drop(['Day_of_Week','Junction_Control','Local_Authority_(District)','Police_Force', 'Junction_Detail','year'], axis=1)
# Performing Bi-Plot again after removing the variables

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(acidental_data_scaled2)

# Perform PCA
pca1 = PCA(n_components=2)
principal_components1 = pca.fit_transform(acidental_data_scaled2)

def biplot(score, coeff, labels=None):
    plt.figure(figsize=(8, 8))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    

# Reshape principal components for biplot
pc1 = principal_components1[:, 0].reshape(-1, 1)
pc2 = principal_components1[:, 1].reshape(-1, 1)
pcs = np.hstack((pc1, pc2))

# Call the biplot function
biplot(pcs, np.transpose(pca.components_), labels=acidental_data_scaled2.columns)
plt.show()

# From 18 Variables in total,We have removed the unnecessary variables from the dataset by seeing the above final Bi_plot. However, we have decided to let few of the variables be the way they are, because we need to work on the Safety Measures to Decrease the count of accidents as per the domain knowledge
# we shall be working on are the 12 Variables and apply the algorithms like Hierarchial and K-means.

#Interpreting the Plot:
#
# Steep Initial Decline: The initial decline in the scree plot is typically sharp, indicating that the first few PCs capture a significant portion of the variance in the data.
# "Elbow" Point: Look for an "elbow" where the curve starts to flatten out. This point represents a drop in the explained variance ratio, suggesting that subsequent PCs capture less and less important information.
# Choosing the Number of PCs: The number of PCs to retain is often chosen based on the elbow point. PCs before the elbow explain a significant amount of variance and are likely more important to keep. Those after the elbow explain a smaller amount and might be less important, depending on your modeling goals.

#%%

# Applying Algorithms:
    # Hierarchial & K means

# Dendrogram
plt.figure(1,figsize=(15,9))
tree_plot = dendrogram(linkage(acidental_data_scaled2, method="ward"))
plt.title("Hierarchial Clustering for Accident Data")
plt.xlabel("Index")
plt.ylabel("Euclidean Distance")
plt.show()
#%%

# Building Hierarchial Model
Hierarchical_model = AgglomerativeClustering(n_clusters=2, linkage="ward")
Hierarchical_predict = Hierarchical_model.fit_predict(acidental_data_scaled2)
cluster_labels = pd.Series(Hierarchical_model.labels_)

# Checking the Silhouette Score
metrics.silhouette_score(acidental_data_scaled2,cluster_labels)
# Best silhouette_score for hierarchy is 0.2017915516157023
#%%

# Kmeans
from sklearn.cluster import KMeans

# Building KMeans clustering
kmeans_bestmodel = KMeans(n_clusters = 4)
kmeans_bestmodel.fit(acidental_data_scaled2)
kmeans_cluster_labels = pd.Series(kmeans_bestmodel.labels_)


# Checking for best n_clusters

clusters_score= {}
clust = list(range(2,9))
for i in clust:
    KMeans_model = KMeans(n_clusters = i)
    KMeans_model.fit(acidental_data_scaled2)
    KMeans_model.labels_
    # Checking silhouette score
    clusters_score[i] = (metrics.silhouette_score(acidental_data_scaled, KMeans_model.labels_)*100)


#%%

# Plotting the elbow graph
plt.figure(figsize=(10,6))
plt.plot(list(clusters_score.keys()), list(clusters_score.values()), marker='o', linestyle='-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Analysis :
# Analysing the elbow curve we get to know the best n_clusters = 4. 

#%%

# Checking the Silhouette Score
metrics.silhouette_score(acidental_data_scaled2,kmeans_cluster_labels)
# Best silhouette_score for KMeans is 0.18379667494172364.

# Analysis:
# Comparing the silhouette_score from both the model's we get to know that the best algorithm amoung 2 is hierarchy.


# As we can see from silhoutte score the best model is hierarchy
# We will add clusters formed from hierarchy for analysis 
acidental_data_scaled2["Clusters"] = pd.Series(cluster_labels) 

# 
acidental_data_final = acidental_data1
acidental_data_final.reset_index(drop =  True, inplace = True)

Labels = cluster_labels.reset_index(drop = True)

acidental_data_final["Clusters"] = pd.Series(Labels) 
acidental_data_final["Clusters"].value_counts()
acidental_data_final['Local_Authority_(District)'].value_counts()
#%%

# Group data

# User defined function
def mode_apply(x):
    return pd.Series.mode(x)

# Group by 'Clusters' and apply the mode calculation for each column
cluster_agg = acidental_data_final[['Time_Category', 'Local_Authority_(District)', 'Weather_Conditions','Vehicle_Type','Accident_Severity','Road_Surface_Conditions','Road_Type','Urban_or_Rural_Area','month']].groupby(acidental_data_scaled2.Clusters).agg(mode_apply)


# Final Analysis:

    # A consistent pattern is seen in both clusters: the majority of incidents happened in the evening, 
    # and Birmingham is the site where accidents happen most frequently, on Single Carriageway mostly involving autos.
    # There is, however, a noticeable difference in the weather and Road Surface Conditions between the clusters. 
    # Rain was shown to be a major contributing factor to accidents in cluster 1, 
    # but good weather was the primary cause of accidents in cluster 0.

    
#-------------------------------------------------------------------------------
