# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.outliers import Winsorizer
import joblib
import pickle

# Loading dataset
acidental_data1 = pd.read_csv("/Users/home/Desktop/Rdata/FML/Final Project/Road Accident Data.csv")
acidental_data1 

# Dropping unwanted columns/Features after checking corelation
acidental_data = acidental_data1.drop(['Time','Accident_Index', 'Carriageway_Hazards',
                                       'Longitude', 'Latitude','Local_Authority_(District)',
                                       'Police_Force'], axis = 1)

# Creating Boxplot to check outliers
num_col = acidental_data.select_dtypes(exclude = ["object"])
num_columns = acidental_data.select_dtypes(exclude = ["object"]).columns

num_col.plot(kind = "box", sharey = False, subplots = True)
plt.show()

# Checking for duplicate values
acidental_data.duplicated().value_counts()

# Checking for nan values
acidental_data.isna().sum()


# Dropping Duplicate values
acidental_data = acidental_data.drop_duplicates()

# Treating outliers using Winsorization method (treating outliers and getting them with in range)
for i in num_columns:
    # Create a Winsorizer object for the current column
    winsorize = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=i)
    winsorize.fit(acidental_data)
    winsorized_values = winsorize.transform(acidental_data)
    acidental_data[i] = winsorized_values[i]

# Boxplots to cross check winsorization
acidental_data[num_columns].plot(kind="box", sharey=False, subplots=True)
plt.show()


# Feature Engineering (Creating new Columns, the accident Date Column has been converted to Standard Date Time Format)
acidental_data['Accident Date'] = pd.to_datetime(acidental_data['Accident Date'])
acidental_data["month"] = acidental_data['Accident Date'].dt.month
acidental_data.drop(['Accident Date'], axis = 1, inplace = True)


# Data Preprocessing
numerical_features = acidental_data.select_dtypes(exclude=["object"]).columns
categorical_features = acidental_data.select_dtypes(include=["object"]).columns

# Imputing missing values for categorical data
imputer_numerical = SimpleImputer(strategy="mean")
imputed_values = imputer_numerical.fit_transform(acidental_data[numerical_features])
acidental_data_numerical_imputed = pd.DataFrame(imputed_values, columns=numerical_features)

# Impute missing values for categorical features with most frequent value and encode them
imputer_categorical = SimpleImputer(strategy="most_frequent")
ordinal_encoder = OrdinalEncoder()
acidental_data_categorical_imputed_encoded = ordinal_encoder.fit_transform(imputer_categorical.fit_transform(acidental_data[categorical_features]))

# Scaling numerical features (Normalizing the Data range between 0-1)
scale = MinMaxScaler()
acidental_data_numerical_scaled = pd.DataFrame(scale.fit_transform(acidental_data_numerical_imputed), 
                                               columns=numerical_features)

# Concatenate the transformed numerical and categorical features
acidental_data_cleaned_scaled = pd.concat([pd.DataFrame(acidental_data_categorical_imputed_encoded,
                                                        columns=categorical_features), 
                                           acidental_data_numerical_scaled], axis=1)


# Applying PCA transformation (Principle Component Analysis)
from sklearn.decomposition import PCA

pca = PCA(n_components=14)
pca_fit = pca.fit(acidental_data_cleaned_scaled)

pca_res = pd.DataFrame(pca.transform(acidental_data_cleaned_scaled))
pca_res.columns = ['pc0','pc1','pc2','pc3','pc4','pc5','pc6',
                      'pc7','pc8','pc9','pc10','pc11','pc12',
                      'pc13']


var_ratio = pca_fit.explained_variance_ratio_

# Checking cummilative varience for all the columns
var = np.cumsum(pca_fit.explained_variance_ratio_)*100
var

# Plotting an elbow curve
plt.plot(var,color='red')

# Plotting a KneeLocator
from kneed import KneeLocator
k1 = KneeLocator(range(len(var)), var, direction="increasing",curve="concave")
k1.elbow

plt.plot(range(len(var)), var)
plt.xlabel("Range")
plt.xticks(range(len(var)))
plt.ylabel("Cum_sum")
plt.axvline(k1.elbow,color="r",label="axvline",ls='--')
plt.show()
final = pd.DataFrame(pca_res.iloc[:,:3]).rename(columns = {0:"Component0",
                                                           1:"Component1",
                                                           2:"Component2"})
final
from scipy.sparse import csr_matrix

# Convert final DataFrame to a sparse matrix (compress the data)
final_sparse = csr_matrix(final.values)


# Building KMeans Model as the data is too large we will use MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans

# Building model
n_clusters = 3  # Number of clusters
kmeans_model = MiniBatchKMeans(n_clusters=n_clusters)
cluster_predict = kmeans_model.fit_predict(final_sparse)
cluster_labels = pd.Series(kmeans_model.labels_)


from sklearn.cluster import Birch

# Creating BIRCH model (Using Extention for Hierarchy as iit is Big data)
birch = Birch(branching_factor=50, n_clusters=3, threshold=0.5, compute_labels=True)

# Fit BIRCH to the data
birch.fit(final_sparse)
cluster_labels = birch.labels_
acidental_data_cleaned_scaled['Cluster_Labels'] = cluster_labels

# View the distribution of cluster labels
acidental_data_cleaned_scaled['Cluster_Labels'].value_counts()

# Checking the accuracy
from sklearn.metrics import silhouette_score
# Calculate silhouette score
silhouette_avg = silhouette_score(final_sparse, cluster_labels)
print("Silhouette Score:", silhouette_avg)
# silhouette_score = 41.97 for Birch



