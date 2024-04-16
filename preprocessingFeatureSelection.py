import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


# Load the dataset
df = pd.read_csv('Breast_Cancer_dataset.csv')

#Impute missing vals, don't OneHotEncoding and normalize yet since we need to detect outliers
numericFeatures = df.select_dtypes(include=['int64', 'float64']).columns
categoricalFeatures = df.select_dtypes(include=['object']).columns
numericTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
])
categoricalTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericTransformer, numericFeatures),
        ('cat', categoricalTransformer, categoricalFeatures)
    ])

dfProcessed = preprocessor.fit_transform(df)
columnNames = numericFeatures.tolist() + preprocessor.named_transformers_['cat']['imputer'].get_feature_names_out(categoricalFeatures).tolist()
dfProcessed = pd.DataFrame(dfProcessed, columns=columnNames)

#Remove outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers = lof.fit_predict(dfProcessed[numericFeatures])  # Applying LOF on numeric features only
dfProcessed['LOF'] = outliers

numOutliers = sum(outliers == -1)
dfProcessed = dfProcessed[dfProcessed['LOF'] == 1]
dfProcessed.drop(columns=['LOF'], inplace=True)
print(f"Number of outliers detected and removed: {numOutliers}")
print(f"Number of rows after removing outliers: {dfProcessed.shape[0]}")

#Normalize with min max numerical data
numericTransformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericTransformer, numericFeatures),
        ('cat', 'passthrough', categoricalFeatures)
    ])

dfProcessed = preprocessor.fit_transform(dfProcessed)
featureNames = [f"num__{name}" for name in numericFeatures] + [f"cat__{name}" for name in categoricalFeatures]
dfProcessed = pd.DataFrame(dfProcessed, columns=featureNames)

#Feature Selection with Variance value
variances = dfProcessed[[name for name in featureNames if name.startswith('num__')]].var()
print(variances)

threshold = 0.01
lowVarianceFeatures = [column for column in variances.index if variances[column] <= threshold]

dfProcessed = dfProcessed.drop(columns=lowVarianceFeatures)

print(f"Removed {len(lowVarianceFeatures)} low-variance features:", lowVarianceFeatures)
print(f"DataFrame shape after removing low-variance features: {dfProcessed.shape}")

print(dfProcessed.tail())

#One Hot Encoding
categoricalTransformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop="first")),
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', [col for col in dfProcessed.columns if col.startswith('num__')]),
        ('cat', categoricalTransformer, [col for col in dfProcessed.columns if col.startswith('cat__')])
    ])

dfProcessed = preprocessor.fit_transform(dfProcessed)
featureNames = preprocessor.get_feature_names_out()
dfProcessed = pd.DataFrame(dfProcessed, columns=featureNames)
print(dfProcessed)


dfProcessed.columns = [col.split('__')[-1] for col in dfProcessed.columns]
status_dead = dfProcessed['Status_Dead']
features_for_pca = [col for col in dfProcessed.columns if col != 'Status_Dead']

#Perform PCA
pca = PCA(n_components=0.9)
principal_components = pca.fit_transform(dfProcessed[features_for_pca])
pcaDf = pd.DataFrame(data=principal_components,
                      columns=[f'Principal Component {i+1}' for i in range(principal_components.shape[1])])
pcaDf['Status_Dead'] = status_dead.reset_index(drop=True)  # Reset index to ensure proper alignment

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.sum(pca.explained_variance_ratio_))
print(pcaDf)

#Save data
#pcaDf.to_csv('pcaDF.csv', index=False)
