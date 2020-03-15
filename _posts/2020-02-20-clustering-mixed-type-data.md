---
layout: post
title:  "How to cluster mixed datatype datasets"
date:   2020-02-20 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [clustering,data-mining,dimension-reduction,decomposition]
---
![png]({{ "/assets/images/2020-02-20-clustering-fig1.png" }})

***Introduction***

Cluster analysis is the task of grouping objects within a population in such a way that objects in the same group or cluster are more similar to one another than to those in other clusters. Clustering is a form of unsupervised learning as the number, size and distribution of clusters is unknown a priori.
Clustering can be applied to a variety of different problems and domains including: customer segmentation for retail sales and marketing, identifying higher or lower risk groups within [insurance portfolios](https://www.casact.org/pubs/dpp/dpp08/08dpp170.pdf), to finding [storm systems on Jupyter](https://astronomycommunity.nature.com/users/253561-ingo-waldmann/posts/48323-deep-learning-saturn), and even [galaxies far far away](https://arxiv.org/abs/1404.3097).

Many real world datasets include combinations of numerical, ordinal (e.g. small, medium, large), and nomial (e.g. France, China, India) data features. However, many popular clustering algorithms and tutorials such as K-means are suitable for numerical data only. Sklearn provides an excellent review of these methods [here](https://scikit-learn.org/stable/modules/clustering.html#clustering). 

This aim of this article is provide a review and practical application of methods for clustering datasets with mixed datatypes. You can find all of my code on [Github here](https://github.com/bpostance/training.data_science/blob/master/ML/2.3_Clustering/10-Clustering-Mixed-Data.ipynb)

***Aim:***
*To define a strategy and method to cluster large datasets containing a variety of dataype's*

***Objectives:***
1. To research and review clustering techniques for mixed datatype datasets. 
1. To research and review feature encoding and engineering strategies. 
1. To apply and review clustering methods on a test dataset.


```python
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
```

### 2. Case Study: 
California auto-insurance claims [dataset](https://www.kaggle.com/xiaomengsun/car-insurance-claim-data)

[***Data dictionary***](https://rpubs.com/data_feelings/msda_data621_hw4), includes:
- Bluebook = car re-sale value. 
- MVR_PTS = [MotorVehicleRecordPoints (MVR) ](https://www.wnins.com/losscontrolbulletins/MVREvaluation.pdf) details an individual’s past driving history indicating violations and accidents over a specified period
- TIF = Time In Force / customer lifetime
- YOJ = years in job
- CLM_FRQ = # of claims in past 5 years
- OLDCLAIM = sum $ of claims in past 5 years

https://community.alteryx.com/t5/Alteryx-Designer-Discussions/Insurance-Datasets/td-p/440035
https://rpubs.com/data_feelings/msda_data621_hw4
https://rdrr.io/cran/HDtweedie/man/auto.html
https://cran.r-project.org/web/packages/insuranceData/insuranceData.pdf



```python
# load data
DATA_PATH = os.path.join(os.getcwd(),'../_data')
df = pd.read_csv(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim.csv'),low_memory=False,)

# convert object to numerical
df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM', 'CLM_AMT',]] = df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM', 'CLM_AMT',]].replace('[^.0-9]', '', regex=True,).astype(float).fillna(0)

# clean textual classes
for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = df[col].str.upper().replace('Z_','',regex=True).replace('[^A-Z]','',regex=True)
        
data_types = {f:t for f,t in zip(df.columns,df.dtypes)}

df[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>KIDSDRIV</th>
      <th>BIRTH</th>
      <th>AGE</th>
      <th>HOMEKIDS</th>
      <th>YOJ</th>
      <th>INCOME</th>
      <th>PARENT1</th>
      <th>HOME_VAL</th>
      <th>MSTATUS</th>
      <th>...</th>
      <th>CAR_TYPE</th>
      <th>RED_CAR</th>
      <th>OLDCLAIM</th>
      <th>CLM_FREQ</th>
      <th>REVOKED</th>
      <th>MVR_PTS</th>
      <th>CLM_AMT</th>
      <th>CAR_AGE</th>
      <th>CLAIM_FLAG</th>
      <th>URBANICITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63581743</td>
      <td>0</td>
      <td>MAR</td>
      <td>60.0</td>
      <td>0</td>
      <td>11.0</td>
      <td>67349.0</td>
      <td>NO</td>
      <td>0.0</td>
      <td>NO</td>
      <td>...</td>
      <td>MINIVAN</td>
      <td>YES</td>
      <td>4461.0</td>
      <td>2</td>
      <td>NO</td>
      <td>3</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0</td>
      <td>HIGHLYURBANURBAN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132761049</td>
      <td>0</td>
      <td>JAN</td>
      <td>43.0</td>
      <td>0</td>
      <td>11.0</td>
      <td>91449.0</td>
      <td>NO</td>
      <td>257252.0</td>
      <td>NO</td>
      <td>...</td>
      <td>MINIVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>HIGHLYURBANURBAN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



### 3 Method

#### 3.1 Data Preparation
 - remove or fill null values


```python
# copy df
tdf = df.copy()

# drop ID and Birth
tdf.drop(labels=['ID','BIRTH'],axis=1,inplace=True)
```


```python
# remove all nan values
tdf['OCCUPATION'].fillna('OTHER',inplace=True)
for col in ['AGE','YOJ','CAR_AGE']:
    tdf[col].fillna(tdf[col].mean(),inplace=True)
if tdf.isnull().sum().sum() == 0: print('No NaNs')
```

    No NaNs
    

***inspect and clean***


```python
data_meta = pd.DataFrame(tdf.nunique(),columns=['num'],index=None).sort_values('num').reset_index()
data_meta.columns = ['name','num']
data_meta['type'] = 'numerical'

# exclude known numericals
data_meta.loc[(data_meta['num']<=15) & (~data_meta['name'].isin(['MVR_PTS','CLM_FREQ','CLAIM_FLAG'])),'type']='categorical'
data_meta.loc[data_meta['name'].isin(['CLM_FREQ','CLAIM_FLAG']),'type']='claim'

cat_features = list(data_meta.loc[data_meta['type']=='categorical','name'])
num_features = list(data_meta.loc[data_meta['type']=='numerical','name'])
```


```python
# shorten names
tdf['URBANICITY'] = tdf['URBANICITY'].map({'HIGHLYURBANURBAN':'URBAN','HIGHLYRURALRURAL':'RURAL'})
tdf['EDUCATION'] = tdf['EDUCATION'].map({'HIGHSCHOOL':'HSCL', 'BACHELORS':'BSC', 'MASTERS':'MSC','PHD':'PHD'})
tdf['CAR_TYPE'] = tdf['CAR_TYPE'].map({'MINIVAN':'MVAN', 'VAN':'VAN', 'SUV':'SUV', 'SPORTSCAR':'SPRT', 'PANELTRUCK':'PTRK', 'PICKUP':'PKUP'})
```


```python
# Mosaic Plots
# https://rpubs.com/data_feelings/msda_data621_hw4
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.graphics.mosaicplot import mostaticmethodc

# props = {}
# for car in tdf['CAR_TYPE'].unique():
#     for i,color in zip([0,1],['grey','blue']):
#         props[(str(i),car)] = {'color':color}
# props

# m = mosaic(tdf, ['CAR_TYPE','CLAIM_FLAG',], title='DataFrame as Source',properties=props)
# plt.show()
```

***categorical feature counts***


```python
fig,axs = plt.subplots(3,4,figsize=(12,9),sharey=True)

for ax,feat in zip(axs.flatten(),cat_features):
    ax.hist(tdf[feat],align='left')
    ax.set_title(feat)
plt.tight_layout()
```


![png]({{ "/assets/images/2020-02-20-clustering-fig2.png" }})


***How are claims distributed amongst categoricals?***


```python
fig,axs = plt.subplots(3,4,figsize=(12,10),sharey=True)

for ax,feat in zip(axs.flatten(),cat_features):
    ((pd.crosstab(tdf['CLAIM_FLAG'],tdf[feat])) / (pd.crosstab(tdf['CLAIM_FLAG'],tdf[feat]).sum())).T.plot.bar(stacked=True,ax=ax,legend=False,title=None)
    ax.set_title(feat)
plt.tight_layout()
```


![png]({{ "/assets/images/2020-02-20-clustering-fig3.png" }})


#### 3.2 Feature Engingeering and Encoding
The data data features should be standardized in order to avoid dependence on the [datatypes](https://towardsdatascience.com/data-types-in-statistics-347e152e8bee) and on the variety or choice of measurement units.

Rule of thumb, when using any algorithm that computes distance or assumes normality, scale your features! [see here](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e), [here](https://stats.stackexchange.com/a/7182/100439), and [here](https://stats.stackexchange.com/questions/385775/normalizing-vs-scaling-before-pca).


- Numerical values:
 - Using ratio
 - normalisation (transform values to be on scale of 0-1)
 - standardisation (how many standard deviations the value is from the sample mean)


- Categorical: nomial or binary symmetric values, where outcomes are of equal importance (e.g. Male or Female)
 - One hot encoding
 - If high cardinality >15, try to reduce dimensionality by feature engineering or apply binary or hash encoding ([see here](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02))
 
 
- Categorical: ordinal or binary asymmetric values, where outcomes are not of equal importance (e.g. Gold, Silver, Bronze)
 - it depends!
 - Label encoding with 0-1 normalisation if values are on equal-importance increasing scale (Bronze=1,Silver=2,Gold=3, where 1=1*1, 2=2*1, 3=3*1, Gold-Bronze = 3-1 = 2 places worse).
 - Rank values with 0-1 normalisation, again if values are on equal-importance increasing scale.
 - If there is some target variable in the dataset (e.g. event occurence, medical diagnosis, iris type) one can also assign frequencies, odd ratios or weights-of-evidence to each ordinal class.

By far ordinal data is the most challenging to handle. There are many arguments between mathmatical purists, statisticians and other data practitioners on wether to treat ordinal data as qualitatively or quantitatively ([see here](https://creativemaths.net/blog/ordinal/)). In this authors opinion, ordianl data should be treated with caution and to adopt rational feature engineering and encdoing strategies based on understanding of the data and its domain.  

*It is essential to understand, for all machine learning models, all these encodings do not work well in all situations or for every dataset. Data Scientists still need to experiment and find out which works best for their specific case. If test data has different classes, then some of these methods won’t work as features won’t be similar. There are few benchmark publications by research communities, but it’s not conclusive, which works best. My recommendation will be to try each of these with the smaller datasets and then decide where to put more focus on tuning the encoding process. You can use the below cheat-sheet as a guiding tool.*


*references:*
 - https://miro.medium.com/max/2924/1*dvvxoZTdewLFs3RyZTJreA.png
 - [Datatypes in statistics](https://towardsdatascience.com/data-types-in-statistics-347e152e8bee)
 - [Binary symmetric and assymetric variables](https://www.quora.com/What-are-binary-symmetric-and-asymmetric-attributes)
 - [datatype conversions in clustering](https://paginas.fe.up.pt/~ec/files_0506/slides/05_Clustering.pdf)
 - [categorical feature engineering](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)
 - [Normalization vs Standardization — Quantitative analysis](https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf)
 - [Normalization vs Standardization](https://stats.stackexchange.com/a/10291/100439)
 
***numerical features***


```python
from sklearn.preprocessing import scale,RobustScaler,StandardScaler, MinMaxScaler
```


```python
# Scale standardisation of numerical values
numerical_features = pd.DataFrame(StandardScaler().fit_transform(tdf[num_features]),index=tdf.index,columns=num_features)
numerical_features = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(tdf[num_features]),index=tdf.index,columns=num_features)
```

***Categorical:Nomial values with binary symmetry***

Remember, here we are taking categorical values that are symmetric in scale only. Without getting into a debate, we could consider EDUCATION and OCCUPATIOM as either nomial (i.e. no order) or ordinal (i.e. hierachal). Here i'm going to take education as ordinal and occupation as nomial.

Observing our plots above we may also want to combine some classes where there are low frequencies or high cardinality. These are:

 - KIDSDRIV: collapse >= 2 to single category
 - HOMEKIDS: collapse >= 4 to single category

We then apply one-hot-encoding.


```python
cat_features.remove('EDUCATION')
tdf['KIDSDRIV'] = tdf['KIDSDRIV'].map({0:'0',1:'1',2:'2+',3:'2+',4:'2+'})
tdf['HOMEKIDS']= tdf['HOMEKIDS'].map({0:'0',1:'1',2:'2',3:'3',4:'4+',5:'4+'})
```


```python
nomial_features = pd.get_dummies(tdf[cat_features])
```

***Categorical:Ordinal values with binary asymmetry***


```python
tdf['EDUCATION'] = tdf['EDUCATION'].map({'HSCL':0, 'BSC':1, 'MSC':2,'PHD':3})
```


```python
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler(feature_range=(0,1))
```


```python
ordinal_features = pd.DataFrame(mx.fit_transform(tdf[['EDUCATION']]),index=tdf.index,columns=['EDUCATION'])
```

***Create datasets for clustering***

<span style="color:red">
Beware! check what transformations each package applies or can handle. For example, some may require features to be prepared as above a priori whilst others may handles this for you.
</span>

Rule of thumb, when using any algorithm that computes distance or assumes normality, scale your features! [see here](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e) and [here](https://stats.stackexchange.com/questions/385775/normalizing-vs-scaling-before-pca).

I will create two copies of the data:
 1. using the above OHE transformations and feature scaling (0,1).
 2. applying the above transformations but without feature scaling.


```python
# transformed and scaled dataset
Xy_scaled = pd.concat([numerical_features,nomial_features,ordinal_features],axis=1)
print(f'Data min:max {Xy_scaled.min().min(),Xy_scaled.max().max()}')
```

    Data min:max (0.0, 1.0)
    


```python
# original data
Xy_original = tdf.drop(labels=['CLAIM_FLAG'],axis=1)
```

### 3.3 Similarity Measures
<span style="color:red">
Again beware! check what transformations each package handles. For example, some may require features to be prepared as above a priori whilst others may handles this for you.
</span>


Common distance metrics include: 
- see scipy reference [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist) and [here](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.pdist.html)
- Euclidean distances are root sum-of-squares of differences. 
- Cityblock or Manhattan distances are the sum of absolute differences.
- Minkowski, a generalization of both the Euclidean distance and the Manhattan distance.
- Gower's distance, also Gower's coefficient (1971), is expressed as a dissimilarity and requires that a particular standardisation will be applied to each variable. The “distance” between two units is the sum of all the variable-specific distances. 
 - [$Gower$ $distance$](https://www.jstor.org/stable/2528823?seq=1) was proposed to measure dissimilarity between subjects with mixed types of variables using the mathematical concept of distance.
 - [R docs](https://rdrr.io/cran/gower/api)
 
By transforming and scaling our features we can in theory apply either metric. However, the formula for Gower handlse mixed datatypes. See the [Python Gower](https://www.thinkdatascience.com/post/2019-12-16-introducing-python-package-gower/) package.


```python
Image(url= "https://upload.wikimedia.org/wikipedia/commons/0/08/Manhattan_distance.svg")
```
![png]({{ "/assets/images/2020-02-20-clustering-fig3.1.png" }})



```python
import gower
```

You can use Gower to find similarity between a single entity and a list of candidates. 

This seems to pick out Urban, 40-50 year old males, who drive red minivan's, no kids, and with home values of around $160 K.


```python
%time sd = gower.gower_topn(Xy_original.iloc[4:5,:], Xy_original.iloc[:,:], n = 10)
print(Xy_original.iloc[sd['index']].describe().loc[['mean']])
Xy_original.iloc[sd['index']]
```

    Wall time: 120 ms
           AGE        YOJ   INCOME  HOME_VAL  EDUCATION  TRAVTIME  BLUEBOOK  TIF  \
    mean  49.0  13.847406  25948.5  161218.9        0.0      28.4   11585.0  6.0   
    
          OLDCLAIM  CLM_FREQ  MVR_PTS  CLM_AMT  CAR_AGE  
    mean       0.0       0.0      0.5    583.3      4.6  
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KIDSDRIV</th>
      <th>AGE</th>
      <th>HOMEKIDS</th>
      <th>YOJ</th>
      <th>INCOME</th>
      <th>PARENT1</th>
      <th>HOME_VAL</th>
      <th>MSTATUS</th>
      <th>GENDER</th>
      <th>EDUCATION</th>
      <th>...</th>
      <th>TIF</th>
      <th>CAR_TYPE</th>
      <th>RED_CAR</th>
      <th>OLDCLAIM</th>
      <th>CLM_FREQ</th>
      <th>REVOKED</th>
      <th>MVR_PTS</th>
      <th>CLM_AMT</th>
      <th>CAR_AGE</th>
      <th>URBANICITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>51.0</td>
      <td>0</td>
      <td>14.000000</td>
      <td>0.0</td>
      <td>NO</td>
      <td>306251.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>7673</th>
      <td>0</td>
      <td>46.0</td>
      <td>0</td>
      <td>13.000000</td>
      <td>33953.0</td>
      <td>NO</td>
      <td>164542.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>2535.0</td>
      <td>6.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>8699</th>
      <td>0</td>
      <td>41.0</td>
      <td>0</td>
      <td>15.000000</td>
      <td>38601.0</td>
      <td>NO</td>
      <td>151038.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>760</th>
      <td>0</td>
      <td>44.0</td>
      <td>0</td>
      <td>14.000000</td>
      <td>25588.0</td>
      <td>NO</td>
      <td>119825.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>2</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>9699</th>
      <td>0</td>
      <td>52.0</td>
      <td>0</td>
      <td>16.000000</td>
      <td>24575.0</td>
      <td>NO</td>
      <td>118811.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>7119</th>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>14.000000</td>
      <td>68958.0</td>
      <td>NO</td>
      <td>190128.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>6645</th>
      <td>0</td>
      <td>62.0</td>
      <td>0</td>
      <td>13.000000</td>
      <td>0.0</td>
      <td>NO</td>
      <td>157022.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>1</td>
      <td>3298.0</td>
      <td>1.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>7542</th>
      <td>0</td>
      <td>54.0</td>
      <td>0</td>
      <td>10.474062</td>
      <td>37424.0</td>
      <td>NO</td>
      <td>155505.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>0</td>
      <td>46.0</td>
      <td>0</td>
      <td>13.000000</td>
      <td>5015.0</td>
      <td>NO</td>
      <td>105811.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>URBAN</td>
    </tr>
    <tr>
      <th>10171</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>16.000000</td>
      <td>25371.0</td>
      <td>NO</td>
      <td>143256.0</td>
      <td>YES</td>
      <td>M</td>
      <td>0</td>
      <td>...</td>
      <td>11</td>
      <td>MVAN</td>
      <td>YES</td>
      <td>0.0</td>
      <td>0</td>
      <td>NO</td>
      <td>1</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>URBAN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24 columns</p>
</div>



And to calculate a matrix of similaritities between all entities


```python
# create list of cat_feature indicator
# [(x,True) if x in cat_features else (x,False) for x in Xy_original.columns]
cat_ind = [True if x in cat_features else False for x in Xy_original.columns]

try: 
    gd = np.load(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_gower_distance.npy'))
    print('Gower distances loaded from file.')
except:
    print('Calculating Gower dsitances...1-5 minutes')
    %time gd = gower.gower_matrix(Xy_original, cat_features=cat_ind)
    np.save(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_gower_distance.npy'),gd)
```

    Gower distances loaded from file.
    


```python
pd.DataFrame(gd[:5,:5])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.223359</td>
      <td>0.180066</td>
      <td>0.378062</td>
      <td>0.217091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.223359</td>
      <td>0.000000</td>
      <td>0.182596</td>
      <td>0.397125</td>
      <td>0.127068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.180066</td>
      <td>0.182596</td>
      <td>0.000000</td>
      <td>0.354758</td>
      <td>0.194679</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.378062</td>
      <td>0.397125</td>
      <td>0.354758</td>
      <td>0.000000</td>
      <td>0.316547</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.217091</td>
      <td>0.127068</td>
      <td>0.194679</td>
      <td>0.316547</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 4 Clustering methods for mixed datatypes

1. Hierachal (Gower distance matrix from original features):
 - [see here](https://stackoverflow.com/a/55306715/4538066) for discusion of suitable distance methods.
1. K-medoids (transformed and scaled features):
 - [distance metrics for k-medoids](https://stats.stackexchange.com/a/94178/100439)
 - [k-mediods in pyclustering package](https://pypi.org/project/pyclustering/)
    - [ISSUE](https://github.com/annoviko/pyclustering/issues/503) pyclustering package does not implement PAM as suggested on other sites.
 - K-medoids is poor performing on large datasets.
 - [k-mediods python implmentation in scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/latest/install.html)
 - [C++ build tools may be required on windows](https://www.scivision.dev/python-windows-visual-c-14-required/)
1. CLARANS (transformed and scaled features)
 - [Raymond, T., et al. 2002. CLARANS](http://www.cs.ecu.edu/dingq/CSCI6905/readings/CLARANS.pdf)
 - [clarans in python](https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4)
1. PAM partition-around-medoids (transformed and scaled features)
 - [PAM  is a variation of K-medoids](https://stats.stackexchange.com/a/141208/100439)
 - [Self defined PAM k-medoids in python](https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05)
1. K-means (transformed and scaled features)


```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
```

## 4.1 Hierachal (Gower distance matrix from original features)

[scipy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage) is used to generate:
 - $Z$, an ($n-1$) by 4 matrix . 
 - At the -th iteration, clusters with indices $Z[i, 0]$ and $Z[i, 1]$ are combined to form cluster.
 - A cluster with an index less than corresponds to one of the original observations. 
 - The distance between clusters $Z[i, 0]$ and $Z[i, 1]$ is given by $Z[i, 2]$. 
 - The fourth value $Z[i, 3]$ represents the number of original observations in the newly formed cluster.


```python
# output matrix has format [idx1, idx2, dist, sample_count]
try: 
    Z = np.load(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_linkage-complete.npy'))
    print('Z linkages loaded from file.')
except:
    print('Calculating Gower dsitances...1-5 minutes')
    %time Z = linkage(gd,method='complete')
    np.save(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_linkage-complete.npy'),Z)
    
Z_df = pd.DataFrame(Z,columns=['id1','id2','dist','n'])
```

    Z linkages loaded from file.
    

Visualise using a [scipy.dendogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram)

[*note*](https://stackoverflow.com/questions/9838861/scipy-linkage-format) it is not practical to infer the cluster or each observation using linkage and dendogram ouput.


```python
fig,axs = plt.subplots(1,1,figsize=(25,5))
dn = dendrogram(Z, truncate_mode='level',p=6,show_leaf_counts=True,ax=axs);
print(f"Leaves = {len(dn['leaves'])}")
```

    Leaves = 121
    


![png]({{ "/assets/images/2020-02-20-clustering-fig4.png" }})


Now to find the optimal number of clusters we apply:
1. [fcluster](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster) to form flat clusters from the hierarchical clustering defined by the linkage matrix ($Zd$).
1. [Silhouette scoring](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) to determine an appropriate number of clusters ($k$) or level in the dendogram. The Silhouette Coefficient ($S$) is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample where:
$$S = (b - a) / max(a, b)$$
 - ($S_1$) is computed using the "precomputed" Gower distances.
 - ($S_2$) is computed using predefined distance measures from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#) or [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html), in this instance "correlation". BUT, this only works on our transformed and scaled features created earlier.
 


```python
# find k clusters
results = dict()

k_cand = [2,3,4,5,7,9,]
k_cand.extend(list(np.arange(10,55,5)))

for k in k_cand:
    cluster_array = fcluster(Z, k, criterion='maxclust')
    score1 = silhouette_score(gd, cluster_array, metric='precomputed')
    score2 = silhouette_score(Xy_scaled, cluster_array,metric='correlation')
    results[k] = {'k':cluster_array,'s1':score1,'s2':score2}
    
plt.plot([i for i in results.keys()],[i['s1'] for i in results.values()],label='gower')
plt.plot([i for i in results.keys()],[i['s2'] for i in results.values()],label='correlation')
plt.legend()
plt.xticks(k_cand);
```


![png]({{ "/assets/images/2020-02-20-clustering-fig5.png" }})


Visualising the silhouette scores indicates that there are increases "spikes" in scores at 7 and 40 clusters.
Lets investigate these two sets.


```python
# assign 7 and 40
tdf['k-medoids-7'] =results[7]['k']
tdf['k-medoids-40'] =results[40]['k']
```


```python
fig,axs = plt.subplots(6,2,figsize=(10,15),sharex=True)

for ax,feat in zip(axs.flatten(),num_features):
    pd.plotting.boxplot(tdf,column=[feat],by='k-medoids-7',ax=ax)
    ax.set_xlabel('')  
plt.tight_layout()
```


![png]({{ "/assets/images/2020-02-20-clustering-fig6.png" }})



```python
fig,axs = plt.subplots(11,1,figsize=(10,40),sharex=True)

for ax,feat in zip(axs.flatten(),num_features):
    pd.plotting.boxplot(tdf,column=[feat],by='k-medoids-40',ax=ax)
    ax.set_xlabel('')

plt.tight_layout()
```


![png]({{ "/assets/images/2020-02-20-clustering-fig7.png" }})



```python
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
```

K-medoids can be caluclated using many distance metrics. Here the *Minkowski distance* or "cityblock" is used as this provides a suitable measure where there are both categorical and numerical features [see here](https://www2.cs.duke.edu/courses/spring18/compsci216/lectures/07-clustering.pdf).




```python
try: 
    Xy_scaled_minkowski = np.load(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_Xy_scaled_minkowski.npy'))
    print('Minkowski distances loaded from file.')
except:
    print('Calculating Minkowski dsitances...1-5 minutes')
    %time Xy_scaled_minkowski = squareform(pdist(Xy_scaled, 'minkowski'))
    np.save(os.path.join(DATA_PATH,'car-insurance-claim-data/car_insurance_claim_Xy_scaled_minkowski.npy'),Xy_scaled_minkowski)
```

    Minkowski distances loaded from file.
    

I'm going to use the pyclustering implmentation in order to have comparison's to the similar, yet more memory efficient, PAM and CLARANS medoid methods.

<span style="color:red">
Beware! pyclustering return clusters in an $n$ length list of lists, where $n=k$ and $list$[$n$][$i$] is the index postion from the input distance matrix. Here i use a dataframe to convert the pyclustering output to the form expected by scikit-learn silhouette score.
    
- [see this issue](https://github.com/annoviko/pyclustering/issues/593)  
</span>


```python
# find k clusters
results_kmedoids = dict()

k_cand = [3,7,15,30,45,60]
#k_cand.extend(list(np.arange(10,55,5)))

for k in k_cand:
    # initiate k random medoids - sets k clusters
    initial_medoids = np.random.randint(0,1000,size=k)
    kmedoids_instance = kmedoids(Xy_scaled_minkowski,initial_medoids, data_type='distance_matrix')    

    # run cluster analysis and obtain results
    %time kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()

    # convert cluster output
    cluster_array = pd.DataFrame([(x,e) for e,i in enumerate(clusters) for x in i if len(i)>1]).sort_values(by=0)[1].values
    
    # score
    score1 = silhouette_score(Xy_scaled_minkowski, cluster_array, metric='precomputed')
    score2 = silhouette_score(Xy_scaled, cluster_array,metric='correlation')
    
    # store
    results_kmedoids[k] = {'k':cluster_array,'s1':score1,'s2':score2}
    
plt.plot([i for i in results_kmedoids.keys()],[i['s1'] for i in results_kmedoids.values()],label='Minkowski')
plt.plot([i for i in results_kmedoids.keys()],[i['s2'] for i in results_kmedoids.values()],label='correlation')
plt.legend()
plt.xticks(k_cand);
```

    Wall time: 1min 14s
    Wall time: 55.9 s
    Wall time: 52 s
    Wall time: 55.6 s
    Wall time: 51.6 s
    Wall time: 49.7 s
    

![png]({{ "/assets/images/2020-02-20-clustering-fig8.png" }})



## 4.3 CLARANS (transformed and scaled features)
### *Clustering Large Applications based on RANdomized Search*
 - [clarans in python](https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4)


```python
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import euclidean_distance_square
```


```python
Xy_scaled_list = Xy_scaled.to_numpy().tolist()
len(Xy_scaled_list)

# sample euclid
# [euclidean_distance_square(Xy_scaled_list[0],Xy_scaled_list[i]) for i in range(4)]

"""
data: Input data that is presented as list of points (objects), each point should be represented by list or tuple.
number_clusters: amount of clusters that should be allocated.
numlocal: the number of local minima obtained (amount of iterations for solving the problem).
maxneighbor: the maximum number of neighbors examined.     
The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.
"""

clarans_instance = clarans(data=Xy_scaled_list[:50], number_clusters=3, numlocal=1, maxneighbor=2)
%time clarans_instance.process()
clusters = clarans_instance.get_clusters()

#returns the clusters & medoids
clusters = clarans_instance.get_clusters()
medoids = clarans_instance.get_medoids()
```




    10302




```python
Xy_scaled.shape
```




    (10302, 49)



**OK! Something strange is happening here.**

Our supposedly efficient CLARANS is grinding through on our 10302*49 dimension data, taking some 5 minutes to process just 1000 rows. I think that we are hitting the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), as the CLARANS implementaiton in pyclustering uses a euclidean distance metric. Euclidean distance breaks down in high dimensional space (see [here](https://towardsdatascience.com/how-to-cluster-in-high-dimensions-4ef693bacc6) and [here](https://en.wikipedia.org/wiki/Clustering_high-dimensional_data). There are several things we could try here:

 - One option would be to re-write the CLARANS method using a distance metric that is robust to high dimensionality, such as the Manhattan or Minkowski distance.
 - Apply an alternative clustering approach that is less affected by high dimensionality such as [spectral](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) or a Hierarchal approach as above or [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).
 - Anoter would be to reduce dimensionality using [*tSNE*](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) (see guides [here](https://www.datacamp.com/community/tutorials/introduction-t-sne) and [here](https://towardsdatascience.com/entity-embedding-using-t-sne-973cb5c730d7)).


```python
from sklearn.manifold import TSNE
```


```python
tsne_model = TSNE(n_components=3, verbose=1, random_state=0, n_iter=500)
tsne = tsne_model.fit_transform(Xy_scaled)
```

    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 10302 samples in 0.108s...
    [t-SNE] Computed neighbors for 10302 samples in 16.173s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 10302
    [t-SNE] Computed conditional probabilities for sample 2000 / 10302
    [t-SNE] Computed conditional probabilities for sample 3000 / 10302
    [t-SNE] Computed conditional probabilities for sample 4000 / 10302
    [t-SNE] Computed conditional probabilities for sample 5000 / 10302
    [t-SNE] Computed conditional probabilities for sample 6000 / 10302
    [t-SNE] Computed conditional probabilities for sample 7000 / 10302
    [t-SNE] Computed conditional probabilities for sample 8000 / 10302
    [t-SNE] Computed conditional probabilities for sample 9000 / 10302
    [t-SNE] Computed conditional probabilities for sample 10000 / 10302
    [t-SNE] Computed conditional probabilities for sample 10302 / 10302
    [t-SNE] Mean sigma: 0.534209
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 80.365295
    


```python
# find k clusters
results_tsne_clarans = dict()

k_cand = [3,7,15,30,45,60]
#k_cand.extend(list(np.arange(10,55,5)))

"""
data: Input data that is presented as list of points (objects), each point should be represented by list or tuple.
number_clusters: amount of clusters that should be allocated.
numlocal: the number of local minima obtained (amount of iterations for solving the problem).
maxneighbor: the maximum number of neighbors examined.     
The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.
"""

for k in k_cand:
    clarans_instance = clarans(data=tsne.tolist(), number_clusters=k, numlocal=1, maxneighbor=2)
    %time clarans_instance.process()
    #returns the clusters & medoids
    clusters = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()

    # convert cluster output
    cluster_array = pd.DataFrame([(x,e) for e,i in enumerate(clusters) for x in i if len(i)>1]).sort_values(by=0)[1].values
    
    # score
    score1 = 0 # silhouette_score(Xy_scaled_minkowski, cluster_array, metric='precomputed')
    score2 = silhouette_score(Xy_scaled, cluster_array,metric='correlation')
    
    # store
    results_tsne_clarans[k] = {'k':cluster_array,'s1':score1,'s2':score2}
    
plt.plot([i for i in results_tsne_clarans.keys()],[i['s1'] for i in results_tsne_clarans.values()],label='Minkowski')
plt.plot([i for i in results_tsne_clarans.keys()],[i['s2'] for i in results_tsne_clarans.values()],label='correlation')
plt.legend()
plt.xticks(k_cand);
```

    Wall time: 21.8 s
    Wall time: 1.6 s
    Wall time: 1.37 s
    Wall time: 3.03 s
    Wall time: 3.95 s
    Wall time: 6.02 s
    


![png]({{ "/assets/images/2020-02-20-clustering-fig10.png" }})



```python
fig, axs = plt.subplots(3,2,figsize=(20, 20),sharex=True,sharey=True)

for j,ax in zip(results_tsne_clarans.keys(),axs.flatten()):
    plotting = pd.DataFrame([(e,i) for e,i in enumerate(results_tsne_clarans[j]['k'])],columns=['id','k']).sort_values(by='id').sort_values(by='id')
    plotting['x'] = tsne[:,0]
    plotting['y'] = tsne[:,1]
    groups = plotting.groupby('k')
    
    for name, group in groups:
        ax.plot(group['x'], group['y'], marker='o', linestyle='', label=name)
        #ax.legend()
        ax.set_title(f'CLARANS-$tSNE$ ($k$={j})')
plt.show()   
```


![png]({{ "/assets/images/2020-02-20-clustering-fig11.png" }})



## 4.4 PAM partition-around-medoids (transformed and scaled features)
 - [PAM  is a variation of K-medoids](https://stats.stackexchange.com/a/141208/100439)
 - [Self defined PAM k-medoids in python](https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05)

## To Do list

### Inspect value's between sets of clusters and intra-clusters:
 - Descriptive statistics
 - Categorical features: with chi squared
 - Numerical features: t-test, non-paramteric tests, correlaiton
 

- [Notes on data mining](https://gist.github.com/AKST/d27b9006bb0f9670e370)
- http://eric.univ-lyon2.fr/~ricco/cours/slides/en/classif_interpretation.pdf
- https://online.stat.psu.edu/stat414/node/116/
- https://www2.cs.duke.edu/courses/spring18/compsci216/lectures/07-clustering.pdf

# *References*

- https://towardsdatascience.com/clustering-on-mixed-type-data-8bbd0a2569c3
- https://medium.com/@rumman1988/clustering-categorical-and-numerical-datatype-using-gower-distance-ab89b3aa90d9
- https://www2.cs.duke.edu/courses/spring18/compsci216/lectures/07-clustering.pdf
- https://towardsdatascience.com/hierarchical-clustering-on-categorical-data-in-r-a27e578f2995
- https://www.researchgate.net/post/What_is_the_best_way_for_cluster_analysis_when_you_have_mixed_type_of_data_categorical_and_scale
- https://www.google.com/search?client=firefox-b-d&q=python+gower+distance
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
- https://discuss.analyticsvidhya.com/t/clustering-technique-for-mixed-numeric-and-categorical-variables/6753
- https://stackoverflow.com/questions/24196897/r-distance-matrix-and-clustering-for-mixed-and-large-dataset
- https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
- https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
- https://rpubs.com/data_feelings/msda_data621_hw4
- https://pypi.org/project/gower/
- https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html
- https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
- https://www.rdocumentation.org/packages/cluster/versions/2.1.0/topics/pam
- https://github.com/annoviko/pyclustering/issues/499
- https://stats.stackexchange.com/questions/2717/clustering-with-a-distance-matrix
- https://www.kaggle.com/fabiendaniel/customer-segmentation
- https://dkopczyk.quantee.co.uk/claim-prediction/ (http://web.archive.org/web/20190429040211/https://dkopczyk.quantee.co.uk/claim-prediction/)
- https://www.casact.org/pubs/dpp/dpp08/08dpp170.pdf
- https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4
- https://www.uio.no/studier/emner/matnat/math/nedlagte-emner/STK2510/v08/undervisningsmateriale/ch8b.pdf
- https://github.com/annoviko/pyclustering/issues/499
- https://stackoverflow.com/questions/3081066/what-techniques-exists-in-r-to-visualize-a-distance-matrix
- https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
- https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data
- http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf
- https://www.researchgate.net/post/What_is_the_best_way_for_cluster_analysis_when_you_have_mixed_type_of_data_categorical_and_scale
- https://towardsdatascience.com/hierarchical-clustering-on-categorical-data-in-r-a27e578f2995
- https://gist.github.com/AKST/d27b9006bb0f9670e370



```python

```
