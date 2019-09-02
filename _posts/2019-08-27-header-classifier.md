---
layout: post
title:  "Machine Learning model to classify column headers for data cleaning tasks"
date:   2019-08-27 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [data-cleaning,classification,machine-learning]
---

###  Problem statement:
When dealing with large volumes of inbound data from multiple different sources the data recieved can often come in a variety of formats, structures and to varying standards. 
One particularly challenging isssue is data files that, although representing the same type of information, feature a variety of different label and data formats. 
For instance, addresses coded with "Zip" or "Postal Code", "Street" or "Line 1" and "£1000", "£1 K", "GBP 1000" or "one thousand pounds".

###  Machine Learning solution:
To build a model that can ingest messy labelled data (i.e. missing and with variable field names) and to make predctions for what the data fields are.


*The script to recreate the open source and synthetic dataset used to train and test the model is documented in this Jupyter notebook [here]().


```python
# load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
%matplotlib inline
```


```python
# change the paths to your local directory
data_path = 'P:\\MyWork\\demoColumnTyper\\data'
data_path_external = 'P:\\MyWork\\demoColumnTyper\\data\\external'
data_path_model = 'P:\\MyWork\\demoColumnTyper\\data\\model'
```

## Load the training-data
 
**Fields & Formats**<BR>
The training-data has the correct headers attached. We want to predict these on inbound messy "unlabelled" data.
I have included some very generic data fields and common formats. For instance, *money* varies between text and symbol currency values and *phone* includes a variety of formats and extensions. We also have some generic text_categorical and numeric values in there.


```python
training_data = pd.read_csv(os.path.join(data_path_model,'training_data.csv'))
training_data[:5]
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
      <th>address1</th>
      <th>address2</th>
      <th>city</th>
      <th>state</th>
      <th>zip</th>
      <th>lat</th>
      <th>lng</th>
      <th>money</th>
      <th>reference</th>
      <th>email</th>
      <th>person_name</th>
      <th>phone</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>txt_cat</th>
      <th>num_cat</th>
      <th>numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1745 T Street Southeast</td>
      <td>NaN</td>
      <td>Washington</td>
      <td>DC</td>
      <td>20020</td>
      <td>38.867033</td>
      <td>-76.979235</td>
      <td>GBP 487760.350</td>
      <td>SPCH308</td>
      <td>mex@yahoo.com</td>
      <td>Kimberlee Turlington</td>
      <td>0345 42 0274</td>
      <td>Kimberlee</td>
      <td>Turlington</td>
      <td>B</td>
      <td>3</td>
      <td>5181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6007 Applegate Lane</td>
      <td>NaN</td>
      <td>Louisville</td>
      <td>KY</td>
      <td>40219</td>
      <td>38.134301</td>
      <td>-85.649851</td>
      <td>€ 7321963.108</td>
      <td>VDEY870</td>
      <td>rg@aol.com</td>
      <td>Miguel Eveland</td>
      <td>095086-173-31-37</td>
      <td>Miguel</td>
      <td>Eveland</td>
      <td>A</td>
      <td>4</td>
      <td>2163</td>
    </tr>
    <tr>
      <th>2</th>
      <td>560 Penstock Drive</td>
      <td>NaN</td>
      <td>Grass Valley</td>
      <td>CA</td>
      <td>95945</td>
      <td>39.213076</td>
      <td>-121.077583</td>
      <td>EUR 3341992.053</td>
      <td>ZFPH671</td>
      <td>ejbyy@hotmail.com</td>
      <td>Alonzo Schroyer</td>
      <td>057843 018 15-85</td>
      <td>Alonzo</td>
      <td>Schroyer</td>
      <td>D</td>
      <td>4</td>
      <td>8193</td>
    </tr>
    <tr>
      <th>3</th>
      <td>150 Carter Street</td>
      <td>NaN</td>
      <td>Manchester</td>
      <td>CT</td>
      <td>6040</td>
      <td>41.765560</td>
      <td>-72.473091</td>
      <td>€ 4397323.917</td>
      <td>WMDG542</td>
      <td>tfanfw@gmail.com</td>
      <td>Devon Osei</td>
      <td>0698-1368378</td>
      <td>Devon</td>
      <td>Osei</td>
      <td>C</td>
      <td>1</td>
      <td>6134</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2721 Lindsay Avenue</td>
      <td>NaN</td>
      <td>Louisville</td>
      <td>KY</td>
      <td>40206</td>
      <td>38.263793</td>
      <td>-85.700243</td>
      <td>$ 7755233.425</td>
      <td>SLLE128</td>
      <td>xm@hotmail.com</td>
      <td>Val Hoffmeyer</td>
      <td>007 916 73 79</td>
      <td>Val</td>
      <td>Hoffmeyer</td>
      <td>E</td>
      <td>0</td>
      <td>6663</td>
    </tr>
  </tbody>
</table>
</div>



### Reshape our training-data so that it is ready for modelling.
*Using "X", we want to predict our target variable "Y"*


```python
Xy = pd.concat([pd.DataFrame(data={'X':list(training_data[col]),'Y':col}) for col in training_data.columns])
Xy.reset_index(drop=True,inplace=True)
Xy.fillna('',inplace=True) # fill nan values with empty string
Xy.sample(5)
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
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24710</th>
      <td>Dung</td>
      <td>first_name</td>
    </tr>
    <tr>
      <th>15162</th>
      <td>GBP 7421858.699</td>
      <td>money</td>
    </tr>
    <tr>
      <th>12446</th>
      <td>-76.9858</td>
      <td>lng</td>
    </tr>
    <tr>
      <th>8184</th>
      <td>95776</td>
      <td>zip</td>
    </tr>
    <tr>
      <th>245</th>
      <td>320 Northwest 22nd Street</td>
      <td>address1</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Engineering - simple
In raw text form the training-data values of X are unreadable by a computer.<BR>
We must engineer numeric representations termed features of the X values.

Here we define a function to create some generic features on our X data:
- length — The length of each string in characters
- num_length — The number of digits characters in a string
- space_length — The number of spaces within a string
- special_length — The number of special (non-alphanumeric) characters
- vowel_length - the number of vowels


```python
def feat_eng(df):
    """ function to engineer basic text features """
    df['X'] = df['X'].astype(str).copy()
    df['n_chars'] = df['X'].str.len() # string length
    df['n_digits'] = df['X'].str.count(r'[0-9]') # n digits
    df['n_space'] = df['X'].str.count(r' ') # n spaces
    df['n_special'] = df['X'].str.len() - df['X'].str.count(r'[\w+ ]') # n special chars
    df['n_vowels'] = df['X'].str.count(r'(?i)[aeiou]') # n vowels
    return df.fillna(0)

Xy = feat_eng(Xy)
```

## Feature Engineering - advanced

Here we use a more sophisticated approach to generate numeric features on our training data. It applies a simplified version of [TF-IDF (wiki)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). TF-IDF is a statistic used to measure how unique and important a word is in the body of text. For instance, the word "and" appears in many texts and yields little information regarding the topic of the text. The word "lawsuit" is generally rarer and yields inormation about the topic of the text.  Ok so this doesnt quite apply to this case as our texts are really just single values. However, we can still use this approach to identify and label key strings in our training data. 

We can see below a sample of the output. The TF-IDF measure has identified a number of strings as *important*. These include the strings:
 - "street","road" and "ave" (subset of "avenue") related to address data.
 - currency symbols.
 - "@", ".co" and ".com" on email addresses. I'd imagine this would change our results if we added a URL field to the training data?
 - We also see that small numbers 0,1,2 are returned. This is likely due to phone numbers beginning with zero and/or [Benfords law](https://en.wikipedia.org/wiki/Benford%27s_law) of small numbers.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char',
                             lowercase=True,
                             stop_words=None,
                             max_df=1.0,
                             min_df=0.005,
                             ngram_range=(1,6),
                             binary=True,
                             norm=None,
                             use_idf=False
                            )

X_vect = vectorizer.fit_transform(Xy['X'].astype(str).str.encode('utf-8'))
#print(X_vect.shape)

X_vect_df = pd.DataFrame(data = X_vect.todense(), columns=vectorizer.get_feature_names())
Xy_vect = Xy.merge(X_vect_df,how='left',left_index=True,right_index=True).copy()
#Xy_vect[['X','£']].loc[Xy_vect['X'].str.contains('£')==True]
```


```python
Xy_vect.sample(5)[['X','street','road','ave','£','$','@','.co','.com','drive','0']]
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
      <th>X</th>
      <th>street</th>
      <th>road</th>
      <th>ave</th>
      <th>£</th>
      <th>$</th>
      <th>@</th>
      <th>.co</th>
      <th>.com</th>
      <th>drive</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27454</th>
      <td>Gural</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22112</th>
      <td>004-570 4-61</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22620</th>
      <td>07 6246014</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10521</th>
      <td>38.512894</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16661</th>
      <td>VSQO404</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Selection - univariate 

Ok. So that lest step actually created about 11,000 numeric features on our training data. We need to trim this down by selecting the data value features that we believe are most likley to be correlated to our target column headers. Lets use some Chi-square correlation analyses to test the features. 

**If you want to skip the stats, the bar plot below shows the 75 features that were found to have the highest correlation to our target column headers that we want to predict.**

### [Chi-squared](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)

*The chi-square test is a statistical test of independence to determine the dependency of two variables. 
It shares similarities with coefficient of determination, R².
However, chi-square test is only applicable to categorical or nominal data while R² is only applicable to numeric data.*
 - If Statistic >= Critical Value: significant result, reject null hypothesis (H0), dependent. *There **IS** a relationship*.
 - If Statistic < Critical Value: not significant result, fail to reject null hypothesis (H0), independent. *There **IS NOT** a relationship*.


```python
# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2

prob = 0.95
alpha = 1.0 - prob

# run test
stat, p, dof, expected = chi2_contingency(pd.crosstab(Xy['Y'],Xy['n_vowels']))

# dof
print('\ndof=%d' % dof)

# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('\nprobability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('\tDependent (reject H0)')
else:
    print('\tIndependent (fail to reject H0)')
    
    
# interpret p-value
print('\nsignificance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('\tDependent (reject H0)')
else:
    print('\tIndependent (fail to reject H0)')
```

    
    dof=240
    
    probability=0.950, critical=277.138, stat=60001.619
    	Dependent (reject H0)
    
    significance=0.050, p=0.000
    	Dependent (reject H0)
    


```python
features = Xy_vect.columns[2:]
stats = list()
p_values = list()
dofs = list()

for feat in features:
    # run test
    stat, p, dof, expected = chi2_contingency(pd.crosstab(Xy_vect['Y'],Xy_vect[feat]))
    stats.append(stat)
    p_values.append(p)
    dofs.append(dof)

chi2_results = pd.DataFrame({'feature':features,'X2':stats,'DoF':dofs,'pvalue':p_values,'sig':[x<=0.05 for x in p_values]})
chi2_results.sort_values(by='X2',ascending=False, inplace=True)
```


```python
fig,axs = plt.subplots(figsize=(25,5))
n = 75
axs.bar(x = range(n), height=chi2_results['X2'][:n])
axs.set_xticks(range(n));
axs.set_xticklabels(chi2_results['feature'][:n],rotation=90)
axs.set_title('Results of Correlation Analysis (top 75 features)',fontsize=15);
```


![png]({{ "/assets/images/2019-08-27-header-classifier-fig1.png" }})


# Model Training

**Here we are building a model that can predict the column type using: i) the training data, and ii) the subset of engineered-features selected by correlation analysis.**


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
```


```python
model_features = chi2_results['feature'][:100].values
```


```python
# model parameters
NB_params = {'model__alpha':(1e-1, 1e-3)}

NB_pipe = Pipeline([('model', MultinomialNB())])

gs_NB = GridSearchCV(NB_pipe, param_grid=NB_params, n_jobs=2, cv=5)
gs_NB = gs_NB.fit(Xy_vect[model_features], Xy_vect['Y'])
print(gs_NB.best_score_, gs_NB.best_params_)
```

    0.7691470588235294 {'model__alpha': 0.1}
    

# Load Test Data
Here lets load the testing data. Note that the data is missing column headers. If we were cleaning up this data we would have to manually add these.


```python
testing_data = [pd.read_csv(os.path.join(data_path_model,x),skiprows=1,header=None) for x in os.listdir(data_path_model) if 'testing' in x]
testing_data = pd.concat(testing_data)
testing_data[:10]
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
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11501 Maple Way</td>
      <td>NaN</td>
      <td>Louisville</td>
      <td>KY</td>
      <td>40229</td>
      <td>38.097617</td>
      <td>-85.659825</td>
      <td>EUR 2274211.206</td>
      <td>WBSW110</td>
      <td>vmtjcbp@gmail.com</td>
      <td>Jena Quilliams</td>
      <td>039520 805 804</td>
      <td>Jena</td>
      <td>Quilliams</td>
      <td>C</td>
      <td>2</td>
      <td>4795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>98 Lee Drive</td>
      <td>NaN</td>
      <td>Annapolis</td>
      <td>MD</td>
      <td>21403</td>
      <td>38.933313</td>
      <td>-76.493310</td>
      <td>EUR 3405197.530</td>
      <td>SWLP214</td>
      <td>hdst@gmail.com</td>
      <td>Laila Arpin</td>
      <td>08249 265 6568</td>
      <td>Laila</td>
      <td>Arpin</td>
      <td>B</td>
      <td>2</td>
      <td>5021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>126 Sunshine Road</td>
      <td>O</td>
      <td>Savannah</td>
      <td>GA</td>
      <td>31405</td>
      <td>32.059784</td>
      <td>-81.202271</td>
      <td>EUR 7458286.525</td>
      <td>VUYW948</td>
      <td>aminwvel@yahoo.com</td>
      <td>Cesar Severson</td>
      <td>08-095 51-90</td>
      <td>Cesar</td>
      <td>Severson</td>
      <td>A</td>
      <td>0</td>
      <td>5600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4313 Wisconsin Street</td>
      <td>#APT 000007</td>
      <td>Anchorage</td>
      <td>AK</td>
      <td>99517</td>
      <td>61.181060</td>
      <td>-149.942792</td>
      <td>€ 5481056.898</td>
      <td>SEXL728</td>
      <td>gevypgf@mail.kz</td>
      <td>Chi Hollinsworth</td>
      <td>030164-378 59 20</td>
      <td>Chi</td>
      <td>Hollinsworth</td>
      <td>B</td>
      <td>1</td>
      <td>3746</td>
    </tr>
    <tr>
      <th>4</th>
      <td>829 Main Street</td>
      <td>NaN</td>
      <td>Manchester</td>
      <td>CT</td>
      <td>6040</td>
      <td>41.770678</td>
      <td>-72.520917</td>
      <td>GBP 4247524.506</td>
      <td>VQZY333</td>
      <td>yibhnaha@mail.kz</td>
      <td>Jan Reagans</td>
      <td>00132-612-74-32</td>
      <td>Jan</td>
      <td>Reagans</td>
      <td>C</td>
      <td>4</td>
      <td>7541</td>
    </tr>
    <tr>
      <th>5</th>
      <td>37 Spring Street</td>
      <td>NaN</td>
      <td>Groton</td>
      <td>CT</td>
      <td>6340</td>
      <td>41.320683</td>
      <td>-71.991475</td>
      <td>$ 5717317.815</td>
      <td>BSRO462</td>
      <td>lppsimxwb@mail.kz</td>
      <td>Marcos Hoistion</td>
      <td>018-8094-25</td>
      <td>Marcos</td>
      <td>Hoistion</td>
      <td>E</td>
      <td>3</td>
      <td>5201</td>
    </tr>
    <tr>
      <th>6</th>
      <td>266 South J Street</td>
      <td>NaN</td>
      <td>Livermore</td>
      <td>CA</td>
      <td>94550</td>
      <td>37.680570</td>
      <td>-121.768021</td>
      <td>$ 1921095.325</td>
      <td>SNPX594</td>
      <td>tgjt@aol.com</td>
      <td>Loan Wadsworth</td>
      <td>040 9272 23</td>
      <td>Loan</td>
      <td>Wadsworth</td>
      <td>C</td>
      <td>3</td>
      <td>7463</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7952 South Algonquian Way</td>
      <td>NaN</td>
      <td>Aurora</td>
      <td>CO</td>
      <td>80016</td>
      <td>39.573350</td>
      <td>-104.716211</td>
      <td>$ 8932865.979</td>
      <td>ZNNS529</td>
      <td>snjral@aol.com</td>
      <td>Marisa Blaskovich</td>
      <td>087 469 8912</td>
      <td>Marisa</td>
      <td>Blaskovich</td>
      <td>B</td>
      <td>0</td>
      <td>7738</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9223 Elgin Circle</td>
      <td>NaN</td>
      <td>Anchorage</td>
      <td>AK</td>
      <td>99502</td>
      <td>61.136803</td>
      <td>-149.965463</td>
      <td>GBP 3666134.645</td>
      <td>ZCJN210</td>
      <td>ebl@gmail.com</td>
      <td>Nguyet Lytch</td>
      <td>00748296 39 4</td>
      <td>Nguyet</td>
      <td>Lytch</td>
      <td>A</td>
      <td>1</td>
      <td>6734</td>
    </tr>
    <tr>
      <th>9</th>
      <td>224 Michael Sears Road</td>
      <td>NaN</td>
      <td>Belchertown</td>
      <td>MA</td>
      <td>1007</td>
      <td>42.234610</td>
      <td>-72.359730</td>
      <td>EUR 2406344.382</td>
      <td>ZOSG574</td>
      <td>qeaf@aol.com</td>
      <td>Corrie Tolhurst</td>
      <td>053601-410-69-0</td>
      <td>Corrie</td>
      <td>Tolhurst</td>
      <td>D</td>
      <td>2</td>
      <td>1521</td>
    </tr>
  </tbody>
</table>
</div>



Below we apply the same feature engineering and selection as applied to our training data. 


```python
testing_data = [pd.read_csv(os.path.join(data_path_model,x)) for x in os.listdir(data_path_model) if 'testing' in x]
testing_data = pd.concat(testing_data)
Xy_test = pd.concat([pd.DataFrame(data={'X':list(testing_data[col]),'Y':col}) for col in testing_data.columns])
Xy_test.reset_index(drop=True,inplace=True)
Xy_test.fillna('',inplace=True) # fill nan values with empty string

Xy_test = feat_eng(Xy_test)

test_vect = vectorizer.transform(Xy_test['X'].astype(str).str.encode('utf-8'))
test_vect = pd.DataFrame(data = test_vect.todense(), columns=vectorizer.get_feature_names())
Xy_test = Xy_test.merge(test_vect,how='left',left_index=True,right_index=True).copy()
```

### How good is the model?
Overall our model is able to correctly identify and label 77 % of the test data. Remember the model has never seen the test data before so that is pretty good. 


```python
Xy_test['pred'] = gs_NB.predict(Xy_test[model_features])
print('Model accuracy: %.3f' %np.mean(Xy_test['pred'] == Xy_test['Y']))
```

    Model accuracy: 0.774
    

There are also some other things we could try to improve our model further. Below we see a more detailed report and bar plot that shows us how good our model was on each data column we were trying to predict. We could for isntance, look where our model is underperfoming and try to create and select other features that better capture the data fields. 

[See this guide for interpretation of accuracy, precision, recall etc..](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)


```python
print(metrics.classification_report(Xy_test['Y'], Xy_test['pred']))
```

                  precision    recall  f1-score   support
    
        address1       0.53      0.99      0.69      1220
        address2       0.86      0.08      0.14      1220
            city       0.64      0.56      0.60      1220
           email       1.00      1.00      1.00      1220
      first_name       0.46      0.51      0.48      1220
       last_name       0.51      0.46      0.49      1220
             lat       0.98      1.00      0.99      1220
             lng       0.88      1.00      0.94      1220
           money       1.00      1.00      1.00      1220
         num_cat       0.82      1.00      0.90      1220
         numeric       0.70      0.68      0.69      1220
     person_name       0.88      0.98      0.92      1220
           phone       0.99      0.85      0.91      1220
       reference       1.00      0.96      0.98      1220
           state       0.80      0.50      0.61      1220
         txt_cat       0.74      1.00      0.85      1220
             zip       0.68      0.59      0.63      1220
    
       micro avg       0.77      0.77      0.77     20740
       macro avg       0.79      0.77      0.75     20740
    weighted avg       0.79      0.77      0.75     20740
    
    

**The bar charts below show the distribution of predictions for each target.**

For instance:
 - **address1**: the model was able to identify all of the address1 values in the testing data. 
 - **city**: The model indentified roughly 50% of the correct city values, but often confused these with human names.


```python
fig,axs = plt.subplots(5,4,figsize=(10,20),sharey=True)

for target,ax in zip(Xy_test['Y'].unique(),axs.flatten()):
    heights = Xy_test.loc[Xy_test['Y']==target,'pred'].value_counts()
    x = range(len(heights))
    ax.bar(x=x,height=heights/1220,color='blue',edgecolor='black',alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(heights.index,rotation=90)
    ax.set_title(target)
    
plt.tight_layout()
```


![png]({{ "/assets/images/2019-08-27-header-classifier-fig2.png" }})


Thats all folks.<BR>
Thank you for reading.

Ben Postance
