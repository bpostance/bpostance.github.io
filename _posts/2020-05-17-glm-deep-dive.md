---
layout: post
title:  "A deep dive on GLM's in frequency severity models"
date:   2020-05-17 18:00:00 +0000
categories: [blog-post,data-analysis]
tags: [glm,insurance,monte carlo]
math: true
comments: true
---
<img src="/assets/images/2020-05-17-glm-fig1.png" alt="drawing" width="800" height="350"/>


**[Jupyter notebook here](https://github.com/bpostance/dsc.learn/blob/main/Risk/02-GLM-FrequencySeverity-model.ipynb)**

This notebook is a deep dive into [General Linear Models (GLM's)](https://online.stat.psu.edu/stat504/node/216/) with a focus on the GLM's used in insurance risk modeling and pricing (Yan, J. 2010).I have used GLM's before including: a Logistic Regression for landslide geo-hazards (Postance, 2017), for modeling extreme rainfall and developing catastrophe models (Postance, 2017). The motivation for this post is to develop a deeper knowledge of the assumptions and application of the models and methods used by Insurance Actuaries, and to better understand how these compare to machine learning methods.


### Case study dataset: motorcylce insurance
The [Ohlsson dataset](https://cran.r-project.org/web/packages/insuranceData/insuranceData.pdf#Rfn.dataOhlsson.1) is from a former Swedish insurance company Wasa. The data includes aggregated  customer, policy and claims data for 64,548 motorcycle coverages for the period 1994-1998. The data is used extensively in actuarial training and syllabus worldwide [Ohlsson, (2010)](https://www.springer.com/gp/book/9783642107900#aboutBook).


Variables include:
- *data available [here](https://staff.math.su.se/esbj/GLMbook/case.html)*

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
      <th>dtype</th>
      <th>null</th>
      <th>nunique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Age</td>
      <td>float64</td>
      <td>0</td>
      <td>85</td>
    </tr>
    <tr>
      <td>Sex</td>
      <td>category</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Geog</td>
      <td>category</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>EV</td>
      <td>category</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VehAge</td>
      <td>int64</td>
      <td>0</td>
      <td>85</td>
    </tr>
    <tr>
      <td>NCD</td>
      <td>category</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PYrs</td>
      <td>float64</td>
      <td>0</td>
      <td>2577</td>
    </tr>
    <tr>
      <td>Claims</td>
      <td>int64</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Severity</td>
      <td>float64</td>
      <td>0</td>
      <td>590</td>
    </tr>
    <tr>
      <td>Claim</td>
      <td>int64</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>SeverityAvg</td>
      <td>float64</td>
      <td>0</td>
      <td>590</td>
    </tr>
  </tbody>
</table>
</div>



### EDA

All data: low number of claims and frequency (1% freq)


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
      <th>Age</th>
      <th>VehAge</th>
      <th>PYrs</th>
      <th>Claims</th>
      <th>Severity</th>
      <th>Claim</th>
      <th>SeverityAvg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
      <td>64548.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>42.416062</td>
      <td>12.540063</td>
      <td>1.010759</td>
      <td>0.010798</td>
      <td>264.017785</td>
      <td>0.010380</td>
      <td>246.964360</td>
    </tr>
    <tr>
      <td>std</td>
      <td>12.980960</td>
      <td>9.727445</td>
      <td>1.307356</td>
      <td>0.107323</td>
      <td>4694.693604</td>
      <td>0.101352</td>
      <td>4198.994975</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002740</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>31.000000</td>
      <td>5.000000</td>
      <td>0.463014</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>44.000000</td>
      <td>12.000000</td>
      <td>0.827397</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>52.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>92.000000</td>
      <td>99.000000</td>
      <td>31.339730</td>
      <td>2.000000</td>
      <td>365347.000000</td>
      <td>1.000000</td>
      <td>211254.000000</td>
    </tr>
  </tbody>
</table>
</div>



Claims data: 


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
      <th>Age</th>
      <th>VehAge</th>
      <th>PYrs</th>
      <th>Claims</th>
      <th>Severity</th>
      <th>Claim</th>
      <th>SeverityAvg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>670.000000</td>
      <td>670.000000</td>
      <td>670.000000</td>
      <td>670.000000</td>
      <td>670.000000</td>
      <td>670.0</td>
      <td>670.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>35.476119</td>
      <td>7.965672</td>
      <td>1.579415</td>
      <td>1.040299</td>
      <td>25435.552239</td>
      <td>1.0</td>
      <td>23792.620149</td>
    </tr>
    <tr>
      <td>std</td>
      <td>12.851056</td>
      <td>6.768896</td>
      <td>2.983317</td>
      <td>0.196805</td>
      <td>38539.415033</td>
      <td>0.0</td>
      <td>33765.250000</td>
    </tr>
    <tr>
      <td>min</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>0.002740</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>1.0</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>25.000000</td>
      <td>2.000000</td>
      <td>0.430137</td>
      <td>1.000000</td>
      <td>3031.500000</td>
      <td>1.0</td>
      <td>3007.750000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>30.000000</td>
      <td>7.000000</td>
      <td>0.790411</td>
      <td>1.000000</td>
      <td>9015.000000</td>
      <td>1.0</td>
      <td>8723.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>47.000000</td>
      <td>12.000000</td>
      <td>1.497945</td>
      <td>1.000000</td>
      <td>29304.500000</td>
      <td>1.0</td>
      <td>26787.750000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>68.000000</td>
      <td>55.000000</td>
      <td>31.167120</td>
      <td>2.000000</td>
      <td>365347.000000</td>
      <td>1.0</td>
      <td>211254.000000</td>
    </tr>
  </tbody>
</table>
</div>


Claims and losses by each variable:


![png]({{"/assets/images/2020-05-17-glm-fig7_0.png"}})


### Modeling

***Train-Test split***

```python
# train-test splits stratifies on claims
# take copies to overcome chained assignment
train,test = train_test_split(df,test_size=0.3,random_state=1990,stratify=df['Claim'])
train = df.loc[train.index].copy()
test = df.loc[test.index].copy()
a,b,c,d = train['Claim'].sum(),train['Severity'].sum(),test['Claim'].sum(),test['Severity'].sum()
print(f"Frequency\nTrain:\t{len(train)}\t{a}\t${b}\nTest:\t{len(test)}\t{c}\t${d}\n")

# severity train test
train_severity = train.loc[train['Claim']>0].copy()
test_severity = test.loc[test['Claim']>0].copy()
a,b,c,d = train_severity['Claim'].sum(),train_severity['Severity'].sum(),test_severity['Claim'].sum(),test_severity['Severity'].sum()
print(f"Severity\nTrain:\t{len(train_severity)}\t{a}\t${b}\nTest:\t{len(test_severity)}\t{c}\t${d}\n")
```

    Frequency
    Train:	45183	469	$11664342.0
    Test:	19365	201	$5377478.0
    
    Severity
    Train:	469	469	$11664342.0
    Test:	201	201	$5377478.0
    
    

### *Claim Frequency*

For predicting the occurrence of a single claim (i.e. binary classification) one can use the Binomial distribution (a.k.a Bernoulli trial or coin-toss experiment).

When predicting claim counts or frequency, $Y$, a model that produces Poisson distributed outputs is required. For instance, a Poisson model is suitable for estimating the number of insurance claims per policy per year, or to estimate the number of car crashes per month. 

The key components and assumptions of a Poisson distributed process are: 
 1. event occurrence is independent of other events. 
 1. events occur within a fixed period of time.
 1. the mean a variance of the distribution are equal e.g. $mu(X) = Var(X) = λ$
 
[*STAT 504: Poisson Distribution*](https://online.stat.psu.edu/stat504/node/57/)


If the mean and variance are unequal the distribution is said to be over-dispersed (var > mean) or under-dispersed (var < mean). Over-dispersion commonly arises in data where there are large number of zero's (a.k.a [zero-inflated](https://en.wikipedia.org/wiki/Zero-inflated_model)).

In the case of zero-inflated data, it is "*A sound practice is to estimate both Poisson and negative binomial models.*" [*Cameron, 2013*](http://faculty.econ.ucdavis.edu/faculty/cameron/racd2/). Also see this practical example for [beverage consumption in pdummy_xyhon](https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/)

In GLM's, link-functions are applied in order to make the mean outcome (prediction) fit to some linear model of input variables from other distributions. "*A natural fit for count variables that follow the Poisson or negative binomial distribution is the log link. The log link exponentiates the linear predictors. It does not log transform the outcome variable.*" - [*Count Models: Understanding the Log Link Function,TAF*](https://www.theanalysisfactor.com/count-models-understanding-the-log-link-function/)

For more information on link-functions see also [here](https://bookdown.org/castillo_sam_d/Exam-PA-Study-Manual/glms-for-classification.html#link-functions).

![link functions](https://i0.wp.com/www.theanalysisfactor.com/wp-content/uploads/2016/12/StataCombos-CM2-Blog-JM.png?w=535&ssl=1)


Lastly, for any form of count prediction model one can also set an offset or exposure. An offset, if it is known, is applied in order to account for the relative differences in exposure time for of a set of inputs. For example, in insurance claims we might expect to see more claims on an account with 20 years worth of annual policies compared to an account with a single policy year. Offsets account for the relative exposure, surface area, population size, etc and is akin to the relative frequency of occurrence (*Claims/years*). See these intuitive SO answers [here](https://stats.stackexchange.com/questions/232666/should-i-use-an-offset-for-my-poisson-glm), [here](https://github.com/statsmodels/statsmodels/issues/1486#issuecomment-40945831), and [here](https://stats.stackexchange.com/questions/25415/using-offset-in-binomial-model-to-account-for-increased-numbers-of-patients).


```python
# Mean & Variance
mu = df.Claims.mean()
var = np.mean([abs(x - mu)**2 for x in df.Claims])
print(f'mu =  {mu:.4f}\nvar = {var:.4f}')
```

    mu =  0.0108
    var = 0.0115
    

Here we observe an over-dispersed zero-inflated case as the variance of claim occurrence ($v=0.0115$) exceeds its mean ($mu=0.0108$).

As suggested in Cameron (2013) we should therefore try both $Poisson$ and $Negative Binomial$ distributions.

For good measure, and to illustrate its relationship, lets also include a $Binomial$ distribution model. The Binomial model with a logit link is equivalent to a binary Logistic Regression model [[a](https://www.researchgate.net/post/what_is_the_difference_between_running_a_binary_logistic_regression_and_generalised_linear_model),[b](https://towardsdatascience.com/the-binomial-regression-model-everything-you-need-to-know-5216f1a483d3)]. Modeling A binary outcome is not a totally unreasonable approach in this case given that the number of accounts with claims $n>1$ is low (22) and as the $Binomial$ distribution extends to a $Poisson$ when trials $N>20$ is high and $p<0.05$ is low (see [wiki](https://en.wikipedia.org/wiki/Poisson_distribution#Related_distributions), [here](https://math.stackexchange.com/questions/1050184/difference-between-poisson-and-binomial-distributions) and [here](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc331.htm)).

However, there is one change we need to make with the Binomial model. That is to alter the way we handle exposure. A few hours of research on the matter led me down the rabbit hole of conflicting ideas in textbooks, papers [[i](https://doi.org/10.1093/ije/dyu029)] and debates on CrossValidated [[a](https://stats.stackexchange.com/questions/246318/difference-between-offset-and-weights),[b](https://stats.stackexchange.com/questions/25415/using-offset-in-binomial-model-to-account-for-increased-numbers-of-patients/35478)]. In contrast to Poisson and neg binomial there is no way to add a constant term or offset in the binomial formulation [(see here)](https://stats.stackexchange.com/a/35478/100439). Rather it is appropriate to either: [include the exposure as a predictor variable](https://stats.stackexchange.com/a/35436/100439), or to use weights for each observation (see [here](https://stackoverflow.com/a/62798889/4538066) and the [statsmodels guidance on methods for GLM with weights and observed frequencies](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM). I opted for the weighted GLM. The model output of the binomial GLM is the probability of at least 1 claim occurring weighted by the observation time $t=Pyrs$. Note there is no equivalent setting on the predict side, the predictions assume a discrete equivalent time exposure t=1.

In addition it is common in insurance risk models to use a [quasi-poisson or zero inflated Poisson (ZIP)](https://towardsdatascience.com/an-illustrated-guide-to-the-zero-inflated-poisson-model-b22833343057) model in scenarios with high instances of zero claims. In data science and machine learning we would refer to this as an unbalanced learning problem (see bootstrap, cross validation, SMOTE). The ZIP model combines:
 - a binomial model to determine the likelihood of one or more claims occurring (0/1)
 - a negative binomial or poisson to estimate the number of claims (0...n)
 - a severity model to estimate the avg size of each claim (1...n)

Statsmodels uses patsy [formula notation](https://www.statsmodels.org/devel/example_formulas.html). This includes: [notation for categorical variables ](https://www.statsmodels.org/devel/contrasts.html), setting [reference/base levels](https://stackoverflow.com/a/22439820/4538066), [encoding options](https://www.statsmodels.org/devel/contrasts.html), and [operators](https://www.statsmodels.org/devel/example_formulas.html#categorical-variables).

```python
# # examples of formula notation in smf
# print(' + '.join(train.columns))
# expr = "Claims ~ Age+C(Sex)+C(Geog, Treatment(reference=3))+EV+VehAge+NCD"

# including PYrs as parameter commented out in glm()
expr = "Claims ~ Age + Sex + Geog + EV + VehAge + NCD" # + np.log(PYrs)

FreqPoisson = smf.glm(formula=expr,
                      data=train,
                      offset=np.log(train['PYrs']),
                      family=sm.families.Poisson(link=sm.families.links.log())).fit()

FreqNegBin = smf.glm(formula=expr,
                      data=train,
                      offset=np.log(train['PYrs']),
                      family=sm.families.NegativeBinomial(link=sm.families.links.log())).fit()

# uses the binary "Claim" field as target
# offset is Pyrs (Years complete 0.0...n)
# aka a binary logistic regression
FreqBinom = smf.glm(formula="Claim ~ Age + Sex + Geog + EV + VehAge + NCD " ,
                    data=train,
                    freq_weights=train['PYrs'],
                    family=sm.families.Binomial(link=sm.families.links.logit())).fit()
```

### Model coefficients

***Poisson Parameters***
- [Poisson GLM params](https://stackoverflow.com/questions/14923684/interpreting-the-output-of-glm-for-poisson-regression)
- [Find lambda](https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram)

We can derive the model output using predict or the raw coefficients. Sampling the Poisson rate (*lambda*) illustrates the difference in predicted rates for the intercept and for a when a driver is male. 


```python
FreqPoisson.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Claims</td>      <th>  No. Observations:  </th>  <td> 45183</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td> 45161</td> 
</tr>
<tr>
  <th>Model Family:</th>         <td>Poisson</td>     <th>  Df Model:          </th>  <td>    21</td> 
</tr>
<tr>
  <th>Link Function:</th>          <td>log</td>       <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -2541.7</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 14 Oct 2020</td> <th>  Deviance:          </th> <td>  4135.6</td>
</tr>
<tr>
  <th>Time:</th>                <td>21:43:22</td>     <th>  Pearson chi2:      </th> <td>1.83e+05</td>
</tr>
<tr>
  <th>No. Iterations:</th>         <td>22</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -0.9052</td> <td>    0.284</td> <td>   -3.184</td> <td> 0.001</td> <td>   -1.462</td> <td>   -0.348</td>
</tr>
<tr>
  <th>Sex[T.M]</th>  <td>    0.3881</td> <td>    0.162</td> <td>    2.392</td> <td> 0.017</td> <td>    0.070</td> <td>    0.706</td>
</tr>
<tr>
  <th>Geog[T.2]</th> <td>   -0.6478</td> <td>    0.135</td> <td>   -4.811</td> <td> 0.000</td> <td>   -0.912</td> <td>   -0.384</td>
</tr>
<tr>
  <th>Geog[T.3]</th> <td>   -0.9043</td> <td>    0.137</td> <td>   -6.606</td> <td> 0.000</td> <td>   -1.173</td> <td>   -0.636</td>
</tr>
<tr>
  <th>Geog[T.4]</th> <td>   -1.3826</td> <td>    0.123</td> <td>  -11.254</td> <td> 0.000</td> <td>   -1.623</td> <td>   -1.142</td>
</tr>
<tr>
  <th>Geog[T.5]</th> <td>   -1.5218</td> <td>    0.389</td> <td>   -3.912</td> <td> 0.000</td> <td>   -2.284</td> <td>   -0.759</td>
</tr>
<tr>
  <th>Geog[T.6]</th> <td>   -1.4581</td> <td>    0.315</td> <td>   -4.623</td> <td> 0.000</td> <td>   -2.076</td> <td>   -0.840</td>
</tr>
<tr>
  <th>Geog[T.7]</th> <td>  -22.0308</td> <td> 1.77e+04</td> <td>   -0.001</td> <td> 0.999</td> <td>-3.47e+04</td> <td> 3.46e+04</td>
</tr>
<tr>
  <th>EV[T.2]</th>   <td>    0.0923</td> <td>    0.237</td> <td>    0.389</td> <td> 0.697</td> <td>   -0.372</td> <td>    0.557</td>
</tr>
<tr>
  <th>EV[T.3]</th>   <td>   -0.4219</td> <td>    0.199</td> <td>   -2.115</td> <td> 0.034</td> <td>   -0.813</td> <td>   -0.031</td>
</tr>
<tr>
  <th>EV[T.4]</th>   <td>   -0.3602</td> <td>    0.215</td> <td>   -1.678</td> <td> 0.093</td> <td>   -0.781</td> <td>    0.061</td>
</tr>
<tr>
  <th>EV[T.5]</th>   <td>   -0.0334</td> <td>    0.204</td> <td>   -0.164</td> <td> 0.870</td> <td>   -0.433</td> <td>    0.367</td>
</tr>
<tr>
  <th>EV[T.6]</th>   <td>    0.4132</td> <td>    0.202</td> <td>    2.042</td> <td> 0.041</td> <td>    0.017</td> <td>    0.810</td>
</tr>
<tr>
  <th>EV[T.7]</th>   <td>    0.3316</td> <td>    0.483</td> <td>    0.687</td> <td> 0.492</td> <td>   -0.614</td> <td>    1.278</td>
</tr>
<tr>
  <th>NCD[T.2]</th>  <td>   -0.1441</td> <td>    0.181</td> <td>   -0.795</td> <td> 0.426</td> <td>   -0.499</td> <td>    0.211</td>
</tr>
<tr>
  <th>NCD[T.3]</th>  <td>    0.0184</td> <td>    0.192</td> <td>    0.096</td> <td> 0.923</td> <td>   -0.357</td> <td>    0.394</td>
</tr>
<tr>
  <th>NCD[T.4]</th>  <td>    0.3047</td> <td>    0.184</td> <td>    1.660</td> <td> 0.097</td> <td>   -0.055</td> <td>    0.664</td>
</tr>
<tr>
  <th>NCD[T.5]</th>  <td>   -0.0535</td> <td>    0.215</td> <td>   -0.249</td> <td> 0.804</td> <td>   -0.475</td> <td>    0.368</td>
</tr>
<tr>
  <th>NCD[T.6]</th>  <td>    0.0967</td> <td>    0.206</td> <td>    0.470</td> <td> 0.639</td> <td>   -0.307</td> <td>    0.501</td>
</tr>
<tr>
  <th>NCD[T.7]</th>  <td>    0.1835</td> <td>    0.137</td> <td>    1.334</td> <td> 0.182</td> <td>   -0.086</td> <td>    0.453</td>
</tr>
<tr>
  <th>Age</th>       <td>   -0.0580</td> <td>    0.004</td> <td>  -13.633</td> <td> 0.000</td> <td>   -0.066</td> <td>   -0.050</td>
</tr>
<tr>
  <th>VehAge</th>    <td>   -0.0762</td> <td>    0.008</td> <td>   -9.781</td> <td> 0.000</td> <td>   -0.091</td> <td>   -0.061</td>
</tr>
</table>




    Lambda intercept: 0.40
    Lambda intercept + male: 0.60
    


![png]({{"/assets/images/2020-05-17-glm-fig18_1.png"}})



***Negative binomial coefficients and incidence rate ratio***

 - https://stats.idre.ucla.edu/stata/dae/negative-binomial-regression/
 - https://stats.stackexchange.com/questions/17006/interpretation-of-incidence-rate-ratios
 - https://stats.stackexchange.com/questions/414752/how-to-interpret-incidence-rate-ratio
 - https://www.cdc.gov/csels/dsepd/ss1978/lesson3/section5.html


```python
print(FreqNegBin.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                 Claims   No. Observations:                45183
    Model:                            GLM   Df Residuals:                    45161
    Model Family:        NegativeBinomial   Df Model:                           21
    Link Function:                    log   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -2535.9
    Date:                Wed, 14 Oct 2020   Deviance:                       3754.7
    Time:                        21:43:23   Pearson chi2:                 1.82e+05
    No. Iterations:                    21                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.8851      0.288     -3.069      0.002      -1.450      -0.320
    Sex[T.M]       0.3927      0.163      2.403      0.016       0.072       0.713
    Geog[T.2]     -0.6545      0.137     -4.778      0.000      -0.923      -0.386
    Geog[T.3]     -0.9104      0.139     -6.541      0.000      -1.183      -0.638
    Geog[T.4]     -1.3888      0.125    -11.100      0.000      -1.634      -1.144
    Geog[T.5]     -1.5350      0.391     -3.928      0.000      -2.301      -0.769
    Geog[T.6]     -1.4693      0.317     -4.635      0.000      -2.091      -0.848
    Geog[T.7]    -21.1594   1.13e+04     -0.002      0.999   -2.22e+04    2.22e+04
    EV[T.2]        0.0957      0.240      0.399      0.690      -0.375       0.566
    EV[T.3]       -0.4359      0.202     -2.155      0.031      -0.832      -0.039
    EV[T.4]       -0.3671      0.217     -1.690      0.091      -0.793       0.059
    EV[T.5]       -0.0319      0.207     -0.154      0.877      -0.437       0.373
    EV[T.6]        0.4212      0.205      2.054      0.040       0.019       0.823
    EV[T.7]        0.3309      0.486      0.681      0.496      -0.622       1.283
    NCD[T.2]      -0.1456      0.183     -0.795      0.427      -0.505       0.214
    NCD[T.3]       0.0171      0.194      0.089      0.929      -0.362       0.397
    NCD[T.4]       0.2965      0.186      1.591      0.112      -0.069       0.662
    NCD[T.5]      -0.0509      0.217     -0.234      0.815      -0.477       0.375
    NCD[T.6]       0.1034      0.208      0.498      0.618      -0.303       0.510
    NCD[T.7]       0.1871      0.139      1.343      0.179      -0.086       0.460
    Age           -0.0581      0.004    -13.509      0.000      -0.067      -0.050
    VehAge        -0.0766      0.008     -9.730      0.000      -0.092      -0.061
    ==============================================================================
    

    

![png]({{"/assets/images/2020-05-17-glm-fig21_2.png"}})


***Binomial Model Coefficients and Logits Log-Odds, Odds and Probabilties***
- https://sebastiansauer.github.io/convert_logit2prob/


```python
print(FreqBinom.summary())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                  Claim   No. Observations:                45183
    Model:                            GLM   Df Residuals:                 45690.29
    Model Family:                Binomial   Df Model:                           21
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -3504.3
    Date:                Wed, 14 Oct 2020   Deviance:                       7008.7
    Time:                        21:43:23   Pearson chi2:                 4.72e+04
    No. Iterations:                    24                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -2.7911      0.289     -9.672      0.000      -3.357      -2.225
    Sex[T.M]       0.7379      0.153      4.832      0.000       0.439       1.037
    Geog[T.2]     -0.4605      0.141     -3.265      0.001      -0.737      -0.184
    Geog[T.3]     -0.6028      0.140     -4.320      0.000      -0.876      -0.329
    Geog[T.4]     -0.1851      0.111     -1.668      0.095      -0.403       0.032
    Geog[T.5]     -1.8854      0.514     -3.667      0.000      -2.893      -0.878
    Geog[T.6]     -2.0365      0.456     -4.469      0.000      -2.930      -1.143
    Geog[T.7]    -21.7246   1.56e+04     -0.001      0.999   -3.07e+04    3.06e+04
    EV[T.2]       -0.0613      0.261     -0.234      0.815      -0.574       0.451
    EV[T.3]        0.3441      0.198      1.734      0.083      -0.045       0.733
    EV[T.4]       -0.5492      0.226     -2.426      0.015      -0.993      -0.105
    EV[T.5]        0.2531      0.204      1.238      0.216      -0.147       0.654
    EV[T.6]        0.3888      0.207      1.882      0.060      -0.016       0.794
    EV[T.7]       -1.5831      1.025     -1.544      0.123      -3.593       0.427
    NCD[T.2]      -0.3731      0.198     -1.887      0.059      -0.761       0.015
    NCD[T.3]      -0.0929      0.203     -0.458      0.647      -0.490       0.305
    NCD[T.4]       0.0394      0.206      0.191      0.849      -0.365       0.443
    NCD[T.5]      -0.7865      0.295     -2.671      0.008      -1.364      -0.209
    NCD[T.6]      -0.6098      0.273     -2.230      0.026      -1.146      -0.074
    NCD[T.7]       1.0612      0.123      8.654      0.000       0.821       1.302
    Age           -0.0384      0.003    -11.086      0.000      -0.045      -0.032
    VehAge        -0.0705      0.006    -11.589      0.000      -0.082      -0.059
    ==============================================================================
    


```python
# Example conversions from logits to probabilities
const = FreqBinom.params[0]
odds = np.exp(const)
probability = odds / (1+odds)
print(f'Intercept: p = {probability:.3f}')
_ = np.exp(const+FreqBinom.params[1])/(1+(np.exp(const+FreqBinom.params[1])))
print(f'Intercept + Male: p = {_:.3f} ({_-probability:.3f})')
```

    Intercept: p = 0.058
    Intercept + Male: p = 0.114 (0.056)
    
  
    


![png]({{"/assets/images/2020-05-17-glm-fig25_1.png"}})


### Prediction
```python
test['Fnb'] = FreqNegBin.predict(transform=True,exog=test,offset=np.log(test['PYrs']))
test['Fpo'] = FreqPoisson.predict(transform=True,exog=test,offset=np.log(test['PYrs']))
test['Fbi'] = FreqBinom.predict(transform=True,exog=test)

fig,axs = plt.subplots(1,3,figsize=(13,3.3),sharex=True,sharey=True)
sns.histplot(test['Fpo'],ax=axs[0],label='Poisson')
sns.histplot(test['Fnb'],ax=axs[1],label='NegBinomial')
sns.histplot(test['Fbi'],ax=axs[2],label='Binomial')
test.sample(5)
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
      <th>Age</th>
      <th>Sex</th>
      <th>Geog</th>
      <th>EV</th>
      <th>VehAge</th>
      <th>NCD</th>
      <th>PYrs</th>
      <th>Claims</th>
      <th>Severity</th>
      <th>Claim</th>
      <th>SeverityAvg</th>
      <th>Fnb</th>
      <th>Fpo</th>
      <th>Fbi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>63584</td>
      <td>69.0</td>
      <td>M</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>0.361644</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.000694</td>
      <td>0.000690</td>
      <td>0.004815</td>
    </tr>
    <tr>
      <td>2446</td>
      <td>21.0</td>
      <td>M</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>0.767123</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.018421</td>
      <td>0.018198</td>
      <td>0.030839</td>
    </tr>
    <tr>
      <td>63505</td>
      <td>69.0</td>
      <td>M</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>7</td>
      <td>0.665753</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.003833</td>
      <td>0.003781</td>
      <td>0.011952</td>
    </tr>
    <tr>
      <td>7309</td>
      <td>25.0</td>
      <td>M</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>0.263014</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.015058</td>
      <td>0.014718</td>
      <td>0.017250</td>
    </tr>
    <tr>
      <td>29573</td>
      <td>42.0</td>
      <td>M</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>0.257534</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.003391</td>
      <td>0.003337</td>
      <td>0.008624</td>
    </tr>
  </tbody>
</table>
</div>




![png]({{"/assets/images/2020-05-17-glm-fig27_1.png"}})


Looking at the model summaries, the histograms and results the of predicted values on the test, we see that each model weights covariates similarly and produces similar scores on the test data. ***Again note*** that the $Binomial$ model was only used to demonstrate its similarity in this case, but this may not hold for other data.

### *Claim Severity*

```python
# including PYrs as parameter commented out in glm()
expr = "SeverityAvg ~ Age + Sex + Geog + EV + VehAge + NCD"

### Estimate severity using GLM-gamma with default inverse-power link
SevGamma = smf.glm(formula=expr,
              data=train_severity,
              family=sm.families.Gamma(link=sm.families.links.inverse_power())).fit()
```
 

Ignore the warning for now we will come back to that.

After fitting a GLM-Gamma, how do we find the Gamma shape (*a*) and scale (*b*) parameters of predictions for *Xi*?

"*Regression with the gamma model is going to use input variables Xi and coefficients to make a pre-diction about the mean of yi, but in actuality we are really focused on the scale parameter βi.  This is so because we assume that αi is the same for all observations, and so variation from case to case in μi=βiα is due simply to variation in βi.*" [technical overview of gamma glm](https://pj.freefaculty.org/guides/stat/Regression-GLM/Gamma/GammaGLM-01.pdf)

- [gamma handout](https://pj.freefaculty.org/guides/stat/Distributions/DistributionWriteups/Gamma/Gamma-02.pdf)
- [Gamma Choice of link function](http://people.stat.sfu.ca/~raltman/stat402/402L26.pdf)
- [exmaple finding scale in R](https://stat.ethz.ch/pipermail/r-help/2011-July/283736.html)
- [generalized linear model - Dispersion parameter for Gamma family - Cross Validated](https://stats.stackexchange.com/questions/247624/dispersion-parameter-for-gamma-family)
- [Pdummy_xyhon: Calculating scale/dispersion of Gamma GLM using statsmodels](https://stackoverflow.com/questions/60215085/calculating-scale-dispersion-of-gamma-glm-using-statsmodels)

[Alternatively you can infer gamma parameters from CI](https://math.stackexchange.com/questions/2873763/is-it-possible-to-determine-shape-and-scale-for-a-gamma-distribution-from-a-mean?newreg=d61b4517cd304ecca335b8e69220bf0c)
- [gamma.shape.glm: Estimate the Shape Parameter of the Gamma Distribution R MASS](https://rdrr.io/cran/MASS/man/gamma.shape.glm.html)
- [The identity link function does not respect the domain of the Gamma family? - Cross Validated](https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the-gamma-family)

Below I illustrate the range of predicted severity values for the intercept and each level in Geography.


```python
# shape is
shape = 1/SevGamma.scale
print(f'Shape: {shape:.2f}')

# intercept 
constant,intercept = SevGamma.params[0],SevGamma.scale/SevGamma.params[0]
print(f'Intercept: {intercept:.0f}')

# predicted mean G(Yi) is exp(Bo + Bi*Xi..)
geogs = [(i,SevGamma.scale/(constant+c)) for i,c in zip(SevGamma.params.index,SevGamma.params) if 'Geog' in i]

# plot
fig,axs = plt.subplots(1,6,sharex=True,sharey=True,figsize=(15,3))
for ax,x in zip(axs.flatten(),geogs):
    sns.histplot(np.random.gamma(shape=shape,scale=x[1],size=10000),stat="count",element="step",ax=ax,)
    ax.set_title(x[0])
    ax.set_xlim(0,14e4)
```

    Shape: 0.54
    Intercept: 52203
    


![png]({{"/assets/images/2020-05-17-glm-fig32_1.png"}})


The GLM-Gamma model gives us a prediction of the average severity of a claim should one occur.


```python
test_severity['Giv'] = SevGamma.predict(transform=True,exog=test_severity)
test_severity[:3]
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
      <th>Age</th>
      <th>Sex</th>
      <th>Geog</th>
      <th>EV</th>
      <th>VehAge</th>
      <th>NCD</th>
      <th>PYrs</th>
      <th>Claims</th>
      <th>Severity</th>
      <th>Claim</th>
      <th>SeverityAvg</th>
      <th>Giv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>35445</td>
      <td>45.0</td>
      <td>M</td>
      <td>4</td>
      <td>6</td>
      <td>2</td>
      <td>7</td>
      <td>4.920548</td>
      <td>1</td>
      <td>2480.0</td>
      <td>1</td>
      <td>2480.0</td>
      <td>28728.757724</td>
    </tr>
    <tr>
      <td>9653</td>
      <td>26.0</td>
      <td>M</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>0.589041</td>
      <td>1</td>
      <td>46000.0</td>
      <td>1</td>
      <td>46000.0</td>
      <td>21782.480267</td>
    </tr>
    <tr>
      <td>2039</td>
      <td>21.0</td>
      <td>M</td>
      <td>2</td>
      <td>2</td>
      <td>19</td>
      <td>1</td>
      <td>0.432877</td>
      <td>1</td>
      <td>11110.0</td>
      <td>1</td>
      <td>11110.0</td>
      <td>11537.649676</td>
    </tr>
  </tbody>
</table>
</div>



Now, remember the error we got using the inverse-power link function. The warning is fairly self explanatory "the inverse_power link function does not respect the domain of the Gamma family". Oddly this is a design feature of Statsmodels at the [time of writing](https://github.com/statsmodels/statsmodels/issues/3316#issuecomment-266453597). Rather it is better to use the log link function to ensure all predicted values are > 0.

***GLM-Gamma log link***

```python
# formula
expr = "SeverityAvg ~ Age + Sex + Geog + EV + VehAge + NCD"

### Estimate severity using GLM-gamma with default log link
SevGamma = smf.glm(formula=expr,
                   data=train_severity,
                   family=sm.families.Gamma(link=sm.families.links.log())).fit()
```


```python
# dispersion aka rate
dispersion = SevGamma.scale
print(f'Dispersion: {dispersion:.4f}')

# shape is 1/dispersion
shape = 1/dispersion
print(f'Shape: {shape:.4f}')

# intercept
constant,intercept = SevGamma.params[0],np.exp(SevGamma.params[0])
print(f'Intercept: {intercept:.2f}')

# predicted mean G(Yi) is exp(Bo + Bi*Xi..)
# tuple(name,Yi,scale)
geogs = [(i,
          np.exp(constant+c),
          np.exp(constant+c)*dispersion)
         for i,c in zip(SevGamma.params.index,SevGamma.params) if 'Geog' in i]

# plot
fig,axs = plt.subplots(1,6,sharex=True,sharey=True,figsize=(13,3))

for ax,x in zip(axs.flatten(),geogs):
    sns.kdeplot(np.random.gamma(shape=shape,scale=x[2],size=10000),shade=True,ax=ax,)
    ax.set_title(x[0])
```

    Dispersion: 2.0528
    Shape: 0.4872
    Intercept: 69330.81
    


![png]({{"/assets/images/2020-05-17-glm-fig37_1.png"}})


Statsmodels uses patsy design matrices behind the scenes. We can apply the design matrix to calculate the distribution parameters for both the frequency and severity models, and for any data set. Train, test, synthetic data and portfolios. You name it.

This is an incredibly powerfull approach as it enables the development of highly parameterised Monte Carlo and risk simulation models.I will walk the steps using pandas dataframes so that it is clear:


```python
# 1. Define a dummy model for your data and using the same formula and settings defined earlier.
dummy_ = smf.glm(formula=expr,data=test_severity,family=sm.families.Gamma(link=sm.families.links.log()))

# 2. we can then access the desing matrix. 
# Which is simply the data values, but also handling the intercept and reference categories.
a = pd.DataFrame(dummy_.data.exog,columns=dummy_.data.param_names)

# 3. Retrieve and transpose the trained model coefficients
b = pd.DataFrame(SevGamma.params).T

# 4. And multiply together
# but this only works if indexes are equal
c = a.multiply(b,axis=0)

# 5. It is much cleaner to use arrays. 
# et voila
c = pd.DataFrame(dummy_.data.exog * SevGamma.params.values,columns=dummy_.data.param_names)
```

We can then use the coefficients to estimate and plot the values and samples of each row, or the entire dataset.


```python
dispersion = SevGamma.scale
shape = 1/dispersion

num=5
fig,axs = plt.subplots(1,num,figsize=(15,4),sharex=True,sharey=True)

for i,pred,ax in zip(c.index[:num],SevGamma.predict(test_severity)[:num],axs.flatten()):
    scale = np.exp(c.loc[[i]].sum(axis=1))*dispersion
    sample = np.random.gamma(shape=shape,scale=scale,size=10000)
    sns.kdeplot(sample,label=i,ax=ax,shade=True)
    ax.set_title(f'Sample Mu {sample.mean():.0f}\nPredicted Mu {pred:.0f}',fontsize=12)
```


![png]({{"/assets/images/2020-05-17-glm-fig41_0.png"}})


# Putting it all together: Frequency-Severity model
    

***Portfolio price***

Given that this is motor insurance, lets assume a thin margin for %0.05


```python
# make a dummy portfolio and make predictions
portfolio = test.reset_index(drop=True).copy()
portfolio['annual_frq'] = FreqBinom.predict(portfolio)
portfolio['expected_sev'] = SevGamma.predict(portfolio)

# expected annual loss
portfolio['annual_exp_loss'] = portfolio['annual_frq'] * portfolio['expected_sev']

# set pricing ratio
tolerable_loss = 0.05
pricing_ratio = 1.0-tolerable_loss
portfolio['annual_tech_premium'] = portfolio['annual_exp_loss']/pricing_ratio
portfolio['result'] = portfolio['annual_tech_premium'] - portfolio['Severity']
portfolio.iloc[:3,-4:]
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
      <th>expected_sev</th>
      <th>annual_exp_loss</th>
      <th>annual_tech_premium</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>27818.169806</td>
      <td>277.316838</td>
      <td>291.912461</td>
      <td>291.912461</td>
    </tr>
    <tr>
      <td>1</td>
      <td>21753.203517</td>
      <td>300.218299</td>
      <td>316.019262</td>
      <td>316.019262</td>
    </tr>
    <tr>
      <td>2</td>
      <td>18524.970346</td>
      <td>298.920877</td>
      <td>314.653555</td>
      <td>314.653555</td>
    </tr>
  </tbody>
</table>
</div>



This summary illustrates the:
- observed Claims and Losses (Severity)
- the predicted frq, losses, and expected-losses
- the technical premium
- the profit loss result (calculated on full Severity rather than SeverityAvg)


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
      <th>Claim</th>
      <th>Severity</th>
      <th>annual_frq</th>
      <th>expected_sev</th>
      <th>annual_exp_loss</th>
      <th>annual_tech_premium</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Summary</td>
      <td>201.0</td>
      <td>5377478.0</td>
      <td>249.038867</td>
      <td>4.054346e+08</td>
      <td>5.764319e+06</td>
      <td>6.067705e+06</td>
      <td>690226.503245</td>
    </tr>
  </tbody>
</table>
</div>



Now we can build a Monte Carlo simulation to:
- provide a more robust estimate of our expected profit / loss
- sample the parameter space to gauge the variance and uncertainty in our model and assumptions
- calculate metrics for the AAL, AEP, and AXS loss and return


```python
# simulate portfolio
simulation = dict()
frq = list()
sev = list()
N=999

for it in range(N):
    print(f"{it}/{N}",end='\r')

    frq.append(np.random.binomial(n=1,p=portfolio['annual_frq']))
    sev.append(np.random.gamma(shape=portfolio['gamma_shape'],scale=portfolio['gamma_scale']))

# calculate Frq * Sev
frq_sev = np.array(frq)*np.array(sev)
# summarise the simulations
simulation['sim'] = pd.DataFrame({'iteration':range(N),
                                  'claim_num':np.array(frq).sum(axis=1),# num claims
                                  'loss_min':np.array(frq_sev).min(axis=1), # min claim
                                  'loss_sum':np.array(frq_sev).sum(axis=1), # total
                                  'loss_max':np.array(frq_sev).max(axis=1), # max claim
                                  'loss_avg':np.array(frq_sev).mean(axis=1)
                              })
```

    998/999

Calculate the Annual Exceedence Probability

- https://sciencing.com/calculate-exceedance-probability-5365868.html
- https://serc.carleton.edu/quantskills/methods/quantlit/RInt.html
- https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/floods-return-period-and-probability/


```python
def exceedance_prob(df,feature,ascending=False):
    data = df.copy()
    data['value'] = data[feature]
    data[f'Rank'] = data['value'].rank(ascending=ascending,method='dense')
    data[f'RI'] = (len(data)+1.0)/data[f'Rank']
    data[f'EP'] = 1.0/data[f'RI']
    data.sort_values([f'value'],ascending=ascending,inplace=True)
    data.reset_index(drop=True,inplace=True)
    return data

# profit loss based on technical premium
simulation['sim']['technical_premium'] = portfolio['annual_tech_premium'].sum()
simulation['sim']['result'] = (simulation['sim']['technical_premium'] - simulation['sim']['loss_sum'])

# recurrence intervals of each measure
simulation['loss_sum'] = exceedance_prob(simulation['sim'],'loss_sum')
simulation['loss_max'] = exceedance_prob(simulation['sim'],'loss_max')
simulation['result'] = exceedance_prob(simulation['sim'],'result',ascending=True)
```

This illustrates the profit and loss scenarios for our pricing strategy across $N$ iterations.

![png]({{"/assets/images/2020-05-17-glm-fig57_0.png"}})


And more intuitive EP curves.


![png]({{"/assets/images/2020-05-17-glm-fig59_0.png"}})