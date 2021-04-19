---
layout: post
comments: true
title:  "Bayesian Inference with PyMC3: pt 2 making predictions"
date:   2021-02-20 18:00:00 +0000
categories: [blog-post,data-analysis]
tags: [bayesian-inference,bayes-theory,pymc3,glm]
math: true
---

**[Jupyter notebook here](https://github.com/bpostance/dsc.learn/blob/main/statistics/bayesian-inference/03.1-bayesian-regression-oos-predictions.ipynb)**

In this post I will show how Bayesian inference is applied to train a model and make predictions on out-of-sample test data. For this, we will build two models using a case study of predicting student grades on a classical dataset. The first model is a classic frequentist normally distributed regression General Linear Model (GLM). While the second is, again A normal GLM, but built using the Bayesian inference method.

# Case Study: predicting student grades
The objective is to develop a model that can predict student grades given several input factors about each student. The publicly available [UCI dataset](https://archive.ics.uci.edu/ml/datasets/student+performance#) contains grades and factors for 649 students taking a Portuguese language course. #

***Dependent variable or "Target"***
- "G3" is the student's final grade for Portuguese (numeric: from 0 to 20, output target)

***Independent variables or "Features"***

A subset of numeric and categorical features is used to build the initial model:
- "age" student age from 15 to 22
- "internet"  student has internet access at home (binary: yes or no)
- "failures" is the number of past class failures (cat: n if 1<=n<3, else 4) 
- "higher" wants to take higher education (binary: yes or no) 
- "Medu" mother's education (cat: 0 - none, 1 - primary education (4th grade), 2 5th to 9th grade, 3 secondary education or 4 higher education)
-  ""Fedu father's education (cat: 0 - none, 1 - primary education (4th grade), 2 5th to 9th grade, 3 secondary education or 4 higher education)
-  "studytime" weekly study time (cat: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- "absences" number of school absences (numeric: from 0 to 93) 

The final dataset looks like this and here is the distribution of acheived student grades for Portuguese.

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
      <th>internet</th>
      <th>higher</th>
      <th>age</th>
      <th>absences</th>
      <th>failures</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>studytime</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.displot(data=df,x='G3',kde=True);
```
![png](/assets/images/2021-02-20-pymc3-predictions/output_6_1.png)
    

# GLM: frequentist model

First, I will build a regular GLM using ```statsmodels``` to illustrate the frequentist approach.

There are two binary categorical features for *Internet* usage and *Higher* education. *Age* is a numerical feature. The remaining features are numerical values that have been cut into ordinal categories. Practitioners may treat these differently depending on the model objective and nature of the feature. Here I will treat them as numerical within range ```0 to max(feature)``` range. In statsmodels we can use patsy design matrices and formula to specify how we want to treat each variable. For instance, for categoricals, we can use  ```C(Internet, Treatment(0))``` to encode Internet as a categorical variable with the reference level set to (0).


```python
# model grade data
formula = [f"{f}" if f not in categoricals else f"C({f})" for f in features]
formula = f'{target[0]} ~ ' + ' + '.join(formula)
glm_ = smf.glm(formula=formula,
              data=df,
              family=sm.families.Gaussian())
glm = glm_.fit()
glm.summary()
```

<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>G3</td>        <th>  No. Observations:  </th>  <td>   634</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   625</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Gaussian</td>     <th>  Df Model:          </th>  <td>     8</td> 
</tr>
<tr>
  <th>Link Function:</th>       <td>identity</td>     <th>  Scale:             </th> <td>  5.1895</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -1417.1</td>
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 12 Mar 2021</td> <th>  Deviance:          </th> <td>  3243.4</td>
</tr>
<tr>
  <th>Time:</th>                <td>13:39:15</td>     <th>  Pearson chi2:      </th> <td>3.24e+03</td>
</tr>
<tr>
  <th>No. Iterations:</th>          <td>3</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>    3.6741</td> <td>    1.430</td> <td>    2.570</td> <td> 0.010</td> <td>    0.872</td> <td>    6.476</td>
</tr>
<tr>
  <th>C(internet)[T.1]</th> <td>    0.2882</td> <td>    0.225</td> <td>    1.282</td> <td> 0.200</td> <td>   -0.152</td> <td>    0.729</td>
</tr>
<tr>
  <th>C(higher)[T.1]</th>   <td>    1.8413</td> <td>    0.328</td> <td>    5.622</td> <td> 0.000</td> <td>    1.199</td> <td>    2.483</td>
</tr>
<tr>
  <th>age</th>              <td>    0.3185</td> <td>    0.081</td> <td>    3.945</td> <td> 0.000</td> <td>    0.160</td> <td>    0.477</td>
</tr>
<tr>
  <th>absences</th>         <td>   -0.0739</td> <td>    0.020</td> <td>   -3.674</td> <td> 0.000</td> <td>   -0.113</td> <td>   -0.034</td>
</tr>
<tr>
  <th>failures</th>         <td>   -1.4217</td> <td>    0.170</td> <td>   -8.339</td> <td> 0.000</td> <td>   -1.756</td> <td>   -1.088</td>
</tr>
<tr>
  <th>Medu</th>             <td>    0.3847</td> <td>    0.108</td> <td>    3.567</td> <td> 0.000</td> <td>    0.173</td> <td>    0.596</td>
</tr>
<tr>
  <th>Fedu</th>             <td>    0.0332</td> <td>    0.108</td> <td>    0.306</td> <td> 0.760</td> <td>   -0.179</td> <td>    0.246</td>
</tr>
<tr>
  <th>studytime</th>        <td>    0.4307</td> <td>    0.112</td> <td>    3.839</td> <td> 0.000</td> <td>    0.211</td> <td>    0.651</td>
</tr>
</table>

<br>
Some quick observations on the above. Most features have a statistically significant linear relaitonship with grade with the exception of Fathers education. The sign of the regression coefficients also hold with our logic. More absensences and failures is shown to have a negative influence on predicted grade. Whereas studytime and desire to go on to higher education having positive influence on predicted grade.

Below we see there is an outlier in the data.

![png](/assets/images/2021-02-20-pymc3-predictions/output_14_0.png)
    


# PyMC3 GLM: Bayesian model

Now let's re-build our model using PyMC3. As described in this [blog post](https://twiecki.io/blog/2013/08/12/bayesian-glms-1/) PyMC3 has its own ```glm.from_formula()``` function that behaves similar to statsmodels. It even accepts the same patsy formula. 


```python
import pymc3 as pm
import arviz as avz

# note we can re-use our formula 
print(formula)

bglm = pm.Model()
with bglm:
    # Normally distributed priors
    family = pm.glm.families.Normal()
    # create the model 
    pm.GLM.from_formula(formula,data=df,family=family)
    # sample
    trace = pm.sample(1000,return_inferencedata=True)
```

    G3 ~ C(internet) + C(higher) + age + absences + failures + Medu + Fedu + studytime

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sd, studytime, Fedu, Medu, failures, absences, age, C(higher)[T.1], C(internet)[T.1], Intercept]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress 
  value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'>
  </progress>
  100.00% [8000/8000 00:27<00:00 Sampling 4 chains, 0 divergences]
</div>

<br>
Once the model has run we can examine the model posterior distribution samples. This is akin to viewing the ```model.summary()``` of a regular GLM as above. In this Bayesian model summary table the mean is the coefficient estimate from the posterior distribution. Here we see the posterior distribution of the model intercept is around 4.9. Indicating a student is expected to attain at least a grade of 4.9 irrespective of what we know about them. 


```python
summary = avz.summary(trace)
summary[:5]
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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>3.724</td>
      <td>1.421</td>
      <td>0.997</td>
      <td>6.260</td>
      <td>0.026</td>
      <td>0.019</td>
      <td>2972.0</td>
      <td>2599.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>C(internet)[T.1]</th>
      <td>0.287</td>
      <td>0.221</td>
      <td>-0.126</td>
      <td>0.706</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>4183.0</td>
      <td>2580.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>C(higher)[T.1]</th>
      <td>1.836</td>
      <td>0.332</td>
      <td>1.188</td>
      <td>2.450</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>4244.0</td>
      <td>2697.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.315</td>
      <td>0.080</td>
      <td>0.167</td>
      <td>0.463</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2950.0</td>
      <td>2281.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>absences</th>
      <td>-0.074</td>
      <td>0.021</td>
      <td>-0.114</td>
      <td>-0.035</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5348.0</td>
      <td>2799.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

<br>
Rather than p-values we have highest posterior density "hpd". The 3-97% hpd fields and value range indicates the credible interval for the true value of our parameter. As for classical models if this range crosses 0, from negative affect to positive affect, then perhaps the data signal is too weak to draw conclusions for this variable. This is the case for Internet usage - darnit Covid19. 

The posterior distributions can also be viewed as traceplots. 


```python
avz.plot_trace(trace)
```
 
![png](/assets/images/2021-02-20-pymc3-predictions/output_21_0.png)
    
## Interpret Variable affect on Predicted Grade

With the sampling complete. We can explore how each feature affects and contributes to predicted grade. I found this ```model_affect()``` function in this [Gist](https://gist.github.com/WillKoehrsen/fa59f7f28aefa09bc80138d3de8d6052#file-query_vars-py) for PyMC3.

Again, the number of absences has a negative affect on epected grade. Increasing age has a positive affect.

```python
X = pd.DataFrame(data=glm_.data.exog,columns=glm_.data.param_names) # dmatrix
model_affect('absences',trace,X)
model_affect('age',trace,X)
```

![png](/assets/images/2021-02-20-pymc3-predictions/output_24_0.png)
![png](/assets/images/2021-02-20-pymc3-predictions/output_24_1.png)
    

## Posterior Predictive Checks 
[Posterior predictive checks (PPC's)](https://docs.pymc.io/notebooks/posterior_predictive.html) are conducted to validate that the Bayesian model has captured the true distribution of the underlying data. Alternatively, PPC's are used to evaluate how the true data distribution compares to the distribution of data generates by the Bayesian model. . As per the PyMC3 documentation, PPC's are also a crucial part of the Bayesian modeling workflow. PPC's have two main benefits:

- *They allow you to check whether you are indeed incorporating scientific knowledge into your model – in short, they help you check how credible your assumptions before seeing the data are.*
- *They can help sampling considerably, especially for generalized linear models, where the outcome space and the parameter space diverge because of the link function.*


Below, PPC is used to sample the Y outcome of our model 200 times (left). Similar to the above trace plots, PPC can also provide a sampling of model parameters such as Age (right).

![png](/assets/images/2021-02-20-pymc3-predictions/output_26_5.png)
    

## Predictions on Out-Of-Sample data

Now that the model is trained and fitted on the data and we have inspected variable affects we can use the model to make predictions on out-of-sample observations and test cases. In contrast to other modeling approaches and packages, such as statsmodels and scikit-learn, it is not as straight forward as simply calling ```model.predict(Xdata)```. In PyMC3 I have discovered several strategies for applying models to out-of-sample data. 


### method 1: the mean coefficient model
We can use the MCMC trace to obtain a sample mean of each model coefficient and apply this to reconstruct a typical GLM formula. Remember we used statsmodels-patsy formulation to encode our categorical variables, well we can again use patsy to construct a helper. 

The benefit to this approach is its ease and simplicity. The downside is that we are now omitting and missing out on a chunk of that MCMC sampling for the confidence and uncertainty in our data that we obtained by taking a Bayes approach in the first place. 


```python
# mean model coefficients
mean_model = summary[['mean']].T.reset_index(drop=True)

# create a  design matrix of exog data values
# same as for GLM's
X = pd.DataFrame(data=glm_.data.exog,columns=glm_.data.param_names)

# add columns for the standard deviations output from the bayesian fit
for x in mean_model.columns:
    if x not in X.columns:
        X[x] = 1
        
# multiply and work out mu predictions
coefs = X * mean_model.values
pred_mu = coefs.iloc[:,:-1].sum(axis=1)[:]
pred_sd = coefs['sd']
print('Predictions:')
n = 5
for m,s in zip(pred_mu[:n],pred_sd[:n]): print(f"\tMu:{m:.2f} Sd:{s:.2f}")
```

    Predictions:
    	Mu:13.47 Sd:2.29
    	Mu:12.34 Sd:2.29
    	Mu:11.41 Sd:2.29
    	Mu:13.48 Sd:2.29
    	Mu:12.72 Sd:2.29


> **_NOTE:_**: There is an issue  using Jupyter, pymc3.GLM and Theano. At the time of writing this is where things with ```pm.GLM.from_formula()``` start to break down using Jupyter. The following two methods are recommended in the PyMC3 documentation. However, both generate the following Theano error when used in conjunction with ```GLM.from_formula()``` on Jupyter. 
> ```ERROR! Session/line number was not unique in database. History logging moved to new session 11``` 
> This seems to be an issue with the way GLM.from_formula() uses patsy and interacts with Theano in Jupyter Notebooks.<br>
> [_source code_](https://github.com/pymc-devs/pymc3/blob/master/pymc3/glm/linear.py#L101)<br>
> [_question on SO_](https://stackoverflow.com/questions/50369957/dependency-between-session-line-number-was-not-unique-in-database-error-and-p)<br>
> I have not tested running either of the following methods using a .py script but it seems reasonable that the following methods would work outside Jupyter.


### method 2: using Theano shared variable

Given the above, for now we will lose ```GLM.from_formula()``` and reconstruct the model in standard PyMC3 form and using both: patsy to generate a design matrix, and Theano to create a shared X variable. To keep things short I have simplified my Betas using ```shape(n)```. This will degrade model tuning performance as all priors are set at the uniform initial value and it may lead to some zero errors [e.g see here](https://discourse.pymc.io/t/mass-matrix-contains-zeros-on-the-diagonal/4981). In practice, you should set the Beta's individually using informative priors. 


```python
import patsy
from theano import shared

design = patsy.dmatrices(formula_like=formula,data=df,return_type='dataframe')
# design[1].design_info

train,test = df[:500],df[500:]
trainx = patsy.build_design_matrices([design[1].design_info],train,return_type="dataframe")[0]
testx = patsy.build_design_matrices([design[1].design_info],test,return_type="dataframe")[0]

# Shared theano variable for modeling
# must be np.array()
modelx = shared(np.array(trainx))

bglm = pm.Model()
with bglm:
    alpha = pm.Normal('alpha', mu=4, sd=10)
    betas = pm.Normal('beta', mu=1, sd=6, shape=8)
    sigma = pm.HalfNormal('sigma', sd=1)a)
    mu = alpha + pm.math.dot(betas, modelx.T)
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=trainy)  
    trace = pm.sample(1000,init="adapt_diag",return_inferencedata=True)
```

<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:20<00:00 Sampling 4 chains, 0 divergences]
</div>

<br>
Now we can update our shared X variable with the test set and use the model to make predictions. The prediction here really means sampling the posterior distributions of each coefficient on the test set observations.

We can specify the number of sampling rounds to perform and visualise individual samples and aggregates of samples.


```python
samples = 50

# Update model X and make Prediction
modelx.set_value(np.array(testx)) # update X data
ppc = pm.sample_posterior_predictive(trace, model=bglm, samples=samples,random_seed=6)
print(ppc['y'].shape)

n = 5
plt.figure(figsize=(15,3))
plt.title('Observed & Predicted Grades')
plt.plot(test.reset_index()['G3'],'k-',label='Y observed')
plt.plot(ppc['y'][0,:],lw=1,linestyle=':',c='grey',label='Y 1st trace')
plt.plot(ppc['y'][:n,:].mean(axis=0),lw=1,linestyle='--',c='grey',label=f'Y trace [0:{n}]th mean')
plt.legend(bbox_to_anchor=(1,1));
```

<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='50' class='' max='50' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [50/50 00:00<00:00]
</div>

![png](/assets/images/2021-02-20-pymc3-predictions/output_34_3.png)
    
### method 3: shared X variable

This method is very similar above but instead using the ```pm.Data()``` to hold our X data in train and test rounds. Functionally this is cleaner as we don't need to import and use Theano ```shared()``` explicitly.


```python
bglm = pm.Model()
with bglm:
    alpha = pm.Normal('alpha', mu=4, sd=10)
    betas = pm.Normal('beta', mu=1, sd=6, shape=8)
    sigma = pm.HalfNormal('sigma', sd=1)
    xdata = pm.Data("pred", trainx.T)
    mu = alpha + pm.math.dot(betas, xdata)
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=trainy)
    trace = pm.sample(1000,init="adapt_diag",return_inferencedata=True)

# Update X values and predict outcomes and probabilities
with bglm:
    pm.set_data({"pred": testx.T})
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y"], samples=600)
    model_preds = posterior_predictive["y"]
```

<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:20<00:00 Sampling 4 chains, 0 divergences]
</div>

<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='600' class='' max='600' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [600/600 00:02<00:00]
</div>

<br>
Again, let's increase the number of samples visualise the:

- posterior distribution of predictions for each observation in the OOS test data (left)
- and the Observed, Mean, Credible-Interval or "Highest Density Interval" using arviz ```hdi()```. 


```python
from matplotlib.gridspec import GridSpec

fig=plt.figure(figsize=(15,5))
gs=GridSpec(nrows=1,ncols=2,width_ratios=[1,2]) # 2 rows, 3 columns

ax0 = fig.add_subplot(gs[0])
ax0.set_title("Yi kde")
sns.kdeplot(data=pd.DataFrame(model_preds[:,:5]),ax=ax0)

ax1 = fig.add_subplot(gs[1])
ax1.set_title("test predictions")
ax1.plot(test.reset_index(drop=True)['G3'],'k-',lw=1,label='obs')
ax1.plot(model_preds.mean(0),c='orange',label='mean pred')

alpha = 1-0.5
ax1.plot(avz.hdi(model_preds,alpha)[:,0],ls='--',lw=1,c='red',label=f'CI lower {alpha}')
ax1.plot(avz.hdi(model_preds,alpha)[:,1],ls='--',lw=1,c='red',label=f'CI upper {alpha}')

ax1.legend(bbox_to_anchor=(1,1));
```

![png](/assets/images/2021-02-20-pymc3-predictions/output_38_1.png)
    

# Conclusion

This post demonstrates how to develop a Bayesian inference General Linear Model. A case study for modeling student grades was used to demonstrate a classical frequentist approach in statsmodels and with a Bayes's approach in PyMC3 with several implementations on predicting out of sample data. 

# References

- [Getting started with GLM in PyMC3](https://docs.pymc.io/notebooks/getting_started.html?highlight=glm)
- [All GLM examples in PyMC3](https://docs.pymc.io/notebooks/GLM.html)
- [Robust GLM's in PyMC3](https://docs.pymc.io/notebooks/GLM-robust.html)
- [PyMC3 OLS Regression](https://docs.pymc.io/notebooks/GLM-linear.html)
- [PyMC3 Logistic Regression](https://docs.pymc.io/notebooks/GLM-logistic.html)
- [PyMC3 Bayesian Linear Regression prediction with sklearn.datasets](https://stackoverflow.com/questions/37312817/pymc3-bayesian-linear-regression-prediction-with-sklearn-datasets)
- [Bayes affect plot](https://gist.github.com/WillKoehrsen/fa59f7f28aefa09bc80138d3de8d6052#file-query_vars-py)