---
layout: post
title:  "Portfolio Optimisation: Monte Carlo method"
date:   2020-08-04 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [optimisation,monte-carlo]
---

*Given a fixed amount of avliable resources, optimise allocation to maximise returns across a set of products with variable returns.*

<img src="https://cdn-images-1.medium.com/max/1000/1*QBQXfHZxYzAsbZGqAxa9nw.jpeg" width="500" height="300" />

Following the Modern Portfolio Theory model [(Markowitz 1957)](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.3.3.255) we can formulate this problem as:

*Given a fixed quantity of money (say $1000), how much should we invest in a series of stocks so as to (a) have a one month expected return of at least a given threshold, and (b) minimize the risk (variance) of the portfolio return.*

## Solution using Monte Carlo
Monte Carlo (MC) based solutions encompass a wide array of algorithms that exploit repeat random sampling and uncertainty to solve large, complex and generally intractable mathematical problems. MC is akin to exhaustive search type solutions. However in MC framed problems, the input model parameters, initial, boundary and environmental conditions are unknown and or subject to a degree of uncertainty. Therefore a true exahuastive search is not possible. The aim of MC is to conduct repeat random sampling of the input space to generate large numbers (e.g. $n$ = 100,000) of "plausible realities" from which metrics, risk analyses, and further assessments are drawn. Thus one of the most challenging aspects of the monte carlo method is in determining accuracte covariances and probability distributions of the input parameter space.

Lets make some simplifying assumptions:
 - We have \\$1000 to invest.
 - The risk-free rate is 0.03 (3 %). This is the return we can garuntee by instead putting our money in a savings account.
 - Stocks have a known and fixed starting price. 
 - The monthly returns of a stock follow a standard normal distribution.

***Stocks***
First let's define some stocks that are avaliable to invest in. For simplicity the stocks are heuristically assignined with a range of average daily return $mu$ and volatility $sigma$ values. For a more realistic simulation, one could derive these values from actual investment instruments. The average return and volatility of each stock is summarised in the table and figure 1.

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
      <th>n</th>
      <th>name</th>
      <th>price</th>
      <th>mu</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>jumanji inc</td>
      <td>10</td>
      <td>0.10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>evolution ltd</td>
      <td>20</td>
      <td>0.20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>the incredibles inc</td>
      <td>30</td>
      <td>0.30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>men in black &amp; co</td>
      <td>20</td>
      <td>0.08</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>goldmember gmbh</td>
      <td>50</td>
      <td>0.05</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>dantes peak</td>
      <td>10</td>
      <td>0.25</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>deep impact</td>
      <td>20</td>
      <td>0.02</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

![png]({{ "/assets/images/2020-08-04-portfolios-fig1.png" }})

***Monte Carlo Portfolio's***

The second step is to generate random portfolios of value $V$. Method is adapted from this [SO](https://stackoverflow.com/a/36818198/4538066) answer, but is not optimal. 


```python
# total number of stocks in universe
N = len(stocks) 

# V = total to invest 
investment = 10**3 
print(f'Investment: ${investment:.2f}')
      
mc_portfolio_runs = list()
pmax=5000
for p in range(pmax):
    print(f'{p}/{pmax}',end='\r')
    
    # random portfolio size
    portfolio_size = np.random.randint(2, N+1) 

    # create a df portfolio of them
    df = stocks_df.iloc[np.random.choice(N, portfolio_size, replace=False)].copy()

    # sum numbers
    while True:
        df['value'] = np.random.random(portfolio_size)
        T = df['value'].sum()
        if T != 0: break

    # calculate normalised value and number of shares 
    df['value'] *= investment/T
    df['shares'] = df['value']/df['price']
    df['p'] = p
    
    mc_portfolio_runs.append(df)
```

    Investment: $1000.00
    4999/5000


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
      <th>n</th>
      <th>name</th>
      <th>price</th>
      <th>mu</th>
      <th>sigma</th>
      <th>value</th>
      <th>shares</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>men in black &amp; co</td>
      <td>20</td>
      <td>0.08</td>
      <td>40</td>
      <td>201.353368</td>
      <td>10.067668</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>deep impact</td>
      <td>20</td>
      <td>0.02</td>
      <td>5</td>
      <td>19.906906</td>
      <td>0.995345</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



***Balanced Portfolio***
We also create A balanced portfolio. This provides a usefull benchamrk and represents the strategy of investing in each stock equally.

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
      <th>n</th>
      <th>name</th>
      <th>price</th>
      <th>mu</th>
      <th>sigma</th>
      <th>value</th>
      <th>shares</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>jumanji inc</td>
      <td>10</td>
      <td>0.10</td>
      <td>10</td>
      <td>333.33</td>
      <td>33.3330</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>evolution ltd</td>
      <td>20</td>
      <td>0.20</td>
      <td>20</td>
      <td>333.33</td>
      <td>16.6665</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>the incredibles inc</td>
      <td>30</td>
      <td>0.30</td>
      <td>30</td>
      <td>333.33</td>
      <td>11.1110</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>men in black &amp; co</td>
      <td>20</td>
      <td>0.08</td>
      <td>40</td>
      <td>333.33</td>
      <td>16.6665</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>goldmember gmbh</td>
      <td>50</td>
      <td>0.05</td>
      <td>15</td>
      <td>333.33</td>
      <td>6.6666</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>dantes peak</td>
      <td>10</td>
      <td>0.25</td>
      <td>20</td>
      <td>333.33</td>
      <td>33.3330</td>
      <td>balanced</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>deep impact</td>
      <td>20</td>
      <td>0.02</td>
      <td>5</td>
      <td>333.33</td>
      <td>16.6665</td>
      <td>balanced</td>
    </tr>
  </tbody>
</table>
</div>



***Calculate Portfolio Risk & Return***
Third, we define and execute a function to calculate the return, risk, and Sharpe ratio of each random portfolio and the balance portfolio. The [Sharpe ratio](https://www.investopedia.com/terms/s/sharperatio.asp) is defined as the return earned in excess of the risk free rate (e.g. fixed rate saving account).

```python
def portfolio_returns(portfolio,risk_free_rate=0.03):
    """
    Calculates portfolio returns, risk and summary
    """
    #calculate returns
    portfolio['result'] = np.random.normal(loc=portfolio['mu'],scale=portfolio['sigma'])*portfolio['shares']

    # calculate risk
    # https://math.stackexchange.com/questions/3381762/how-to-combine-standard-deviations-using-variance
    # sqrt((sum(var1^2+var2^2)))
    portfolio['var'] = portfolio['sigma'].apply(lambda x: x**2)
    
    # portfolio summary
    portfolio_summary = portfolio.groupby('p').agg({'value':sum,'var':sum,'result':sum})
    portfolio_summary['return']= ((portfolio_summary['value']+portfolio_summary['result'])/portfolio_summary['value'])*100.0
    portfolio_summary['risk'] = portfolio_summary['var'].apply(lambda x: np.sqrt(x)/100.)
    
    portfolio_summary['sharpe']=(portfolio_summary['return']-risk_free_rate)/portfolio_summary['risk']
    portfolio_summary.drop(labels=['var'],axis=1,inplace=True)

    return portfolio_summary,portfolio
```


```python
summary = dict()
summary['MonteCarlo'],mc = portfolio_returns(mc_portfolios)
summary['Balanced'],_ = portfolio_returns(balanced_portfolio)
```

***Results***
Figure 2 below illustrates the results and relationship between each randomised portfolios (grey dots) risk and investment return (%). The orange dot is the balance portfolio. 

![png]({{ "/assets/images/2020-08-04-portfolios-fig1.png" }})

The green dot is the portfolio with the greatest Sharpe ratio, that is the portfolio with lowest risk in relation to the level of return in excess of the risk free rate. We see that this portfolio has a blend of investments in "Deep Impact" 9%  and "Jumanji Inc" 81 %.

```python
# Max Sharpe portfolio 
mc.loc[mc['p']==summary['MonteCarlo'].sharpe.idxmax()]
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
      <th>n</th>
      <th>name</th>
      <th>price</th>
      <th>mu</th>
      <th>sigma</th>
      <th>value</th>
      <th>shares</th>
      <th>p</th>
      <th>result</th>
      <th>var</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9164</th>
      <td>6</td>
      <td>deep impact</td>
      <td>20</td>
      <td>0.02</td>
      <td>5</td>
      <td>187.602982</td>
      <td>9.380149</td>
      <td>2037</td>
      <td>-18.592925</td>
      <td>25</td>
    </tr>
    <tr>
      <th>9165</th>
      <td>0</td>
      <td>jumanji inc</td>
      <td>10</td>
      <td>0.10</td>
      <td>10</td>
      <td>812.397018</td>
      <td>81.239702</td>
      <td>2037</td>
      <td>1747.558702</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



***Conclusion***
- This method applies monte carlo (i.e. exhaustive search) to calculate a large number of randomised investment portfolios. 
- Risk, Return, and Sharpe measures are calculated for each of the random portfolios, and for a balanced portfolio (i.e. equal allocation portfolio assuming no knowledge of where to invest). 
- An optimal portfolio is identified using the maximum Sharpe ratio that maximises returns whilst minimising risk.
- Increasing the number of randomised portfolios increases the chances of identifying an optimal portfolio. 
- However, finding the optimal portfolio is not garunteed. The likelihood of identifying the most optimal portfolio decreases with increasing number of investment options, and higher degrees of uncertainty of the investment returns and variances.

***References:***

- https://towardsdatascience.com/optimization-with-python-how-to-make-the-most-amount-of-money-with-the-least-amount-of-risk-1ebebf5b2f29
- https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb
- https://insightr.wordpress.com/2017/08/27/pricing-optimization-how-to-find-the-price-that-maximizes-your-profit/
- http://riskdatascience.net/usecases-auswahl/automated-portfolio-optimization-by-combining-ai-and-risk-methods/
- https://towardsdatascience.com/automating-portfolio-optimization-using-python-9f344b9380b9
- https://towardsdatascience.com/best-investment-portfolio-via-monte-carlo-simulation-in-python-53286f3fe93
- https://www.dummies.com/business/accounting/auditing/how-to-optimize-portfolio-risk/
