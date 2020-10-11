---
layout: post
title:  "A deep dive on GLM's in frequency severity models"
date:   2020-05-17 18:00:00 +0000
comments: true
categories: [tutorial]
tags: [glm,insurance,monte carlo]
---
<img src="/assets/images/2020-05-17-glm-fig1.png" alt="drawing" width="800" height="350"/>



This notebook is a deep dive into [General Linear Models (GLM's)](https://online.stat.psu.edu/stat504/node/216/) with a focus on the GLM's used in insurance risk modeling and pricing (Yan, J. 2010).I have used GLM's before including: a Logistic Regression for landslide geo-hazards (Postance, 2017), for modeling extreme rainfall and developing catastrophe models (Postance, 2017). The motivation for this post is to develop a deeper knowledge of the assumptions and application of the models and methods used by Insurance Actuaries, and to better understand how these compare to machine learning methods.

I am currently updating this post, please see the [source on Github](https://github.com/bpostance/training.data_science/blob/master/Risk/02-GLM-FrequencySeverity.ipynb) in the meantime.