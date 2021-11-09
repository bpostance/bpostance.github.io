---
layout: post
title:  "How businesses can use AI effectively and cost-efficiently"
date:   2021-09-21 18:00:00 +0000
categories: [blog-post,thought-piece]
tags: [business-strategy,data-driven,enablers]
math: false
comments: true
---

There is huge potential for AI to transform business. By 2030 AI is projected to deliver an additional $ 13 trillion of global economic activity. In the UK, thats an  increase in annual GDP of $ 310 billion (11%) in just 10 years [[1,2]](#references). These lofty economic projections are based on core assumptions for growth by business strategy transformation and adoption of AI. 

>The National AI Strategy builds on the UK’s strengths but also represents the start of a step-change for AI in the UK, recognising the power of AI to increase resilience, productivity, growth and innovation across the private and public sectors. <br>*[National AI Strategy, 2021. UK Government](https://www.gov.uk/government/publications/national-ai-strategy)*

We know the likes of FAANG (Facebook "Meta", Apple, Amazon, Google) and other BigTech companies (Tik Tok, Snap, Baidu, Huawei) are utilising data and AI at scale to drive revenue and make eye watering profits. Yet despite the proliferation of AI enablers such as data availability, open source technology, cloud computing, the rise data engineering and data scientist teams, amongst others, **AI is yet to take hold in most industries**. So what's going on? 

## Why do initial AI initiatives fail?
Now, I'm not saying that companies are not trying to do AI, most are. These efforts fall broadly into 1 of 2 camps:

1. You might have hired some shiny new AI talent such as a Data Scientist or Machine Learning engineer to augment an existing data team, or to work independently on a key project or business line. Whilst some interesting results and insights are generated. The business, the existing teams, and the new hires expectations are not met and there is a high degree of frustration all around. Generally the new hires are given a loosely defined problem statement and in working through the problem many issues are raised, including: poor and fragmented data access; low data availability and volume; varying  degrees of data capture; no suitable environment and tooling exists to wrangle and model the data - a quick fix is used; the data that does exists has poor coverage and quality; business time (and money) is spent cleaning and fixing the data; eventually a highly tuned and optimised model is generated - yay; the business doesn't understand or know what to do with the model and its output; more business time and resources are spent understanding the problem, refining the data, refining the model, and communicating its potential; if you get this far; the business now wants to use the model in its existing workflow tools and systems; the new dataset and model are totally undocumented and even if you could replicate it, there is no way to integrate and deploy the model within the existing architecture and technology estate; oh bugger!

2. You have identified a quick win and relatively low entry way to create business value with AI. It doesn't come cheap, but the executive team and senior management are happy to provide the funds as the company needs to be seen to be investing and taking AI seriously. This might be in the form of an expensive management consultancy or service partner that will run a AI pilot with you, and all you need to do is share your data...here come all the issues again from point 1 again and if (IF!) you can burn enough time and money to get this far then there is a whole host of more severe issues to come that I wont go into now (buy me a coffee and I'll tell you). Alternatively, you might have invested in a new piece of analytics software or PaaS that enables you and your existing teams to build predictive analytics and machine learning models. There is an initial buzz about the company and some genuine insights are created, as time passes adoption is concentrated to a few isolated power users and use cases within the business, in time usage tapers, eventually you get a call from your manager or from finance asking for ROI numbers for this expensive piece of kit, a cheaper alternative or point replacement solution is found, or the initiative dies completely. AI is widely regarded as "something that [insert individual or team-X] over there does" or worse as an expensive gimmick. Ultimately, the promised land of stratospheric AI gains did not materialise across the company.

If you any of this relates then please do not feel disheartened. It's bloody hard to get this right. The good news is that you've started - the company is aware to some degree. Also remember that even with the help of a seasoned management consultancy, or a Chief Data Officer, and large financial backing, there are going to be some hurdles and failures along on the way - its business after all. **Even the big boys who solved all of the initial AI hurdles fail from time to time**, and spectacularly at that. Here are [five infamous AI fails](https://thinkml.ai/five-biggest-failures-of-ai-projects-reason-to-fail/) that cost companies $ 100's of millions. As I am writing this, I have a tab open to ["How Zillow's Grand Home-Buying Ambitions Imploded"](https://www.bloomberg.com/news/newsletters/2021-11-05/how-zillow-s-grand-home-buying-ambitions-imploded) and news they are pullout of the US property purchasing market. It turns out their AI purchasing model undervalued properties to the tune of $ 500 million. Ouch!

The bottleneck for most AI applications is: 1) understanding and choosing the right business problems to solve, and 2) getting the right data to feed to the AI application. For most companies AI initiatives tend to originate fro the bottom-up, from individuals and small pockets within the business. It rarely starts from the top, unless perhaps you've had a change in senior leadership. As a result there lies the initial challenge of aligning enough key people and resources to make something actually work end to end. So here goes.

# How to do Business & Data-centric AI development 

- Understand The Business Problem: 
   - Understand and frame the problem. Give it some time. Understand it and frame it again. What are the key performance indicators KPI's?
   - What is the potential size of the prize? Is it a tangible outcome or is the size of the prize large enough that simply knowing more about it would generate some business value? Beware the latter - revert to previous step and refine.
   - What pain point are you trying to remove and how does AI solve this problem? What is the AI solution at a high level? Develop, communicate and refine a conceptual model of the present day and the future reality. Keep it narrow and on point.
   - Who is your key customer(s)? Who are your key blockers and enablers?
   - What are the risks of failure? Are there any existing controls and limitations in place that you (and your new AI) have to work within?
   - Get feedback and involve as many people as you can - they might have an idea you haven't thought of. Until you find something that is worth pursuing, doing the above will create a catalogue of possible AI initiatives across the business.
   - Set and communicate a clear aim and timeline with the senior leadership team. Reinforce and uphold them regularly. It's better to under promise and over deliver. Your challenge here is to manage expectations and dampen "AI Hype", whilst keeping your sponsors engaged - a clear business value proposition makes life much easier.
  
- People & Teams
  - Are you the business expert or the AI expert? You will need both a subject matter experts who deeply understand the business problem and context, and someone with AI expertise.
  - Identify employees who know the data very well. You will need their time. Chances are they will want to be involved but will be busy on BAU - how will you overcome or manage this? 
  - As you assemble your team, make clear the purpose of the project and set clear expectations for their involvement.

- Data
  - What data do you need? Use your teams knowledge and expertise to design your ideal dataset and list the most relevant features. 
  - Perform a data fitness test. Is the data available, is there enough of it (volume), what is the data quality and coverage like? Do you need to supplement internal data with open or third-party data?
  - Do you need to develop existing or capture new data? Your project may have to first focus on developing the data sets you need to develop and feed your AI. Is this a delay factored in and tolerable?
  - Do you have the data engineering capability to capture, manage and maintain the data you need from the development phase to the production phase?

- AI
  - Understand what AI methodologies and tooling you need. How will you develop and deliver the solution. To start, this could simply be an open source model package, operating on a daily or weekly dataset, delivered in an existing report or dashboard. Keep costs low.
  - What are the key model performance metrics? How do these related to the business KPI's?
  - AI models constantly evolve. Make building and using AI systematic and repeatable. Document your steps, adopt and re-use your companies existing software development principles to ensure that the AI code base is clean and organised (e.g. GitHub, CI/CD, Templates).
  - Once you have developed your initial models. Focus relentlessly on communicating how it works to the business stakeholders. Collectively explore and understand the models strengths and weaknesses. This will take a long time and there will be many re-visions to the data and the model along the way, your project will fail if you skimp here. 





# References
1. [NOTES FROM THE AI FRONTIER MODELING THE IMPACT OF AI ON THE WORLD ECONOMY, 2018. McKinsey Global Institute](https://www.mckinsey.com/~/media/McKinsey/Featured%20Insights/Artificial%20Intelligence/Notes%20from%20the%20frontier%20Modeling%20the%20impact%20of%20AI%20on%20the%20world%20economy/MGI-Notes-from-the-AI-frontier-Modeling-the-impact-of-AI-on-the-world-economy-September-2018.ashx)
2. [National AI Strategy, 2021. Office for Artificial Intelligence, Department for Digital, Culture, Media & Sport, and Department for Business, Energy & Industrial Strategy](https://www.gov.uk/government/publications/national-ai-strategy)
3. [The economic impact of artificial intelligence on the UK economy, 2017. Price Waterhouse Coopers](https://www.pwc.co.uk/economic-services/assets/ai-uk-report-v2.pdf)
4. [five infamous AI fails](https://thinkml.ai/five-biggest-failures-of-ai-projects-reason-to-fail/)
5.  ["How Zillow's Grand Home-Buying Ambitions Imploded"](https://www.bloomberg.com/news/newsletters/2021-11-05/how-zillow-s-grand-home-buying-ambitions-imploded)