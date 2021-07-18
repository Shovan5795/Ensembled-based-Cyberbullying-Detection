# Ensembled-based-Cyberbullying-Detection
Paper Title: Cyberbullying Detection: An Ensemble Based Machine Learning Approach

Please cite this paper if you want to use the codes:

K. S. Alam, S. Bhowmik and P. R. K. Prosun, "Cyberbullying Detection: An Ensemble Based Machine Learning Approach," 2021 Third International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV), 2021, pp. 710-715, doi: 10.1109/ICICV50876.2021.9388499.


Research on cyberbullying detection is gaining increasing attention in recent years as both individual victims and societies are greatly affected by it. Moreover, ease of access to
social media platforms such as Facebook, Instagram, Twitter, etc. has led to an exponential increase in the mistreatment of people in the form of hateful messages, bullying, sexism, racism,
aggressive content, harassment, toxic comment etc. Thus there is an extensive need to identify, control and reduce the bullying contents spread over social media sites, which has motivated
us to conduct this research to automate the detection process of offensive language or cyberbullying. Our main aim is to build single and double ensemble-based voting model to classify
the contents into two groups: ‘offensive’ or ‘non-offensive’. For this purpose, we have chosen four machine learning classifiers and three ensemble models with two different feature extraction
techniques combined with various n-gram analysis on a dataset extracted from Twitter. In our work, Logistic Regression and Bagging ensemble model classifier have performed individually
best in detecting cyberbullying which has been outperformed by our proposed SLE and DLE voting classifiers. Our proposed SLE and DLE models yield the best performance of 96% when TFIDF (Unigram) feature extraction is applied with K-Fold crossvalidation.

OVERALL PROCESS:
![Overall Process](https://user-images.githubusercontent.com/77354495/126079109-a28323d8-21f3-48b3-90c5-bcaf92b27222.png)

PREPROCESS STEPS:
![Preprocess](https://user-images.githubusercontent.com/77354495/126079129-16d70536-154a-42ce-b878-8141e75ec0bc.png)

VOTING ARCHITECTURE:
![SLE and DLE](https://user-images.githubusercontent.com/77354495/126079138-a0bb78c1-64be-42ea-ac9c-f63174aacd3f.png)

RESULT:

Comparative Analysis
![image](https://user-images.githubusercontent.com/77354495/126079154-7e589e97-6928-4f3b-a6f1-efa0dcff32d5.png)

Cross-Validation Result
![image](https://user-images.githubusercontent.com/77354495/126079174-9b2b5ebc-7673-4997-b193-df7e7af16089.png)
