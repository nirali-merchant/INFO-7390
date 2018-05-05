# INFO-7390

Assignment 1


Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the TAs can run the notebooks.

Find a public dataset with a least 9 columns and 3000 rows. You must post your dataset on the class piazza and get an OK before using it as every student needs to use a different dataset.

Individual Assignments
These are individual assignments. They cannot be done in groups.

Part A Cleaning and EDA (75 points)
Data cleaning

Are there missing values?
Are there inappropraite values?
Remove or impute any bad data.
Answer the following questions for the data in each column:

How is the data distributed?
What are the summary statistics?
Are there anomalies/outliers?
Plot each colmun as appropriate for the data type:

Write a summary of what the plot tells you.
Are any of the columns correlated?

Part B Writing a web scraper (25 points)
Find a public website. You must post your website domain (e.g. amazon.com) on the class piazza and get an OK before using it as every student needs to use a different website.

Website
Collect all of the external links (there must be some on the page of your )
Associate the link with a textual description of it from the website.
Write a function to check whether the link is valid.
Save the external links(urls), textual description, a boolean for valid, and the last vaild datetime check to an excel file.
List of datasets for machine learning research
List of datasets for machine learning research
UC Irvine Machine Learning Repository
Public Data Sets : Amazon Web Services
freebase
Google Public Data Explorer
datahub
data.gov
Data cleaning checklist
Save original data
Identify missing data
Identify placeholder data (e.g. 0's for NA's)
Identify outliers
Check for overall plausibility and errors (e.g., typos, unreasonable ranges)
Identify highly correlated variables
Identify variables with (nearly) no variance
Identify variables with strange names or values
Check variable classes (eg. Characters vs factors)
Remove/transform some variables (maybe your model does not like categorial variables)
Rename some variables or values (if not all data is useful)
Check some overall pattern (statistical/ numerical summaries)
Possibly center/scale variables
Exploratory Data Analysis checklist
Suggest hypotheses about the causes of observed phenomena
Assess assumptions on which statistical inference will be based
Support the selection of appropriate statistical tools and techniques
Provide a basis for further data collection through surveys or experiments
Five methods that are must have:

Five number summaries (mean/median, min, max, q1, q3)
Histograms
Line charts
Box and whisker plots
Pairwise scatterplots (scatterplot matrices)

What values do you see?

What distributions do you see?
What relationships do you see?
What relationships do you think might benefit the prediction problem?
Answer the following questions for the data in each column:
How is the data distributed?
Test distribution assumptions (e.G. Normal distributions or skewed?)
What are the summary statistics?
Are there anomalies/outliers?
Identify useful raw data & transforms (e.g. log(x))
Identify data quality problems
Identify outliers
Identify subsets of interest
Suggest functional relationships


Assignment 2


Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the TAs can run the notebooks. Points will be deducted if the TAs can't run the notebooks.

Use the public dataset that you used for cleaning and EDA in Assignment 1. You MUST get approval if you wish to use a different dataset.

In this assingment you will cluster your data and create predictive linear and logistic models.

Cluster your data:
Use at least two methods to cluster your data. (25 points)
Answer the following questions for the clustering:

Assignment Apache Spark
Due Friday, March 16, 2018

Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the TAs can run the notebooks.

Note a Databricks shared notebook is acceptable as well. See https://docs.databricks.com/user-guide/notebooks/index.html

You MUST use the data set that you will use in your project.

In this assignment, you will:

Use Apache Spark for analysis of your project data.

Use Apache Spark for analysis of your project data
Part A - Set up Apache Spark
Sign Up for Databricks Community Edition https://accounts.cloud.databricks.com/registration.html#signup/community
Read the Databricks User Guide https://docs.databricks.com/user-guide/index.html
or

Download and run Apache Spark on a local machine or the cloud.
Part B - Structured Streaming (30 Points)
On your project data

Set up a Structured Streaming analysis of your project data. https://docs.databricks.com/spark/latest/structured-streaming/index.html
or the below is an alternate to Structured Streaming (30 Points)

Part B - Alternate - Structured Streaming (30 Points)
Come up with a set of at least 10 SQL questions that involve joins, order by, group by and aggregate statements and implement them on your data using Spark SQL.

Part C - MLlib and Machine Learning (40 Points)
On your project data

Apply a MLlib and Machine Learning analysis of your project data. https://docs.databricks.com/spark/latest/mllib/index.html
Part D - GraphX and GraphFrames or (30 Points)
On your project data

Set up a GraphX and GraphFrames analysis of your project data. https://docs.databricks.com/spark/latest/graph-analysis/index.html
or the below is an alternate to MLlib and Machine Learning (40 Points) & GraphX and GraphFrames or (30 Points)

Part C & D - Alternate - Deep Learning with Apache Spark and TensorFlow (70 Points)
Implement a Deep Learning with Apache Spark https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html

Hyperparameter Tuning (30 Points): use Spark to find the best set of hyperparameters for neural network training.

Compare Apache Spark and TensorFlow to TensorFlow not on Apache Spark (40 Points).

Use Apache Spark to apply a trained neural network model using Apache Spark and not using Apache Spark. Write a report of the Pros and Cons.


Assignment 4 - Deep Learning


Submission: Put the data and Jupyter notebook files in a folder. Make sure all links to data are relative to the folder so the TAs can run the notebooks.

Deep Learning for analysis of your project data (if possible).
Apply a Deep Learning model to your project data (if possible). You can use another data set ONLY if it makes no sense to use a Deep Learning model for your project.

There will be NO EXTENSIONS on the assingments due in April so start early

Part A - Deep Learning model (40 points)
On your project data

Apply a Deep Learning model to your project data (if possible). Validate the accuracy.
The Deep Learning model can be a CNN, RNN, Autoencoder, Variational autoencoder (VAE), Restricted Boltzmann machine (RBM), Deep belief network (DBN) or Generative Model. It cannot be a simple multilayer perceptron (MLP).
Part B - Activation function (10 points)
On your Deep Learning model data apply at least two different activation functions.

Change the activation function. How does it effect the accuracy?
How does it effect how quickly the network plateaus?
Various activation functions:
Rectified linear unit (ReLU)
TanH
Leaky rectified linear unit (Leaky ReLU)
Parameteric rectified linear unit (PReLU)
Randomized leaky rectified linear unit (RReLU)
Exponential linear unit (ELU)
Scaled exponential linear unit (SELU)
S-shaped rectified linear activation unit (SReLU)
Identity
Binary step
Logistic
ArcTan
Softsign
Adaptive piecewise linear (APL)
SoftPlus
SoftExponential
Sinusoid
Sinc
Gaussian
Part C - Cost function (10 points)
On your Deep Learning model data at least two different cost functions.

Change the cost function. How does it effect the accuracy?
How does it effect how quickly the network plateaus?
Various forms of cost:
Quadratic cost (mean-square error)
Cross-Entropy
Hinge
Kullback–Leibler divergence
Cosine Proximity
User defined
And many more, see https://keras.io/losses/

Part D - Epochs (10 points)
On your Deep Learning model data

Change the number of epochs initialization. How does it effect the accuracy?
How quickly does the network plateau?
Part E - Gradient estimation (10 points)
On your Deep Learning model data at least two qradient estimation algorithms.

Change the gradient estimation. How does it effect the accuracy?
How does it effect how quickly the network plateaus?
Various forms of gradient estimation:
Stochastic Gradient Descent
Adagrad
RMSProp
ADAM
NAG
Adadelta
Momentum
Part F - Network Architecture (10 points)
On your Deep Learning model data

Change the network architecture. How does it effect the accuracy?
How does it effect how quickly the network plateaus?
Various forms of network architecture:
Number of layers
Size of each layer
Connection type
Pre-trained components?
Part G - Network initialization (10 points)
On your Deep Learning model data at least two network initialization techniques.

Change the network initialization. How does it effect the accuracy?
How does it effect how quickly the network plateaus?
Various forms of network initialization:
0
Uniform
Gaussian
Xavier Glorot Initialization http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
Xavier Uniform
Xavier Gaussian

* Do the clustering methods generate the same clusters?
* Does scaling effect the clustering?  
* Does the clustering produce interesting groupings?  

Generate a linear model for your data:
Find a significant linear relation of your choosing in your data. Create a multivariate linear model. (50 points)
Answer the following questions for the multivariate linear model:

* Is the relationship significant?   
* Are any model assumptions violated?   
* Is there any multi-colinearity in the model?   
* In the multiple regression models are predictor variables independent of all the other predictor variables?   
* In in multiple regression models rank the most significant predictor variables and exclude insignificant ones from the model.   
* Does the model make sense?  
* Cross-validate the model. How well did it do?      
* Does regularization help with creating models that validate better on out of sample data?   

Generate a logistic model for your data:

Find a significant logistic linear model of your choosing in your data. Create a logistic linear model. (25 points)
Answer the following questions for the logistic linear model:

* Is the relationship significant?  
* Are any model assumptions violated?   
* Cross-validate the model. How well did it do?  4


