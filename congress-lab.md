# Congress Voting Data

## Note

This lab will be a lot easier if you use [titanic.ipynb](titanic.ipynb) as a guide.

## Overview

The data for this lab is in the attached file called `congress.csv`. 

Each line corresponds to a congressperson, which party they belong to,
and how they voted on 16 different issues. 

The dataset is from UCI
[here](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records).

Here are the meanings of the columns in the CSV file.

Column                                   |Values
------                                   |------
`party`                                  |`democrat` ` republican`
`handicapped-infants`                    |`?` `n` `y`
`water-project-cost-sharing`             |`?` `n` `y`
`adoption-of-the-budget-resolution`      |`?` `n` `y`
`physician-fee-freeze`                   |`?` `n` `y`
`el-salvador-aid`                        |`?` `n` `y`
`religious-groups-in-schools`            |`?` `n` `y`
`anti-satellite-test-ban`                |`?` `n` `y`
`aid-to-nicaraguan-contras`              |`?` `n` `y`
`mx-missile`                             |`?` `n` `y`
`immigration`                            |`?` `n` `y`
`synfuels-corporation-cutback`           |`?` `n` `y`
`education-spending`                     |`?` `n` `y`
`superfund-right-to-sue`                 |`?` `n` `y`
`crime`                                  |`?` `n` `y`
`duty-free-exports`                      |`?` `n` `y`
`export-administration-act-south-africa` |`?` `n` `y`

Based on the votes on the 16 issues we are going to predict the party
using Random Forests.

# Lab 1: Data Analysis

## Download Data

The data is in `congress.csv` which is attached to this gist.

Look at the CSV data. Looks like it does not have any headers. 

## Load Data

Import packages.

    import pandas as pd
    import numpy as np

Load the CSV. Since the CSV file does not have headers so we load it
with `header=None`.

    df = pd.read_csv('data/congress.csv', header=None)

Do a quick sanity check to see that it loaded.

    df.head()

## Naming Columns

Lets give the columns names. First define some column names.

    df_columns = [
        "party",
        "infants", "water", "budget", "physician",
        "el_salvador", "religious", "satellite", "contras",
        "missile", "immigration", "synfuels", "education",
        "superfund", "crime", "exports", "south_africa",
    ]

Now assign these to `df.columns`. 

    df.columns = df_columns

Check that the columns now have these names.

    df.head()

## Binarizing Data

Random Forests and other machine learning algorithms expect the data
to be in a numeric format. We have to replace all the strings with
numbers.

In our case the values are mostly binary so we will replace them with
0 and 1.

Modify the DataFrame to replace `democrat` with 0 and `republican`
with 1. Then save this result to a DataFrame.

Next, modify the DataFrame to replace `n` and `?` with 0 and `y` with
1 in all 16 columns. Save this result to a DataFrame.

## Abstain Columns

Is it reasonable to convert `?`, `n`, `y` to 2 values? Should we have
converted it to 3? For each column we could have added a new column
which was 0 or 1 if the congressperson abstained.

This would give us another 16 features. The upside is that our model
might perform better. The downside is that the model might take longer
to train with all the additional dimensions (due to the *curse of
dimensionality*).

Instead we are making an assumption here that abstaining is a weaker
form of `n` and so we are grouping them together.

## Analyzing Data

Before we build a model we want to analyze the data to figure out if
there are patterns that stand out. This will give us a good intuition
about the data and suggest how to approach it.

## Water Project Cost Sharing

Lets calculate what percentage of Democrats and Republicans support
the bill called `water-project-cost-sharing`.

To do this group the records by party, and then calculate the mean
value of the `water-project-cost-sharing` field.

Create a scatter matrix of the data. 

## Analyzing Voting Across Issues

Extra Credit: Using the same technique as in the previous exercise
calculate the percentages of Democrats and Republicans who supported
each of the 16 bills.

# Lab 2: Machine Learning Workflow

## Preparation: Defining X and y

Next we are going prepare the data for feeding into our machine
learning algorithm.

scikit-learn uses Numpy arrays. We are going to define `X` and `y`
from our DataFrame and save them as Numpy arrays.

Put the output, the party affiliation, column index 0, into `y`. To do
this slice the DataFrame so that you just get the first column. Then
apply `.values` to this to turn it into a Numpy array. 

Put the inputs, the 16 voting columns, column indexes 1-16, into a
`X`. To do this slice the DataFrame so you get the second through the
last column values. Then apply `.values` to this to turn it into a
Numpy matrix. 

## Preparation: Split 70/30

Do a 70%-30% split of the the DataFrame into a training set and a test
set. Use the scikit-learn method `train_test_split` to do this.

Save the training set as `X_train` and `y_train`. 

Save the test set as `X_test` and `y_test`.

## Training: Random Forest Model

Next we will predict the party affiliation of a congressperson based
on their voting record.

Fit a Random Forest model with `n_estimators=100`. 

## Evaluation: Using Test Data

Evaluate this model on the test set. How well did it do?

## Evaluation: Using Training Data

Evaluate this model on the training set. How well did it do?

Did it do better or worse than on the training data? 

Are you concerned about over-fitting?

## Cross Validation

Use 10-fold cross validation to check if the performance of your model
holds.

How well does your model do?

## Optimization: Tuning Hyperparameters

Next tune the hyperparameters using Grid Search. 

Here are the choices that make up our grid.

Hyperparameters  |Values
---------------  |------
`n_estimators`   |10,20,50,100,200,500
`max_depth`      |1,3,5,10,20
`max_features`   |1,2,4,8,16

Did tuning hyperparameters make the model improve measurably?

Would you consider the optimization step to be worthwhile? 
