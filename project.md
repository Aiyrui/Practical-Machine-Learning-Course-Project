---
title: "PML Course Project"
author: "Aiyrui"
date: "4/20/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Summary

Using the data from: http://groupware.les.inf.puc-rio.br/har, we'll attempt to predict the manner in which people did their exercise. This is done using the "classe" variable in the training data We'll be approaching it using tree algorithms such as a Decision Tree and Random Forest on the training set, and compare their accuracy to find which machine learning algorithm does a better prediction. The final prediction model found from the training data will be used to predict the test data.

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Setting up Environment
```{r libraries, message=FALSE, warning=FALSE}
library(caret)
library(randomForest)
library(rpart)
library(rattle)

# Set seed
set.seed(32343)
```

## Data Processing and Cleansing

Downloading the necessary data sets into working directory and assigned to their appropriate files and objects.
```{r download}
if(!file.exists("trainingSet.csv")){
    urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    filename <- "trainingSet.csv"
    download.file(urlTrain, filename)
}

if(!file.exists("testingSet.csv")){
    urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    filename <- "testingSet.csv"
    download.file(urlTest, filename)
}

if(!exists("orgiTrain")){
    orgiTrain <- read.csv("trainingSet.csv", na.strings = c("", "NA", "#DIV/0!"))
}

if(!exists("orgiTest")){
    orgiTest <- read.csv("testingSet.csv", na.strings = c("", "NA", "#DIV/0!"))
}

```

We'll be removing variables that has zero covariates, NA's, and irrelevant variables (first 7).
```{r cleansing}
# Remove variables that has near zero covariates
nearZero <- nearZeroVar(orgiTrain)
subTrain <- orgiTrain[, -nearZero]

nearZeroTest <- nearZeroVar(orgiTest)
subTest <- orgiTest[, -nearZeroTest]

# Remove columns that has NA's
noNACol <- colSums(is.na(subTrain)) == 0
subTrain <- subTrain[, noNACol]
## dim(subTrain)

noNAColTest <- colSums(is.na(subTest)) == 0
subTest <- subTest[, noNAColTest]

```

```{r, echo=FALSE}

paste("Irrelevant Variable:", names(orgiTrain)[1:7])

```

```{r subset}
subTrain <- subTrain[, 8:59]

# Exploratory analysis on 'classe' variable
## "classe" %in% names(subTrain)
## unique(subTrain$classe)
## table(subTrain$classe)

# Leave alone for prediction later
finalTesting <- subTest[, 8:59]

```

## Data Slicing

Perform cross-validation by partitioning the training data into 2 sets: training (60%) and testing (40%)
```{r}
inTrain <- createDataPartition(y = subTrain$classe, p = 0.60, list = FALSE)
training <- subTrain[inTrain,]
testing <- subTrain[-inTrain,]

```


## Predictions

In this part, we'll be building a prediction model with the tree and random forest algorithm and compare their prediction accuracy.

#### Decision Tree
```{r}

fitTree <- train(classe ~ ., data = training, method = "rpart")
## fancyRpartPlot(fitTree$finalModel)

predictTree <- predict(fitTree, newdata = testing)
confusionMatrix(as.factor(testing$classe), predictTree)$overall["Accuracy"]
```

We find that **accuracy is roughly 50%**, which is not good enough for prediction. Since it is too low, we'll focus on using an algorithm that provides a better accuracy. Random Forest will do the trick.

#### Random Forest
```{r}
fitRF <- randomForest(as.factor(classe) ~., data = training)
predictRF <- predict(fitRF, newdata = testing)
confusionMatrix(as.factor(testing$classe), predictRF)$overall["Accuracy"]
```

Random Forest model has an **accuracy of 99%**.

## Conclusion

Predicting with random forest has an accuracy of approximately 99% so we'll use the Random Forest algorithm for prediction with testing set. Accuracy is the proportion of correctly classified observations in the training set, then 1 - Accuracy = error rate. We can assume that they're the expected accuracy and out of sample error in the test data. Therefore, 1 - Expected Accuracy = Expected Out of Sample Error, In the case of our prediction model with Random Forest, the **expected out of sample error is 1%**.

## Prediction with test data
```{r prediction}
predictTest <- predict(fitRF, newdata = finalTesting)
predictTest
```