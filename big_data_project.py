# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:12:45 2018

@author: Guanhua
"""
from __future__ import print_function
import sys
import re
import numpy as np
from datetime import datetime
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import UserDefinedFunction, monotonically_increasing_id
from pyspark.sql import Row
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, NaiveBayes, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

#import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',\
             'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',\
             "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',\
             'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',\
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',\
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',\
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',\
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',\
             'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',\
             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',\
             'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",\
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',\
             "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",\
             'wouldn', "wouldn't"]

conf = SparkConf()
conf = (conf.setMaster('local[*]').set('spark.executor.memory', '8G'))
print(conf.getAll())
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

regex = re.compile('[^a-zA-Z]')
regex_num = re.compile('[^0-9]')
udf_regex = UserDefinedFunction(lambda x: regex.sub(' ', x.lower()), StringType())

ndim = 20000

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <myscript.py> <training dataset> <testing dataset> <results output> ", file=sys.stderr)
        exit(-1)
        
    lines = sc.textFile(sys.argv[1], 1)
    testlines = sc.textFile(sys.argv[2], 1)
    
    def preprocessdata(lines):    
        lines = lines.map(lambda x: x.split(','))
        lines = lines.map(lambda x: (x[0].replace('"', ""), x[2]))
        lines = lines.map(lambda x: Row(label=x[0], text=x[1]))
        df = sqlContext.createDataFrame(lines)
        df = df.withColumn("index", monotonically_increasing_id())
        df = df.select(*[udf_regex(column).alias('text') if column == 'text' else column for column in df.columns])
        df = df.withColumn("label", df["label"].cast(DoubleType()))
        df = df.withColumn("label", df["label"] - 1.0).cache() # to make labels 1 and 0, originally good = 2, bad = 1
        df.show()
        return df
    
    def split_dataset(dataset):
        return dataset.randomSplit([0.7, 0.3], seed = 2018)

    def TFpipeline():
        regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stopwords)
        countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=ndim, minDF=5)
        pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])
        return pipeline
    
    def TFIDFpipeline():
        regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stopwords)
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=ndim)
        idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
        pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf])
        return pipeline
    
    model = ['Logistic Regression']*4 + ['RandomForest']*4 + ['LinearSVM']*4 + ['Naive Bayes']*4
    feature_type = ['Term Frequencies', 'Term Frequencies', 'TFIDF', 'TFIDF']* 4
    metric = ['pr', 'auc'] * 8
    values = []
    runtimes = []
    
    df = preprocessdata(lines)
    df_test = preprocessdata(testlines)
    
    # Logistic Regression with Term Frequencies
    trainingStartTime = datetime.now()
    print("Logistic Regression Model with Term Frequencies Training started at %s" %str(trainingStartTime))
    pipelineFit = TFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    lr = LogisticRegression(maxIter=100, regParam=0.2)
    lrModel = lr.fit(dataset)
    predictions = lrModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("Total Training time %s" %str(runtime))
    
    # Logistic Regression with TF-IDF
    trainingStartTime = datetime.now()
    print("Logistic Regression Model with TF-IDF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFIDFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)    train_set, val_set = split_dataset(dataset)
    lr = LogisticRegression(maxIter=100, regParam=0.2)
    lrModel = lr.fit(dataset)
    predictions = lrModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("Logistic Regression Model with TF-IDF Total Training time %s" %str(runtime))
    
    ## RandomForest with Term Frequencies (run this on cloud)
    trainingStartTime = datetime.now()
    print("RandomForest Model with TF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 100, maxDepth = 4, maxBins = 32)
    rfModel = rf.fit(dataset)
    predictions = rfModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("RandomForest Model with TF-IDF Total Training time %s" %str(runtime))
    
    ## RandomForest with TF-IDF (run this on cloud)
    trainingStartTime = datetime.now()
    print("RandomForest Model with TF-IDF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFIDFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 100, maxDepth = 4, maxBins = 32)
    rfModel = rf.fit(dataset)
    predictions = rfModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)    
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("RandomForest Model with TF-IDF Total Training time %s" %str(runtime))

    ## LinearSVM with Term Frequencies
    trainingStartTime = datetime.now()
    print("LinearSVM Model with TF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    lsvc = LinearSVC(maxIter=100, regParam=0.1)
    lsvcModel = lsvc.fit(dataset)
    predictions = lsvcModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("LinearSVM Model with TF-IDF Total Training time %s" %str(runtime))
    
    ## LinearSVM with TF-IDF
    trainingStartTime = datetime.now()
    print("LinearSVM Model with TF-IDF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFIDFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    lsvc = LinearSVC(maxIter=100, regParam=0.1)
    lsvcModel = lsvc.fit(dataset)
    predictions = lsvcModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("LinearSVM Model with TF-IDF Total Training time %s" %str(runtime))

    ## Naive Bayes with Term Frequencies
    trainingStartTime = datetime.now()
    print("Naive Bayes Model with TF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    nbModel = nb.fit(dataset)
    predictions = nbModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("Naive Bayes Model with TF Total Training time %s" %str(runtime))
    
    ## Naive Bayes with TF-IDF
    trainingStartTime = datetime.now()  
    print("Naive Bayes Model with TF-IDF Training started at %s" %str(trainingStartTime))
    pipelineFit = TFIDFpipeline().fit(df)
    dataset = pipelineFit.transform(df)
    testset = pipelineFit.transform(df_test)
#    train_set, val_set = split_dataset(dataset)
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    nbModel = nb.fit(dataset)
    predictions = nbModel.transform(testset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    values.append(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    runtime = int((datetime.now() - trainingStartTime).total_seconds()/60)
    runtimes.append(runtime)
    runtimes.append(runtime)
    print("Area under PR curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"}))
    print("Area under ROC curve %0.5f" %evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    print("Naive Bayes Model with TF-IDF Total Training time %s" %str(runtime))
    
    ## output
    result = Row("model", "feature", "metric", "values", "runtimes")
    mylist = []
    for i in range(len(model)):
        record = result(model[i], feature_type[i], metric[i], values[i], runtimes[i])
        mylist.append(record)
    df_result = sqlContext.createDataFrame(mylist)
    df_result.show()
    df_result.coalesce(1).write.csv(sys.argv[3])
    sc.stop()