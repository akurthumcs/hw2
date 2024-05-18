from utils import*
from collections import defaultdict
from helpers import *

posdata, negdata, V = load_training_set(.2, .2)

#Train Naive Bayes Classifier
numpos = len(posdata)
numneg = len(negdata)
numreviews = numpos + numneg

probpos = numpos / numreviews
probneg = numneg / numreviews

wordCountPos = getWordCounts(posdata)
wordCountNeg = getWordCounts(negdata)

probWordGivenPos = getProbWordGivenClass(V, wordCountPos)
probWordGivenNeg = getProbWordGivenClass(V, wordCountNeg)

#Testing
postest, negtest = load_test_set(.2, .2)

results = getResultsNoLog(postest, negtest, probpos, probWordGivenPos, probneg, probWordGivenNeg)
resultslog = getResults(postest, negtest, probpos, probWordGivenPos, probneg, probWordGivenNeg)
#Display metrics
print("Metrics without using logarithm")
confusionMatrix = getConfusionMatrix(results)
print("Confusion Matrix: ")
print(confusionMatrix)
print("Accuracy: ")
print(getAccuracy(confusionMatrix))
print("Precision: ")
print(getPrecision(confusionMatrix))
print("Recall: ")
print(getRecall(confusionMatrix))

print("Metrics when using logarithm")
confusionMatrix2 = getConfusionMatrix(resultslog)
print("Confusion Matrix: ")
print(confusionMatrix2)
print("Accuracy: ")
print(getAccuracy(confusionMatrix2))
print("Precision: ")
print(getPrecision(confusionMatrix2))
print("Recall: ")
print(getRecall(confusionMatrix2))
