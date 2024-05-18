from utils import*
from helpers import*

posdata, negdata, V = load_training_set(1, 1)
postest, negtest = load_test_set(1, 1)

numpos = len(posdata)
numneg = len(negdata)
numreviews = numpos + numneg

probpos = numpos / numreviews
probneg = numneg / numreviews

wordCountPos = getWordCounts(posdata)
wordCountNeg = getWordCounts(negdata)

probWordGivenPos, plc = getProbWordGivenClassWithSmoothing(V, wordCountPos, 1)
probWordGivenNeg, nlc = getProbWordGivenClassWithSmoothing(V, wordCountNeg, 1)

results = getResultsLaplace(postest, negtest, probpos, probWordGivenPos, probneg, probWordGivenNeg, plc, nlc)
confusionMatrix = getConfusionMatrix(results)
print("Results when testing 100 percent of training and testing data: ")
print("Confusion Matrix: ")
print(confusionMatrix)
print("Accuracy: ")
print(getAccuracy(confusionMatrix))
print("Precision: ")
print(getPrecision(confusionMatrix))
print("Recall: ")
print(getRecall(confusionMatrix))
