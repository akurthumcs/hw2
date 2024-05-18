from utils import*
from helpers import*
from matplotlib import pyplot as plt

posdata, negdata, V = load_training_set(.2, .2)
postest, negtest = load_test_set(.2, .2)
alphas = [.0001, .001, .01, .1, 1, 10, 100, 1000]

numpos = len(posdata)
numneg = len(negdata)
numreviews = numpos + numneg

probpos = numpos / numreviews
probneg = numneg / numreviews

wordCountPos = getWordCounts(posdata)
wordCountNeg = getWordCounts(negdata)

accs = []
for alpha in alphas:
    probWordGivenPos, plc = getProbWordGivenClassWithSmoothing(V, wordCountPos, alpha)
    probWordGivenNeg, nlc = getProbWordGivenClassWithSmoothing(V, wordCountNeg, alpha)
    results = getResultsLaplace(postest, negtest, probpos, probWordGivenPos, probneg, probWordGivenNeg, plc, nlc)
    if alpha == 1:
        confusionMatrix = getConfusionMatrix(results)
        print("Confusion Matrix: ")
        print(confusionMatrix)
        print("Accuracy: ")
        print(getAccuracy(confusionMatrix))
        print("Precision: ")
        print(getPrecision(confusionMatrix))
        print("Recall: ")
        print(getRecall(confusionMatrix))
    confusionMatrix = getConfusionMatrix(results)
    accs.append(getAccuracy(confusionMatrix))

plt.xscale("log")
plt.plot(alphas, accs)
plt.xlabel("Values of laplace smoothing parameter")
plt.ylabel("Accuracy of model on test data")
plt.title("Accuracy of Naive Bayes classifier, over different laplace smoothing parameters")
plt.show()

