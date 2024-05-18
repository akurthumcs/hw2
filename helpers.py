from collections import defaultdict
from numpy import log

def getWordCounts(data):
    wordCount = defaultdict(int)
    for review in data:
        for word in review:
            wordCount[word] += 1
    return wordCount

def getProbWordGivenClass(V, wordCount):
    sum = 0
    for word in V:
        sum += wordCount[word]
    probWordGivenClass = defaultdict(int)
    for word in V:
        probWordGivenClass[word] = wordCount[word]/ sum
    return probWordGivenClass

def getProbWordGivenClassWithSmoothing(V, wordCount, alpha):
    sum = 0
    for word in V:
        sum += wordCount[word]
    probWordGivenClass = defaultdict()
    for word in V:
        probWordGivenClass[word] = (wordCount[word] + alpha)/(sum + (alpha*len(V)))
    laplaceConstant = alpha/ (sum + (alpha*len(V)))
    return probWordGivenClass, laplaceConstant

def getResults(postest, negtest, probPos, probWordGivenPos, probNeg, probWordGivenNeg):
    results = []
    for review in postest:
        probPosGivenReview = log(probPos)
        review = set(review)
        for word in review:
            if word in probWordGivenPos.keys() and not probWordGivenPos[word] == 0 :
                probPosGivenReview += log(probWordGivenPos[word])
            else:
                probPosGivenReview += log(.00000001)
        probNegGivenReview = probNeg
        for word in review:
            if word in probWordGivenNeg.keys() and not probWordGivenNeg[word] == 0:
                probNegGivenReview += log(probWordGivenNeg[word])
            else:
                probNegGivenReview += log(.00000001)
        if probPosGivenReview > probNegGivenReview:
            results.append((1, 1))
        else:
            results.append((1, -1))

    for review in negtest:
        probPosGivenReview = log(probPos)
        review = set(review)
        for word in review:
            if word in probWordGivenPos.keys() and not probWordGivenPos[word] == 0:
                probPosGivenReview += log(probWordGivenPos[word])
            else:
                probPosGivenReview += log(.00000001)
        probNegGivenReview = probNeg
        for word in review:
            if word in probWordGivenNeg.keys() and not probWordGivenNeg[word] == 0:
                probNegGivenReview += log(probWordGivenNeg[word])
            else:
                probNegGivenReview += log(.00000001)
        if probPosGivenReview > probNegGivenReview:
            results.append((-1, 1))
        else:
            results.append((-1, -1))

    return results

def getResultsLaplace(postest, negtest, probPos, probWordGivenPos, probNeg, probWordGivenNeg, plc, nlc):
    results = []
    for review in postest:
        review = set(review)
        probPosGivenReview = log(probPos)
        for word in review:
            if word in probWordGivenPos.keys() and not probWordGivenPos[word] == 0 :
                probPosGivenReview += log(probWordGivenPos[word])
            else:
                probPosGivenReview += log(plc)
        probNegGivenReview = probNeg
        for word in review:
            if word in probWordGivenNeg.keys() and not probWordGivenNeg[word] == 0:
                probNegGivenReview += log(probWordGivenNeg[word])
            else:
                probNegGivenReview += log(nlc)
        if probPosGivenReview > probNegGivenReview:
            results.append((1, 1))
        else:
            results.append((1, -1))

    for review in negtest:
        probPosGivenReview = log(probPos)
        review = set(review)
        for word in review:
            if word in probWordGivenPos.keys() and not probWordGivenPos[word] == 0:
                probPosGivenReview += log(probWordGivenPos[word])
            else:
                probPosGivenReview += log(.00000001)
        probNegGivenReview = probNeg
        for word in review:
            if word in probWordGivenNeg.keys() and not probWordGivenNeg[word] == 0:
                probNegGivenReview += log(probWordGivenNeg[word])
            else:
                probNegGivenReview += log(.00000001)
        if probPosGivenReview > probNegGivenReview:
            results.append((-1, 1))
        else:
            results.append((-1, -1))

    return results

def getResultsNoLog(postest, negtest, probPos, probWordGivenPos, probNeg, probWordGivenNeg):
    results = []
    for review in postest:
        probPosGivenReview = probPos
        review = set(review)
        for word in review:
            probPosGivenReview *= probWordGivenPos[word]
        probNegGivenReview = probNeg
        for word in review:
            probNegGivenReview *= probWordGivenNeg[word]
        if probPosGivenReview > probNegGivenReview:
            results.append((1, 1))
        else:
            results.append((1, -1))
    
    for review in negtest:
        probPosGivenReview = probPos
        review = set(review)
        for word in review:
            probPosGivenReview *= probWordGivenPos[word]
        probNegGivenReview = probNeg
        for word in review:
            probNegGivenReview *= probWordGivenNeg[word]
        if probPosGivenReview > probNegGivenReview:
            results.append((-1, 1))
        else:
            results.append((-1, -1))
    return results

def getConfusionMatrix(results):
    TP = 1
    FP = 0
    TN = 0
    FN = 0
    for prediction in results:
        if prediction == (1, 1):
            TP += 1
        elif prediction == (1, -1):
            FN += 1
        elif prediction == (-1, 1):
            FP += 1
        elif prediction == (-1, -1):
            TN += 1
    return [[TP, FN], [FP, TN]]

def getAccuracy(cMatrix):
    return (cMatrix[0][0] + cMatrix[1][1])/(cMatrix[0][0] + cMatrix[0][1] + cMatrix[1][0] + cMatrix[1][1])

def getPrecision(cMatrix):
    return (cMatrix[0][0]) / (cMatrix[0][0] + cMatrix[1][0])

def getRecall(cMatrix):
    return (cMatrix[0][0]) / (cMatrix[0][0] + cMatrix[0][1])