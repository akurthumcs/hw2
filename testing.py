from firstattempt.training import trainBayes
from firstattempt.classify import classifyTestData, argmax
from firstattempt.metrics import displayMetrics
from utils import load_training_set, load_test_set
"""File for testing/debugging gonna do this caveman style"""
posdata, negdata, V = load_training_set(.0004, .0004)
i1, i2, i3, lc1, lc2 = trainBayes(posdata, negdata, V, 1)
posdata, negdata = load_test_set(.0004, .0004)
data = classifyTestData(posdata, negdata, True, lc1, lc2, i1, i2, i3)
displayMetrics(data)
