from datetime import datetime

import numpy as np

from vowpal_platypus import run
from vowpal_platypus.models import logistic_regression
from vowpal_platypus.utils import clean


def ginic(actual, pred):
    actual = np.asarray(actual) 
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1] 
    return ginic(a, p) / ginic(a, a)


start = datetime.now()
print('...Starting at ' + str(start))


model = logistic_regression(name='PortoVW',
                            passes=100,
                            bits=23,
                            l1=0.00001,
                            l2=0.0001,
                            cores=3)

def process_line(item, header, predict=False):
    headers = map(clean, header.split(','))
    items = item.split(',')
    items = dict(zip(headers, items))
    items.pop('id')
    if not predict:
        target = int(items.pop('target'))
    namespaces = {'ind': [], 'reg': [], 'car': []}
    for name, value in items.items():
        if 'bin' in name or 'cat' in name:
            value = name + '_' + clean(value)
        else:
            value = {name: (float(value) if value != "NA" else 0.0)}
        for namespace in namespaces.keys():
            if namespace in name:
                namespaces[namespace].append(value)
    if not predict:
        namespaces['label'] = target
        namespaces['importance'] = 4 if target == 1 else 1
    namespaces['i'] = namespaces.pop('ind')
    namespaces['r'] = namespaces.pop('reg')
    namespaces['c'] = namespaces.pop('car')
    return namespaces


with open('train_train.csv') as f:
    header = f.readline()

results = run(model,
              train_filename='train_train.csv',
              train_line_function=lambda line: process_line(line, header),
              predict_filename='train_test.csv',
              predict_line_function=lambda line: process_line(line, header, predict=True))


actuals = []
with open('train_test.csv') as f:
    _ = f.readline() # throwaway header
    for line in f:
        actuals.append(int(line.split(',')[1]))

end = datetime.now()
print('Gini norm: ' + str(normalized_gini(actuals, results)))
print('Num Predicted: ' + str(len(results)))
print('Elapsted model time: ' + str(end - start))
print('Model speed: ' + str((end - start).total_seconds() * 1000000 / float(len(results))) + ' mcs/row')
