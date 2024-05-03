from sklearn import datasets
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import uniform, randint
import random

X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)

random_float = random.uniform(0, 1) 
random_integer_est = random.randint(100, 1000) 
random_integer = random.randint(1, 10)

gb_classifier = GradientBoostingClassifier(n_estimators=random_integer_est, 
                                           learning_rate=random_float, 
                                           subsample=random_float,
                                           max_depth=random_integer).fit(X_train, y_train)

distributions = dict(C=uniform(0, 4), penalty=['l2', 'l1'])

randomised_search = RandomizedSearchCV(gb_classifier, distributions, random_state=0)
randomised_search.fit(X_train, y_train)

#using the best parameters
best_gb_params = randomised_search.best_params_

best_score = GradientBoostingClassifier(**best_gb_params, random_state=42).fit(X_train, y_train)

y_pred = best_score.predict(X_test)

accuracy = accuracy_score(y_test, y_pred, normalize=True)
print("Validation Accuracy:", accuracy)
print("Best Hyperparameters:", best_gb_params)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()