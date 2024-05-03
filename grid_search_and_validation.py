from sklearn import datasets
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import uniform, randint

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

distributions = {'n_estimators' : uniform(100, 1000),
                'learning_rate' : uniform(0.01, 0.2),
                'subsample' : uniform(0.5, 0.5),
                'max_depth' : randint(4, 10)}

gb_regressor = GradientBoostingRegressor()

randomised_search = RandomizedSearchCV(estimator = gb_regressor, param_distributions=distributions)
randomised_search.fit(X_train, y_train)

#using the best parameters
best_gb_params = randomised_search.best_params_

best_model = GradientBoostingRegressor(**best_gb_params).fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred, normalize=True)
print("Validation Accuracy:", accuracy)
print("Best Hyperparameters:", best_gb_params)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()