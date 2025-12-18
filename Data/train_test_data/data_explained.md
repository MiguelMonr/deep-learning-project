# Train - validation - test data

This folder contains the datasets used for training and testing the models. The same split must be used for comparability of diferent models.

Datasets have been preprocessed to convert boolean data to numeric representations, but have lost column names.

This same split can be achieved by setting `random_state=42` on sklearn's train-test split.

``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.22, random_state=42)
```
