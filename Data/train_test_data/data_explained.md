# Train - test data

This folder contains the datasets used for training and testing the models. The same split must be used for comparability of diferent models.

X\_...\_numeric datasets have been preprocessed to convert boolean data to numeric representations, but have lost column names. X\_...\_boolean have not.

This same split can be achieved by setting `random_state=42` on sklearn's train-test split.

``` python
from sklearn.model_selection import train_test_split
train_test_split(X, y, train_size = 0.8, random_state = 42)
```
