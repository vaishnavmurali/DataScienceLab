print("Starting script...")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Predicted test labels:", y_pred)
x_new = [[5, 5, 4, 4]]
y_new = gnb.predict(x_new)
print("Predicted output for [[5, 5, 4, 4]]:", y_new)
print("Naive Bayes score:", gnb.score(X_test, y_test))
