from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

# Load Dataset
data, target = load_iris(return_X_y=True)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=1, shuffle=True)

svm = LinearSVC(random_state=0, tol=1e-5)
svm.fit(X_train,y_train)

# Save model
pickle.dump(svm, open('./models/model.pkl', 'wb'))