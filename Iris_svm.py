#Using SVM on Iris dataset
import sklearn
from sklearn import svm
from sklearn import datasets
iris = datasets.load_iris()
type(iris)
iris.data
iris.feature_names
iris.target
iris.target_names
A= iris.data[:,2]
B= iris.target
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test= train_test_split(A,B, test_size=0.2, random_state=4)
A_train_mod = A_train.reshape(-1,1)
A_test_mod = A_test.reshape(-1,1)
B_train_mod = B_train.reshape(-1,1)
B_test_mod = B_test.reshape(-1,1)
model=svm.SVC(kernel='linear')
model.fit(A_train_mod,B_train_mod)
Z_pred_mod = model.predict(A_test_mod)
from sklearn.metrics import accuracy_score
print(accuracy_score(B_test_mod,Z_pred_mod))
