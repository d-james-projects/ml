import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
print("######## get dataset ########")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
array = dataset.values
X = array[:,0:4]
print(len(X))
Y = array[:,4]
print(len(Y))

model = LogisticRegression()
model.fit(X, Y)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

