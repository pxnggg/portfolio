import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
import warnings

warnings.filterwarnings("ignore") 

data = pd.read_csv("seattle-weather.csv",usecols=[1,2,3,4,5])
x = data.drop("weather",axis=1)
y = data['weather']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3 , random_state=7)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))

precipitation,temp_max,temp_min,wind = input("precipitation : "),input("temp_max : "),input("temp_min : "),input("wind : ")

input_pre = model.predict([[(precipitation), (temp_max), (temp_min), (wind)]])
print("Weather : ",input_pre[0])



