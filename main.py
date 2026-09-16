import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())    
print(df.tail())

print(df.isnull().sum())

df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})


print(df['Embarked'].value_counts())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Age'] = df['Age'].fillna(df['Age'].median())

X = df[['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
print(df['Gender'])

y = df['Survived']
print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Predicted values:", y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print("The accuracy of the model is:", accuracy)