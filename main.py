import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/train.csv')

def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return pd.DataFrame()

def clean_data(df):
    df = df.drop(['Cabin'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def engineer_features(df):
    fares = df['Fare'].to_numpy()
    df['Fare_norm'] = (fares - fares.min()) / (fares.max() - fares.min())
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['isAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    return df

def print_summary(df):
    print("Data Summary:")
    print(df.head())

def train_model(df):
    # dropping all data that is not numeric as model wont be able to process it
    x = df.drop(['Survived', 'Ticket', 'Name', 'PassengerId'], axis=1)
    y = df['Survived']
    x['Age'] = x['Age'].fillna(x['Age'].median())
    x['Sex'] = np.where(x['Sex'] == "male", 0, 1)
    dumm_embarked = pd.get_dummies(x['Embarked'], prefix="Embarked")
    x['Embarked_C'] = dumm_embarked['Embarked_C']
    x['Embarked_Q'] = dumm_embarked['Embarked_Q']
    x['Embarked_S'] = dumm_embarked['Embarked_S']
    x = x.drop(['Embarked'], axis=1)

    x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model = model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

if __name__ == "__main__":
    df = load_data('data/train.csv')
    df = clean_data(df)
    df = engineer_features(df)
    print_summary(df)
    # print_summary(df)
    train_model(df)