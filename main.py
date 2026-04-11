import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    df = load_data('data/train.csv')
    df = clean_data(df)
    df = engineer_features(df)
    print(df.head())