import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, \
    classification_report

# plot
import matplotlib.pyplot as plt
import seaborn as sns

print('Hi')

train_df = pd.read_csv('./credit-score-classification-train.csv')

train_df.head(5)

train_df.info()

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=0.9)
train_df.isna().sum().plot(kind='bar')

train_df.describe(include='object').T

train_df.describe().T

train_df['Credit_Score'].value_counts(normalize=True)

sns.countplot(x=train_df['Credit_Score'])

for col in train_df.columns:
    print(col)
    print(train_df[col].unique())
    print('======')

# Data Cleaning


# Categorical Cols

categorical_cols = []
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        categorical_cols.append(col)
categorical_cols

for col in categorical_cols:
    train_df[col] = train_df[col].replace(
        {'!@9#%8': 'NAN_spent_NAN_value_payments', '#F%$D@*&8': 'NAN-00-000', '_______': np.nan,
         '__-333333333333333333333333333__': np.nan, 'NM': np.nan, '_': np.nan, '__10000__': np.nan})

# remove _ from some data
strip_list = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Outstanding_Debt']

for col in strip_list:
    train_df[col] = train_df[col].str.strip("_")

train_df['Age'] = train_df['Age'].astype(int)

train_df['Age'] = np.where(train_df['Age'] > 75, np.NAN, train_df['Age'])

train_df['area'] = train_df['SSN'].apply(lambda x: x.split('-')[0])

train_df.insert(5, 'Area', train_df['area'])

train_df['Area'] = train_df['Area'].replace('NAN', np.NAN)

# Change type  of some features
type_col = ['Area', 'Monthly_Inhand_Salary', 'Monthly_Balance', 'Num_of_Delayed_Payment', 'Outstanding_Debt',
            'Amount_invested_monthly', 'Annual_Income', 'Num_of_Loan', 'Changed_Credit_Limit']

for col in type_col:
    train_df[col] = train_df[col].astype(float)

# Feature Engineering

train_df["AgeLevel"] = pd.cut(train_df["Age"], 4, labels=["children", "youth", "adult", "Seniors"])

train_df['New_Payment_Behaviour'] = train_df['Payment_Behaviour'].apply(
    lambda x: x.split('_value_')[0].replace('_spent_', ''))

# calculate Credit_History_Age by months

train_df['Credit_History_Age'] = train_df['Credit_History_Age'].replace(np.NAN, '0 Years and 0 Months')

train_df['year'] = train_df['Credit_History_Age'].apply(lambda x: x.split('and')[0]).str.replace('Years', '')

train_df['year'] = pd.to_numeric(train_df['year'], errors='coerce').apply(lambda x: x * 12)
train_df['year'].dtypes

train_df['month'] = train_df['Credit_History_Age'].apply(lambda x: x.split('and')[1]).str.replace('Months', '')

train_df['month'] = pd.to_numeric(train_df['month'], errors='coerce')
train_df['month'].dtypes

train_df['New_Credit_History_Age (months)'] = train_df['year'] + train_df['month']

# Convert Categorical Data

dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12}
train_df['Month'] = train_df['Month'].map(dict)

from sklearn import preprocessing

lb = preprocessing.LabelEncoder()
enc_list = ['Credit_Score', 'Payment_of_Min_Amount', 'Occupation', 'Credit_Mix', 'AgeLevel', 'New_Payment_Behaviour']
for col in enc_list:
    train_df[col] = lb.fit_transform(train_df[col])


# Missing values
def display_missing_data(df):
    missing_data = df.isnull().sum().reset_index()
    missing_data = missing_data.rename({'index': 'col', 0: 'null'}, axis=1)
    missing_data['null_percenatage'] = missing_data['null'] / len(df)
    missing_data = missing_data.loc[missing_data['null'] > 0]
    return missing_data


print(display_missing_data(train_df))

# calculate   (median, mean,mode) depend on groupby

group_list = ['Age', 'Monthly_Inhand_Salary', 'Payment_of_Min_Amount', 'Monthly_Balance', 'Changed_Credit_Limit',
              'Changed_Credit_Limit', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Amount_invested_monthly']

for col3 in group_list:
    train_df[col3] = train_df.groupby(['Customer_ID'], sort=False)[col3].apply(lambda x: x.fillna(x.median()))

group_list2 = ['Area', 'Occupation', 'Payment_of_Min_Amount', 'Credit_Mix', 'New_Payment_Behaviour']

for col4 in group_list2:
    train_df[col4] = train_df.groupby(['Customer_ID'], sort=False)[col4].apply(lambda x: x.fillna(x.mode()[0]))

train_df.drop(axis=1,
              columns=['ID', 'Customer_ID', 'Name', 'Age', 'area', 'SSN', 'Credit_History_Age', 'Payment_Behaviour',
                       'Type_of_Loan', 'year', 'month'], inplace=True)

# Visualization

sns.set(rc={'figure.figsize': (6, 6)}, font_scale=1.2)
train_df.groupby('Occupation').size().plot(kind='pie', autopct='%.2f', textprops={'fontsize': 8})

sns.stripplot(x='Credit_Score', y='Num_Bank_Accounts', data=train_df)

sns.stripplot(x="Credit_Mix", y='Credit_Score', data=train_df)

sns.set(rc={'figure.figsize': (8, 8)}, font_scale=1.2)

sns.kdeplot(train_df['Annual_Income'], shade=True)

sns.set(rc={'figure.figsize': (8, 8)}, font_scale=1.2)
sns.boxplot(y='Monthly_Inhand_Salary', data=train_df)

# Bivariate plots
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.scatterplot(data=train_df, x='Monthly_Balance', hue='Occupation', y='Credit_Score')
plt.show()

train_df

# Feature Selection
X = train_df.drop(['Credit_Score'], axis=1)
Y = train_df['Credit_Score']
# from sklearn.feature_selection import f_regression, mutual_info_regression
# from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import SelectFromModel

all_features = X.columns
all_features

import lightgbm as lgb

# model = XGBRegressor()
# model=RandomForestRegressor(n_estimators=100,max_depth=2, random_state=33)
model = lgb.LGBMRegressor(num_leaves=50, learning_rate=0.005, n_estimators=40)
# model=RandomForestClassifier(n_estimators=70)

# selector = SelectKBest(k=60, score_func=f_regression)
# selector = SelectKBest(k=40, score_func=mutual_info_regression)
# selector = SelectPercentile(percentile=50, score_func=mutual_info_regression)
selector = SelectFromModel(estimator=model)

selector.fit(X, Y)

selected_features_idx = selector.get_support(indices=True)
selected_features_idx

selected_features = all_features[selected_features_idx]
selected_features

# Scaler data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X[selected_features])

X[selected_features] = scaler.transform(X[selected_features])

X = X[selected_features]
Y = train_df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

models = {
    "LR": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(kernel='sigmoid'),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(n_estimators=70),
    "XGB": XGBClassifier(n_estimators=70),
    "Naive Bayes": GaussianNB()
}

for name, model in models.items():
    print(f'Training Model {name} \n--------------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Training Accuracy: {model.score(X_train, y_train)}')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Testing Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print('-' * 30)
    print(classification_report(y_test, y_pred))
