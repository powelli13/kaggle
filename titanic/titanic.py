import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# load data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_test.columns)


# clean data
def fare_categorize(v):
    if v <= 15:
        return 1

    if v > 15 and v <= 25:
        return 2

    if v > 25:
        return 3


def age_categorize(v):
    if v <= 8:
        return 1

    if v > 8 and v <= 15:
        return 2

    if v > 15 and v <= 25:
        return 3

    if v > 25 and v <= 40:
        return 4

    if v > 40 and v <= 60:
        return 5

    if v > 60:
        return 6


for ds in [df_train, df_test]:
    ds['Sex'] = ds['Sex'].map({'female': 0, 'male': 1})

    # TODO is there any way to get the means and use the sex mean for the NaN replacement
    # female_mean = ds.loc[ds['Sex'] == 0, 'Age'].mean()
    # male_mean = ds.loc[ds['Sex'] == 1, 'Age'].mean()
    #
    # ds.loc[ds['Sex'] == 0, 'Age'].fillna(female_mean, inplace=True)
    # ds.loc[ds['Sex'] == 1, 'Age'].fillna(male_mean, inplace=True)

    ds['Age'].fillna(ds['Age'].mean(), inplace=True)

    # ds['Age'] = ds['Age'].apply(age_categorize)

    # maybe take class into account for fare
    # survived_mean = ds.loc[ds['Survived'] == 1, 'Fare'].mean()
    # not_survived_mean = ds.loc[ds['Survived'] == 0, 'Fare'].mean()
    #
    # ds.loc[ds['Survived'] == 1, 'Fare'].fillna(survived_mean, inplace=True)
    # ds.loc[ds['Survived'] == 0, 'Fare'].fillna(not_survived_mean, inplace=True)


    ds['Pclass'].fillna(2)

# messing with fare categories
survived_mean = df_train.loc[df_train['Survived'] == 1, 'Fare'].mean()
not_survived_mean = df_train.loc[df_train['Survived'] == 0, 'Fare'].mean()

df_train.loc[df_train['Survived'] == 1, 'Fare'].fillna(survived_mean, inplace=True)
df_train.loc[df_train['Survived'] == 0, 'Fare'].fillna(not_survived_mean, inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)

for ds in [df_train, df_test]:
    # ds.loc[ds['Fare'] <= 15, 'Fare'] = 1
    # ds.loc[ds['Fare'] > 15 and ds['Fare'] < 25, 'Fare'] = 2
    # ds.loc[ds['Fare'] > 25, 'Fare'] = 3
    ds['Fare'] = ds['Fare'].apply(fare_categorize)

# predict
predictors = [
    'Sex',
    'Age',
    'Pclass',
    'Fare',
    'SibSp',
    'Parch'
]

# clf = RandomForestClassifier(n_estimators=200, min_samples_split=50)
# clf = KNeighborsClassifier(n_neighbors=8)
clf = AdaBoostClassifier()
clf.fit(df_train[predictors], df_train['Survived'])

predictions = clf.predict(df_test[predictors])
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission_adaboost_812.csv', index=False)

# check_col = 'Fare'
# x = df_train[check_col]
# y = df_train['PassengerId']

# plt.plot(x, y, 'o')
# plt.xlabel(check_col + ' Feature')
# plt.ylabel('Survived')
# plt.axis([0, 100, 0, 100])
#
# plt.legend()
# plt.show()



acc = cross_val_score(clf, df_train[predictors], df_train['Survived'], cv=5)
print("Accuracy {0:.2%} (+/- {1:.2%})".format(acc.mean(), acc.std() * 2))
