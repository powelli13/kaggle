import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

combine = [train_df, test_df]


def predict():
    #predictors = ["YearBuilt", "YearRemodAdd", "HouseStyle", "OverallCond", "ExterQual", "GarageCars", "SaleCondition", "KitchenQual", "RoofStyle", "WoodDeckSF", "3SsnPorch", "ScreenPorch"]
    predictors = ["YearBuilt", "YearRemodAdd", "HouseStyle", "OverallQual", "ExterQual", "GarageCustom",
                  "SaleCondition", "KitchenQual", "RoofStyle", "TotalDeckSize", "TotalBaths",
                  "Neighborhood", "Foundation", "BedroomAbvGr", "CentralAir", "GrLivArea", "SaleType", "HeatingQC",
                  "Electrical", "MasVnrArea", "FireplaceCustom", "Functional", "PavedDrive", "LandSlope", "PoolArea",
                  "BldgType", "BsmtCustom"]
    #predictors = ['YearBuilt', 'OverallCond']

    # clean up data
    kitchen_qual_map = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1}
    
    style_map = {"2Story": 2, "1Story": 4, "1.5Fin": 6, "1.5Unf": 7, "SFoyer": 8, "SLvl": 5, "2.5Unf": 3, "2.5Fin": 1}

    exterqual_map = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4}
    salecond_map = {"Partial": 1, "Normal": 2, "Alloca": 3, "Family": 4, "Abnorml": 5, "AdjLand": 6}

    garage_qual_map = {"TA": 0.75, "Fa": 1.25, "Gd": 1.5, "Ex": 2.0, "Po": 1.0}
    garage_finish_map = {"RFn": 1.0, "Unf": 1.0, "Fin": 1.25}

    roof_style_map = {"Shed": 1, "Hip": 2, "Flat": 3, "Mansard": 4, "Gable": 5, "Gambrel": 6}
    foundation_map = {"PConc": 1, "CBlock": 2, "BrkTil": 3, "Wood": 4, "Slab": 4, "Stone": 4}

    sale_type_map = {"WD": 1, "New": 2, "COD": 3, "ConLD": 4, "ConLI": 5, "CWD": 6, "ConLw": 7, "Con": 8, "Oth": 9}

    heating_qc_map = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
    central_air_map = {"Y": 1, "N": 2}

    electrical_map = {"SBrkr": 1, "FuseF": 2, "FuseA": 3, "FuseP": 4, "Mix": 5}

    fireplace_qual_map = {"Ex": 2.0, "Gd": 1.5, "Fa": 1.25, "Po": 1.0, "TA": 0.75}

    paved_drive_map = {"Y": 2, "N": 1, "P": 0}

    land_slope_map = {"Gtl": 1, "Mod": 2, "Sev": 3}

    bsmt_cond_map = {"Gd": 1.5, "Fa": 1.25, "Po": 1.0, "TA": 0.75}
    bsmt_qual_map = {"Ex": 2.0, "Gd": 1.5, "Fa": 1.25, "TA": 0.75}

    bldg_type_map = {}
    i = 1
    for n in train_df["BldgType"].unique():
        bldg_type_map[n] = i
        i += 1

    functional_map = {}
    i = 1

    for n in train_df["Functional"].unique():
        functional_map[n] = i
        i += 1

    neighborhood_map = {}
    i = 1

    for n in train_df["Neighborhood"].unique():
        neighborhood_map[n] = i
        i += 1

    
    for ds in combine:
        ds["GarageCars"].fillna(ds["GarageCars"].dropna().median(), inplace=True)

        ds["GarageFinish"] = ds["GarageFinish"].map(garage_finish_map)
        ds["GarageFinish"].fillna(1.0, inplace=True)

        ds["GarageQual"] = ds["GarageQual"].map(garage_qual_map)
        ds["GarageQual"].fillna(1.0, inplace=True)

        ds["GarageArea"].fillna(0.0, inplace=True)

        ds["GarageCustom"] = ds["GarageCars"] * ds["GarageQual"] * ds["GarageArea"]# * ds["GarageFinish"]

        ds["FireplaceQu"] = ds["FireplaceQu"].map(fireplace_qual_map)
        ds["FireplaceQu"].fillna(1.0, inplace=True)

        ds["FireplaceCustom"] = ds["Fireplaces"] * ds["FireplaceQu"]

        ds["MasVnrArea"].fillna(0.0, inplace=True)

        ds["KitchenQual"] = ds["KitchenQual"].map(kitchen_qual_map)
        ds["KitchenQual"].fillna(ds["KitchenQual"].dropna().median(), inplace=True)

        ds["HouseStyle"] = ds["HouseStyle"].map(style_map)
        ds["HouseStyle"] = ds["HouseStyle"].fillna(0)

        ds["CentralAir"] = ds["CentralAir"].map(central_air_map)
        ds["HeatingQC"] = ds["HeatingQC"].map(heating_qc_map)

        ds["Electrical"] = ds["Electrical"].map(electrical_map)
        ds["Electrical"].fillna(5, inplace=True)

        ds["ExterQual"] = ds["ExterQual"].map(exterqual_map)
        ds["ExterQual"].fillna(3, inplace=True)

        ds["SaleCondition"] = ds["SaleCondition"].map(salecond_map)
        ds["SaleCondition"].fillna(2, inplace=True)

        ds["RoofStyle"] = ds["RoofStyle"].map(roof_style_map)
        ds["RoofStyle"].fillna(1, inplace=True)

        ds["Neighborhood"] = ds["Neighborhood"].map(neighborhood_map)
        
        ds["Foundation"] = ds["Foundation"].map(foundation_map)
        
        ds["TotalDeckSize"] = ds["WoodDeckSF"] + ds["3SsnPorch"] + ds["ScreenPorch"] + ds["EnclosedPorch"]
    
        ds["TotalBaths"] = ds["BsmtFullBath"] + (0.5*ds["BsmtHalfBath"]) + ds["FullBath"] + (0.5*ds["HalfBath"])
        ds["TotalBaths"].fillna(ds["TotalBaths"].dropna().median(), inplace=True)

        ds["SaleType"] = ds["SaleType"].map(sale_type_map)
        ds["SaleType"].fillna(9, inplace=True)

        ds["Functional"] = ds["Functional"].map(functional_map)
        ds["Functional"].fillna(3, inplace=True)

        ds["PavedDrive"] = ds["PavedDrive"].map(paved_drive_map)

        ds["LandSlope"] = ds["LandSlope"].map(land_slope_map)

        ds["BldgType"] = ds["BldgType"].map(bldg_type_map)

        ds["BsmtCond"] = ds["BsmtCond"].map(bsmt_cond_map)
        ds["BsmtCond"].fillna(1.0, inplace=True)

        ds["BsmtQual"] = ds["BsmtQual"].map(bsmt_qual_map)
        ds["BsmtQual"].fillna(1.0, inplace=True)

        ds["TotalBsmtSF"].fillna(0.0, inplace=True)

        ds["BsmtCustom"] = ds["BsmtCond"] * ds["BsmtQual"] * ds["TotalBsmtSF"]

    # logistic reg
    t_train, t_test, a_train, a_test = train_test_split(train_df[predictors], train_df["SalePrice"], test_size=0.3)

    # used to find NaN values
    #for c in predictors:
    #    print("Column: {0} uniques are: {1}".format(c, test_df[c].unique()))

    alg = LinearRegression()

    alg.fit(train_df[predictors], train_df["SalePrice"])

    #alg.fit(t_train, a_train)
    '''
    preds = alg.predict(t_test)

    for p in preds:
        if p < 0.0:
            print("negative: {}".format(p))
    '''
    #score = mean_squared_error(a_test.tolist(), preds.tolist())
    
    #print("a_test shape: {!s}".format(a_test.shape))
    #print("preds shape: {!s}".format(preds.shape))

    #print(pd.Series(preds).describe())

    #preds_df = pd.DataFrame.from_dict({'Actual': pd.Series(a_test.tolist()), 'Predicted': pd.Series(preds.tolist())})

    #print(preds_df.describe())
    #print(preds_df.columns)
    #print(test_df.columns)

    #scores = alg.score(train_df[predictors], train_df['SalePrice'])

    #print(score)

    sale_predictions = alg.predict(test_df[predictors])
    
    # TODO this is terrible fix this better i think
    #m = len(sale_predictions)
    #for i in range(m):
    #    
    #    if sale_predictions[i] < 0.0:
    #        sale_predictions[i] = 0.0
    
    #print(sale_predictions)
    
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": sale_predictions})
    submission.to_csv('submission_0628.csv', index=False)
    
    scores = cross_val_score(alg, train_df[predictors], train_df["SalePrice"], cv=5)
    print("Accuracy {0:.2%} (+/- {1:.2%})".format(scores.mean(), scores.std() * 2))


def show(col_name):

    col_map = {}
    i = 1

    for n in train_df[col_name].unique():
        col_map[n] = i
        i += 1
        
    print(col_map)
        
    train_df[col_name] = train_df[col_name].map(col_map)

    x = train_df[col_name]

    y = train_df['SalePrice']

    plt.plot(x, y, 'o', label='House')
    plt.xlabel(col_name)
    plt.ylabel('Sale Price')
    #plt.axis([0, 500, 100000, 400000])

    plt.legend()
    plt.show()

if __name__ == "__main__":
    predict()
    #show("SaleType")
