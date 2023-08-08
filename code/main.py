import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

TRAIN_CSV = "/kaggle/input/playground-series-s3e20/train.csv"
TEST_CSV = "/kaggle/input/playground-series-s3e20/test.csv"
SUBMISSION_CSV = "submission.csv"


def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def v1(df, test_df):
    X = df.drop(["ID_LAT_LON_YEAR_WEEK", "emission"], axis=1)
    y = df["emission"]

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    pred_dev = lr_model.predict(X)
    print(f"Training rmse = {rmse(y, pred_dev)}")

    test_df = test_df.fillna(0)
    pred = lr_model.predict(test_df.drop(["ID_LAT_LON_YEAR_WEEK"], axis=1))

    return pred


def v3(df, test_df):
    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    mlp_model = MLPRegressor(random_state=42, verbose=True)
    mlp_model.fit(X, y)

    pred_dev = mlp_model.predict(X)
    print(f"Training rmse = {rmse(y, pred_dev)}")

    test_df = test_df.fillna(0)
    pred = mlp_model.predict(test_df.drop(["ID_LAT_LON_YEAR_WEEK"], axis=1))

    pred[pred < 0] = 0

    return pred


def v5(df, test_df):
    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)

    pred_dev = dt_model.predict(X)
    print(f"Training rmse = {rmse(y, pred_dev)}")

    test_df = test_df.fillna(0)
    pred = dt_model.predict(test_df[["latitude", "longitude", "week_no"]])

    pred[pred < 0] = 0

    return pred


def v6(df, test_df):
    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)

    pred_dev_single = rf_model.predict(X)
    print(f"Training rmse (RF) = {rmse(y, pred_dev_single)}")

    # Grid search
    grid_param = {
        "n_estimators": [100, 200],
        "min_samples_split": [8, 10],
        "min_samples_leaf": [3, 4, 5],
        "max_depth": [80, 90]
    }

    cv = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=grid_param,
        verbose=4,
        scoring="neg_root_mean_squared_error"
    )

    cv.fit(X, y)

    pred_cv = cv.predict(X)
    print(f"Training rmse (RF + GridSearch) = {rmse(y, pred_cv)}")

    test_df = test_df.fillna(0)
    pred = cv.predict(test_df[["latitude", "longitude", "week_no"]])

    pred[pred < 0] = 0

    return pred


def v7(df, test_df):
    # Calculate the average weekly emissions for non-virus years (2019 and 2021)
    avg_emission_non_virus = df[df['year'].isin((2019, 2021))].groupby('week_no')[
        'emission'].mean()

    # Calculate the average weekly emissions for virus year (2020)
    avg_emission_virus = df[df['year'] == 2020].groupby('week_no')[
        'emission'].mean()

    # Calculate the ratios for each week
    ratios_for_weeks = avg_emission_non_virus/avg_emission_virus

    # Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
    df.loc[df['year'] == 2020,
           'emission'] *= df['week_no'].map(ratios_for_weeks)

    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)

    pred_dev = dt_model.predict(X)
    print(f"Training rmse  = {rmse(y, pred_dev)}")

    test_df = test_df.fillna(0)
    pred = dt_model.predict(test_df[["latitude", "longitude", "week_no"]])
    pred[pred < 0] = 0

    return pred


def v8(df, test_df):
    # Calculate the average weekly emissions for non-virus years (2019 and 2021)
    avg_emission_non_virus = df[df['year'].isin((2019, 2021))].groupby('week_no')[
        'emission'].mean()

    # Calculate the average weekly emissions for virus year (2020)
    avg_emission_virus = df[df['year'] == 2020].groupby('week_no')[
        'emission'].mean()

    # Calculate the ratios for each week
    ratios_for_weeks = avg_emission_non_virus/avg_emission_virus

    # Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
    df.loc[df['year'] == 2020,
           'emission'] *= df['week_no'].map(ratios_for_weeks)

    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)

    pred_dev_single = dt_model.predict(X)
    print(f"Training rmse (single) = {rmse(y, pred_dev_single)}")

    # Bagging model
    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(), random_state=42)
    bagging_model.fit(X, y)

    dev_bagging_pred = bagging_model.predict(X)
    print(f"Training rmse (bagging) = {rmse(y, dev_bagging_pred)}")

    test_df = test_df.fillna(0)
    pred = bagging_model.predict(test_df[["latitude", "longitude", "week_no"]])

    pred[pred < 0] = 0

    return pred


def v10(df, test_df):
    # Calculate the average weekly emissions for non-virus years (2019 and 2021)
    avg_emission_non_virus = df[df['year'].isin((2019, 2021))].groupby('week_no')[
        'emission'].mean()

    # Calculate the average weekly emissions for virus year (2020)
    avg_emission_virus = df[df['year'] == 2020].groupby('week_no')[
        'emission'].mean()

    # Calculate the ratios for each week
    ratios_for_weeks = avg_emission_non_virus/avg_emission_virus

    # Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
    df.loc[df['year'] == 2020,
           'emission'] *= df['week_no'].map(ratios_for_weeks)

    X = df[["latitude", "longitude", "week_no"]]
    y = df["emission"]

    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X, y)

    dev_pred = xgb_model.predict(X)
    print(f"Training rmse (single) = {rmse(y, dev_pred)}")

    # Bagging model
    bagging_model = BaggingRegressor(estimator=XGBRegressor(), random_state=42)
    bagging_model.fit(X, y)

    dev_bagging_pred = bagging_model.predict(X)
    print(f"Training rmse (single) = {rmse(y, dev_bagging_pred)}")

    test_df = test_df.fillna(0)
    pred = bagging_model.predict(test_df[["latitude", "longitude", "week_no"]])
    pred[pred < 0] = 0

    return pred


def v11(df, test_df):
    # Calculate the average weekly emissions for non-virus years (2019 and 2021)
    avg_emission_non_virus = df[df['year'].isin((2019, 2021))].groupby('week_no')[
        'emission'].mean()

    # Calculate the average weekly emissions for virus year (2020)
    avg_emission_virus = df[df['year'] == 2020].groupby('week_no')[
        'emission'].mean()

    # Calculate the ratios for each week
    ratios_for_weeks = avg_emission_non_virus/avg_emission_virus

    # Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
    df.loc[df['year'] == 2020,
           'emission'] *= df['week_no'].map(ratios_for_weeks)

    X = df[["latitude", "longitude", "week_no", "year"]]
    y = df["emission"]

    rf_model = RandomForestRegressor(n_estimators=2000, random_state=42)
    rf_model.fit(X, y)

    dev_pred = rf_model.predict(X)
    print(f"Training rmse = {rmse(y, dev_pred)}")

    MAGIC = 1.06
    print(f"Training rmse (with magic) = {rmse(y, dev_pred * MAGIC)}")

    pred = rf_model.predict(
        test_df[["latitude", "longitude", "week_no", "year"]])
    pred[pred < 0] = 0

    return pred


def v12(df, test_df):
    # Calculate the average weekly emissions for non-virus years (2019 and 2021)
    avg_emission_non_virus = df[df['year'].isin((2019, 2021))].groupby('week_no')[
        'emission'].mean()

    # Calculate the average weekly emissions for virus year (2020)
    avg_emission_virus = df[df['year'] == 2020].groupby('week_no')[
        'emission'].mean()

    # Calculate the ratios for each week
    ratios_for_weeks = avg_emission_non_virus/avg_emission_virus

    # Multiply the emission column for each row in 2020 by the corresponding ratio for the week of that row
    df.loc[df['year'] == 2020,
           'emission'] *= df['week_no'].map(ratios_for_weeks)

    X = df[["latitude", "longitude", "week_no", "year"]]
    y = df["emission"]

    rf_model = RandomForestRegressor(n_estimators=2000, random_state=42)
    rf_model.fit(X, y)

    dev_pred = rf_model.predict(X)
    print(f"Training rmse = {rmse(y, dev_pred)}")

    MAGIC = 1.06
    print(f"Training rmse (with magic) = {rmse(y, dev_pred * MAGIC)}")

    pred = rf_model.predict(
        test_df[["latitude", "longitude", "week_no", "year"]])
    pred[pred < 0] = 0

    pred = pred * MAGIC

    return pred


def main():
    df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # pred = v1(df = df, test_df = test_df)
    # pred = v3(df = df, test_df = test_df)
    # pred = v5(df = df, test_df = test_df)
    # pred = v6(df = df, test_df = test_df)
    # pred = v7(df = df, test_df = test_df)
    # pred = v8(df = df, test_df = test_df)
    # pred = v10(df = df, test_df = test_df)
    # pred = v11(df = df, test_df = test_df)
    # pred = v12(df=df, test_df=test_df)
	
    submission_df = pd.DataFrame(zip(test_df["ID_LAT_LON_YEAR_WEEK"], pred), columns=[
                                 "ID_LAT_LON_YEAR_WEEK", "emission"])

    submission_df.to_csv(SUBMISSION_CSV, index=False)


if __name__ == "__main__":
    main()
