# PS3E20 - Predict CO2 Emissions in Rwanda
[Kaggle Playground Series - Season 3 Episode 20](https://www.kaggle.com/competitions/playground-series-s3e20)


## Introductions
The objective of this challenge is to create a machine learning models using open-source CO2 emissions data from Sentinel-5P satellite observations to predict future carbon emissions.

## Actions
- [x] Remove the `year` column (v0).
- [x] ~~Using GridSearchCV and MLPRegressor (v2)~~.
- [x] Predictions should only in positive real (v3). 
- [x] [5 dimensions are enough](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429278) (v3).
- [x] Using DecisionTreeRegressor as a new baseline (v5).
- [x] Using RandomForestRegressor with GridSearchCV (v5).

## Result

| Day        | Version | Model Baseline                          | Features                                         | RMSE (train) | RMSE (test)  |
| ---------- | ------- | --------------------------------------- | ------------------------------------------------ | ------------ | ------------ |
| 06/08/2023 | `v1`    | `sklearn.linear_model.LinearRegression` | All[^1]                                          | 142.25429    | 4851.07446   |
| 06/08/2023 | `v3`    | `sklearn.neural_network.MLPRegressor`   | All except `year`                                | N/A          | 168.39246    |
| 06/08/2023 | `v4`    | `sklearn.neural_network.MLPRegressor`   | All except `year`, `emission = max(0, emission)` | 141.67652    | 166.10065    |
| 06/08/2023 | `v5`    | `sklearn.tree.DecisionTreeRegressor`    | `latitude`, `longitude` and `week_no`            | **15.09919** | **33.35922** |
| 06/08/2023 | `v6`    | `sklearn.tree.RandomForestRegressor`    | `latitude`, `longitude` and `week_no`            |              |              |

[^1]: except the `emission` as the prediction variable. Also, the `year` variable is encoded to the range `[1, len(unique(year))]`

## LICENSE
This project is licensed under [The GNU GPL v3.0](LICENSE)