# PS3E20 - Predict CO2 Emissions in Rwanda
[Kaggle Playground Series - Season 3 Episode 20](https://www.kaggle.com/competitions/playground-series-s3e20)

The objective of this challenge is to create a machine learning models using open-source CO2 emissions data from Sentinel-5P satellite observations to predict future carbon emissions.

**Evaluation**: Root Mean Squared Error (RMSE)

**Final result**: TBD

## Timeline & Result

### Timeline

Day 1 - 06/08/2023

- [x] Remove the `year` column (v0).
- [x] ~~Using GridSearchCV and MLPRegressor (v2)~~.
- [x] Predictions should only in positive real (v3). 
- [x] [5 dimensions are enough](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429278) (v3).
- [x] Using DecisionTreeRegressor as a new baseline (v5).
- [x] Using RandomForestRegressor with GridSearchCV (v5).

Day 2 - 07/08/2023

- [ ] ~~[Volcano eruption](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429232) (v7)~~.
- [x] [COVID-19](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429622) (v7).
- [x] [Using DecisionTreeRegressor & BaggingRegressor (for ensemble)](https://www.kaggle.com/code/johnsmith44/ps3e20-co2-emissions-in-rwanda-compact-trick) (v8). 
- [x] Using XGBRegressor & BaggingRegressor (for ensemble) (v10).
- [x] [Multiplying a magic constant.](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429675)
	- According the post, using a magic constant will boost the RMSE from 30.x to 2x. I'll experimenting this (but to be honest, that doesn't make any sense at all).

Day 3 - 08/08/2023
- [ ] Try using k-fold to re-estimate all previous approaches.
- [x] Try using magic constant with best approach (v12).

### Result

| Day        | Version                           | Model Baseline                                                             | Features         | Training[^2] | Public testing[^3] | Private testing |
| ---------- | --------------------------------- | -------------------------------------------------------------------------- | ---------------- | ------------ | ------------------ | --------------- |
| 06/08/2023 | [v1](notebooks/ps3e20-v01.ipynb)  | `sklearn.linear_model.LinearRegression`                                    | [^1]             | 142.25429    | 4851.07446         |                 |
| 06/08/2023 | [v3](notebooks/ps3e20-v03.ipynb)  | `sklearn.neural_network.MLPRegressor`                                      | [^7]             | 141.67652    | 166.10065          |                 |
| 06/08/2023 | [v5](notebooks/ps3e20-v05.ipynb)  | `sklearn.tree.DecisionTreeRegressor` with default estimators               | [^4]             | 15.09919     | 33.35922           |                 |
| 06/08/2023 | [v6](notebooks/ps3e20-v06.ipynb)  | `sklearn.tree.RandomForestRegressor` with default estimators               | [^4]             | 15.69964     | 33.05568           |                 |
| 07/08/2023 | [v7](notebooks/ps3e20-v07.ipynb)  | `sklearn.tree.DecisionTreeRegressor` with default estimators               | [^4]             | 11.48310     | 31.15227           |                 |
| 07/08/2023 | [v8](notebooks/ps3e20-v08.ipynb)  | `sklearn.tree.DecisionTreeRegressor` & `sklearn.ensemble.BaggingRegressor` | [^4][^6]         | 11.80345     | 31.66813           |                 |
| 07/08/2023 | [v10](notebooks/ps3e20-v10.ipynb) | `xgboost.XGBRegressor` & `sklearn.ensemble.BaggingRegressor`               | [^4][^6]         | 16.64857     | 34.20177           |                 |
| 07/08/2023 | [v11](notebooks/ps3e20-v11.ipynb) | `sklearn.tree.DecisionTreeRegressor` with 2000 estimators                  | [^4][^5][^6]     | **4.612114** | 31.06316           |                 |
| 08/08/2023 | [v12](code/ps3e20.py)             | `sklearn.tree.DecisionTreeRegressor`, 2000 estimators                      | [^4][^5][^6][^7] | 11.07621     | **28.09778**       |                 |

[^1]: aLL, except the `emission` as the prediction variable. Also, the `year` variable is id-encoded.
[^4]: `latitude`, `longitude`, `week_no`
[^5]: with `year` variable.
[^6]: covid `emissions` normalized.
[^7]: All except `year`, `emission = max(0, emission)`
[^8]: clams in the compact trick to multiply the result with 1.06 will somehow boost the result.
[^2]: on the full training dataframe
[^3]: on the public testing dataframe

## LICENSE
This project is licensed under [The GNU GPL v3.0](LICENSE)