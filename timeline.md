# Timeline and Result

## Timeline

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

- ~~[ ] Try using k-fold to re-estimate all previous approaches.~~
- [x] Try using magic constant with best approach (v12).

Day 4 - 12/08/2023

- [x] Using [a magic to "solve" the bug in submission - caused by a week shift](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429717).

Day 5 - 13/08/2023

- [x] Using a Non-ML solution (v17 - v23).
	- Without a machine learning model, using maximum emission records from 2019-2020-2021 with a magic number can boost result to 23.

Day 6 - 14/08/2023

- [x] Try using lower magic.

Day 7 - 15/08/2023

- [x] Try using something other than max emission in predictions.

## Result

| Day        | Version               | Model Baseline                                                             | Features         | Training[^2] | Public testing[^3] | Private testing |
| ---------- | --------------------- | -------------------------------------------------------------------------- | ---------------- | ------------ | ------------------ | --------------- |
| 06/08/2023 | [v1](code/ps3e20.py)  | `sklearn.linear_model.LinearRegression`                                    | [^1]             | 142.25429    | 4851.07446         | 4870.54694      |
| 06/08/2023 | [v3](code/ps3e20.py)  | `sklearn.neural_network.MLPRegressor`                                      | [^7]             | 141.67652    | 166.10065          | 147.04415       |
| 06/08/2023 | [v5](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor`                                       | [^4]             | 15.09919     | 33.35922           | 20.49438        |
| 06/08/2023 | [v6](code/ps3e20.py)  | `sklearn.ensemble.RandomForestRegressor` with default estimators           | [^4]             | 15.69964     | 33.05568           | 20.41293        |
| 07/08/2023 | [v7](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor`                                       | [^4]             | 11.48310     | 31.15227           | 14.86936        |
| 07/08/2023 | [v8](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor` & `sklearn.ensemble.BaggingRegressor` | [^4][^6]         | 11.80345     | 31.66813           | 14.97777        |
| 07/08/2023 | [v10](code/ps3e20.py) | `xgboost.XGBRegressor` & `sklearn.ensemble.BaggingRegressor`               | [^4][^6]         | 16.64857     | 34.20177           | 17.98049        |
| 07/08/2023 | [v11](code/ps3e20.py) | `sklearn.ensemble.RandomForestRegressor` with 2000 estimators              | [^4][^5][^6]     | **4.612114** | 31.06316           | 17.31996        |
| 08/08/2023 | [v12](code/ps3e20.py) | `sklearn.ensemble.RandomForestRegressor`, 2000 estimators                  | [^4][^5][^6][^7] | 11.07621     | 28.09778           | 12.38870        |
| 08/08/2023 | [v16](code/ps3e20.py) | `sklearn.tree.DecisionTreeRegressor`                                       | [^4][^5][^6][^7] | 11.98245     | 29.09904           | 13.58976        |
| 12/08/2023 | [v17](code/ps3e20.py) | `sklearn.tree.DecisionTreeRegressor`                                       | [^4][^5][^6][^7] | 11.07621     | 26.84726           | **12.17540**    |
| 13/08/2023 | [v18](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021                                | [^1][^6]         | N/A          | 26.25738           | 14.08058        |
| 13/08/2023 | [v19](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.992           | [^1][^6]         | N/A          | 26.04316           | 14.31248        |
| 13/08/2023 | [v20](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.07            | [^1][^6]         | N/A          | 31.54728           | 17.42555        |
| 13/08/2023 | [v21](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021                                | [^1]             | N/A          | 23.02231           | 14.96944        |
| 13/08/2023 | [v22](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.992           | [^1]             | N/A          | **22.97095**       | 15.60472        |
| 14/08/2023 | [v23](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.07            | [^1]             | N/A          | 27.44515           | 14.42761        |
| 14/08/2023 | [v24](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.972           | [^1]             | N/A          | 23.29234           | 17.54602        |
| 14/08/2023 | [v25](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.007           | [^1]             | N/A          | 23.15143           | 14.49313        |
| 14/08/2023 | [v26](code/ps3e20.py) | None + using emissions from 2019 + magic number 1.07                       | [^1]             | N/A          | 37.56741           | 13.09110        |
| 14/08/2023 | [v27](code/ps3e20.py) | None + mean emissions from 2019, 2020 & 2021 + magic number 1.07           | [^1]             | N/A          | 29.78338           | 17.32771        |

[^1]: All, except the `emission` as the prediction variable. Also, the `year` variable is id-encoded.
[^4]: `latitude`, `longitude`, `week_no`
[^5]: with `year` variable.
[^6]: covid `emissions` normalized.
[^7]: All except `year`, `emission = max(0, emission)`
[^8]: claimed in the compact trick: multiplying the result with 1.06 will somehow boost the result.
[^2]: on the full training dataframe
[^3]: on the public testing dataframe
