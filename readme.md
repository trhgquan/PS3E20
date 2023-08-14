# PS3E20 - Predict CO2 Emissions in Rwanda
[Kaggle Playground Series - Season 3 Episode 20](https://www.kaggle.com/competitions/playground-series-s3e20)

The objective of this challenge is to create a machine learning models using open-source CO2 emissions data from Sentinel-5P satellite observations to predict future carbon emissions.

**Evaluation**: Root Mean Squared Error (RMSE)

**Final result**: TBD

## Timeline & Result

[View the timeline](timeline.md)

### Result

| Day        | Version               | Model Baseline                                                             | Features         | Training[^2] | Public testing[^3] | Private testing |
| ---------- | --------------------- | -------------------------------------------------------------------------- | ---------------- | ------------ | ------------------ | --------------- |
| 06/08/2023 | [v1](code/ps3e20.py)  | `sklearn.linear_model.LinearRegression`                                    | [^1]             | 142.25429    | 4851.07446         |                 |
| 06/08/2023 | [v3](code/ps3e20.py)  | `sklearn.neural_network.MLPRegressor`                                      | [^7]             | 141.67652    | 166.10065          |                 |
| 06/08/2023 | [v5](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor`                                       | [^4]             | 15.09919     | 33.35922           |                 |
| 06/08/2023 | [v6](code/ps3e20.py)  | `sklearn.ensemble.RandomForestRegressor` with default estimators           | [^4]             | 15.69964     | 33.05568           |                 |
| 07/08/2023 | [v7](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor`                                       | [^4]             | 11.48310     | 31.15227           |                 |
| 07/08/2023 | [v8](code/ps3e20.py)  | `sklearn.tree.DecisionTreeRegressor` & `sklearn.ensemble.BaggingRegressor` | [^4][^6]         | 11.80345     | 31.66813           |                 |
| 07/08/2023 | [v10](code/ps3e20.py) | `xgboost.XGBRegressor` & `sklearn.ensemble.BaggingRegressor`               | [^4][^6]         | 16.64857     | 34.20177           |                 |
| 07/08/2023 | [v11](code/ps3e20.py) | `sklearn.ensemble.RandomForestRegressor` with 2000 estimators              | [^4][^5][^6]     | **4.612114** | 31.06316           |                 |
| 08/08/2023 | [v12](code/ps3e20.py) | `sklearn.ensemble.RandomForestRegressor`, 2000 estimators                  | [^4][^5][^6][^7] | 11.07621     | 28.09778           |                 |
| 08/08/2023 | [v16](code/ps3e20.py) | `sklearn.tree.DecisionTreeRegressor`                                       | [^4][^5][^6][^7] | 11.98245     | 29.09904           |                 |
| 12/08/2023 | [v17](code/ps3e20.py) | `sklearn.tree.DecisionTreeRegressor`                                       | [^4][^5][^6][^7] | 11.07621     | 26.84726           |                 |
| 13/08/2023 | [v18](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021                                | [^1][^6]         | N/A          | 26.25738           |                 |
| 13/08/2023 | [v19](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.992           | [^1][^6]         | N/A          | 26.04316           |                 |
| 13/08/2023 | [v20](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.07            | [^1][^6]         | N/A          | 31.54728           |                 |
| 13/08/2023 | [v21](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021                                | [^1]             | N/A          | 23.02231           |                 |
| 13/08/2023 | [v22](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.992           | [^1]             | N/A          | **22.97095**       |                 |
| 14/08/2023 | [v23](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.07            | [^1]             | N/A          | 27.44515           |                 |
| 14/08/2023 | [v24](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 0.972           | [^1]             | N/A          | 23.29234           |                 |
| 14/08/2023 | [v25](code/ps3e20.py) | None + max emissions from 2019, 2020 & 2021 + magic number 1.007           | [^1]             | N/A          | 23.15143           |                 |
| 14/08/2023 | [v26](code/ps3e20.py) | None + using emissions from 2019 + magic number 1.07                       | [^1]             | N/A          | 37.56741           |                 |
| 14/08/2023 | [v27](code/ps3e20.py) | None + max emissions from 2019 + 2020 + magic number .992                  | [^1]             | N/A          | 29.78338           |                 |

[^1]: All, except the `emission` as the prediction variable. Also, the `year` variable is id-encoded.
[^4]: `latitude`, `longitude`, `week_no`
[^5]: with `year` variable.
[^6]: covid `emissions` normalized.
[^7]: All except `year`, `emission = max(0, emission)`
[^8]: claimed in the compact trick: multiplying the result with 1.06 will somehow boost the result.
[^2]: on the full training dataframe
[^3]: on the public testing dataframe

## LICENSE
This project is licensed under [The GNU GPL v3.0](LICENSE)