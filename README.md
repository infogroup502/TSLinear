# TSLiner
This repository contains the official implementation for the paper [Universal Multi-scale Linear Representation Learning for Multivariate Time Series]().

## Requirements
The recommended requirements for TSG2L are specified as follows:

- Python 3.7
- torch==1.12.0
- numpy==1.21.6
- pandas==1.0.1
- scikit-learn==0.24.2
- scipy==1.7.3

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data 
The datasets can be obtained and put into datasets/ folder in the following way:
### forecast
- [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/forecast/ETTh1.csv`, `datasets/forecast/ETTh2.csv` and `datasets/forecast/ETTm1.csv`.
- [Weather](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)should be placed at `datasets/forecast/Weather.csv`.
- [Airquality](https://archive.ics.uci.edu/dataset/360/air+quality)should be placed at `datasets/forecast/Airquality.csv`.
### classification
- [128UCRdatasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.
Such as [Chinatown](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/Chinatown), [ItalyPowerDemand](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ItalyPowerDemand), [RacketSports](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/RacketSports), [ArrowHead](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ArrowHead), [SharePriceIncrease](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/SharePriceIncrease).
### anomaly
- [MSL](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/anomaly/MSL.csv`.
- [SMD](https://github.com/NetManAIOps/OmniAnomaly) should be placed at `datasets/anomaly/SMD.csv`.
- [SMAP](https://en.wikipedia.org/wiki/Soil_Moisture_Active_Passive) should be placed at `datasets/anomaly/SMAP.csv`.
- [MBA](https://paperswithcode.com/dataset/mit-bih-arrhythmia-database) should be placed at `datasets/anomaly/MBA.csv`.
- [SwaT](https://drive.google.com/drive/folders/1ABZKdclka3e2NXBSxS9z2YF59p7g2Y5I) should be placed at `datasets/anomaly/SwaT.csv`.
## Usage
To train and evaluate TSG2L on a dataset, run the following command:
```bash
python train_forecast.py --dataset <dataset_name>  --run_name <run_name> --loader <loader> --gpu <gpu> 
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| gpu | The gpu no. used for training and inference (defaults to 0) |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 
**Scripts:** The scripts for reproduction are provided in scripts/ folder.
