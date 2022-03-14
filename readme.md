# Monthly user costs prediction.

## Data Description
- data_train_transactions.csv - client's costs 2019
	- party_rk - unique client id
	- account_rk - account id
	- transaction_dttm - cost date
	- transaction_amt_rur - sum in rubles 
	- category - категория траты
- data_test_transactions.csv - client's costs january 2020. For validation.
- data_party_x_socdem.csv - socio-demographic characteristics of clients
- data_story_logs.csv - history of customer interaction with "stories" in the bank application (likes, dislikes).
	- party_rk - unique client id
	- date_time - date of interaction
	- story_id - unique id of story
	- event - customer reaction to the story (like, dislike)
- data_story_texts.csv - contains texts from stories
	- story_id - unique story id 
	- name - story name
	- story_text - story text


The raw data located in `data/raw_data`

## Metric
As metric have choosen accuracy to +-N% from real costs, by default I used 20%. Also as alternative metric and loss training GB I used MAE.

## Usage
The script performs data preparation for training, model training, validation, model saving and conversion to ONNX.

Parameters in  `/configs/config.cfg`
Two modules in config, **data** и **train** , data for data preparation and saving. Train for model's training.
### Data
* **socdem, story_logs, story_texts, train_tr, test_tr** - path to csv file (str)
* **use_texts** - use text data (bool)
* **save_prepared** - save data after preprocessing (bool). Saving to `/data/processed_data`
* **load_prepared** - use preprocessed data (bool)

### Train
* **num_iterations** - number of iterations for selection of parameters in optuna (int)
* **main_metric** - optimization metric. (**accuracy** or **mae**) (str)
* **convert_to_onnx** - convert to ONNX (bool)
* **n_estimators** - GB model number of estimators, final model be n_estimators * 3 (int)
* **metric_thres** - N/100 for accuracy metric (float)

### Run
* `pip install -r requirements.txt`
* `python3 run.py -c ./configs/config.cfg -v 1`
         
The path to the config is passed to the script (required parameter) and the verbose -v parameter, 1 - will display prints during the process, 0 - will hide them. Not required parameter, default 1.

Models saves to `/models`

