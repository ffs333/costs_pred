import time
import pickle

import optuna
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

from core.metric import perc
from utils.utils import enablePrint, convert_to_onnx


def training(config, tr_data, tr_label, ev_data, ev_label):
    """
    Model training process
    :param config: train config
    :param tr_data: train data
    :param tr_label: train labels
    :param ev_data: eval data
    :param ev_label: eval labels
    :return:
    """
    mask = ev_data[~((ev_data.spends_mon0 == 0) & (ev_data.spends_mon1 == 0) & (ev_data.spends_mon2 == 0)
                     & (ev_data.spends_mon3 == 0) & (ev_data.spends_mon4 == 0) &
                     (ev_data.spends_mon5 == 0) & (ev_data.spends_mon6 == 0) & (ev_data.spends_mon7 == 0))].index

    def objective(trial):
        """
        Function to optimize model params
        :param trial:(optuna instance) iteration
        :return:(float) metric of model iteration
        """

        param = {
            'n_jobs': -1,
            'objective': 'mae',
            'random_state': 42,
            'n_estimators': config.n_estimators,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 3),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 13, 15, 17, 19, 21, 25, 28, 31, 35]),
            'num_leaves': trial.suggest_int('num_leaves', 2, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 20000),
            'min_data_per_group': trial.suggest_int('min_data_per_group', 10, 20000)
        }
        model = LGBMRegressor(**param)
        st = time.time()
        model.fit(tr_data, tr_label, eval_set=[(ev_data, ev_label)],
                  callbacks=[early_stopping(60, verbose=0), log_evaluation(-1)])

        curr_preds = model.predict(ev_data)

        masked_preds = curr_preds.copy()
        masked_preds[mask] = 0

        curr_mae_metr = mean_absolute_error(ev_label, curr_preds)
        curr_mae_metr_masked = mean_absolute_error(ev_label, masked_preds)

        percc = perc(curr_preds, ev_label.values.ravel(), config.metric_thres)
        percc_masked = perc(masked_preds, ev_label.values.ravel(), config.metric_thres)

        print(f'MAE: {round(curr_mae_metr, 4)}')
        print(f'MAE masked: {round(curr_mae_metr_masked, 4)}')
        print(f'Accuraty +-{round(config.metric_thres * 100)}%: {round(percc, 5)}')
        print(f'Accuraty +-{round(config.metric_thres * 100)}% masked: {round(percc_masked, 5)}')
        print(f'Fit time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}\n')

        return percc if config.main_metric == 'accuracy' else curr_mae_metr

    print('Starting train parameters optimization process.\n'
          f'With main metric {config.main_metric}')
    optuna.logging.disable_default_handler()
    direct = 'maximize' if config.main_metric == 'accuracy' else 'minimize'
    study = optuna.create_study(direction=direct)
    study.optimize(objective, n_trials=config.num_iterations)

    model_params = study.best_trial.params
    print('Best params:')
    print(model_params, '\n')

    print('Start final model training.')
    opt_model = LGBMRegressor(n_jobs=-1, objective='mae',
                              random_state=42, n_estimators=int(config.n_estimators*3), **model_params)
    opt_model.fit(tr_data, tr_label, eval_set=[(ev_data, ev_label)],
                  callbacks=[early_stopping(100, verbose=0), log_evaluation(-1)])

    final_preds = opt_model.predict(ev_data)
    mask = ev_data[~((ev_data.spends_mon0 == 0) & (ev_data.spends_mon1 == 0) & (ev_data.spends_mon2 == 0)
                     & (ev_data.spends_mon3 == 0) & (ev_data.spends_mon4 == 0) &
                     (ev_data.spends_mon5 == 0) & (ev_data.spends_mon6 == 0) & (ev_data.spends_mon7 == 0))].index

    masked_fpreds = final_preds.copy()
    masked_fpreds[mask] = 0

    mae_metr = mean_absolute_error(ev_label, final_preds)
    mae_metr_masked = mean_absolute_error(ev_label, masked_fpreds)

    accuracy = perc(final_preds, ev_label.values.ravel(), config.metric_thres)
    accuracy_masked = perc(masked_fpreds, ev_label.values.ravel(), config.metric_thres)

    print(f'MAE: {round(mae_metr, 4)}')
    print(f'MAE masked: {round(mae_metr_masked, 4)}')
    print(f'Accuraty +-{round(config.metric_thres * 100)}%: {round(accuracy, 5)}')
    print(f'Accuraty +-{round(config.metric_thres * 100)}% masked: {round(accuracy_masked, 5)}\n')

    with open(f'models/best_lightgbm_model', 'wb') as curr_saved_model:
        pickle.dump(opt_model, curr_saved_model)

    print('Start onnx converting.')
    convert_to_onnx(opt_model, ev_data, ev_label, final_preds, config.metric_thres)
    enablePrint()
    print('Models saved in models folder.')