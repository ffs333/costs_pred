import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error

from onnxmltools import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt

from core.metric import perc


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def train_eval_split(df, months, cols, save=False):
    """
    Splits data to train and eval by last month
    :param df: input DataFrame
    :param months: list of months in data
    :param cols: columns to train
    :param save: save prepared date
    :return: train and eval data and labels
    """
    train = df[df['end'] != months[-1]][cols]
    evals = df[df['end'] == months[-1]][cols]
    print(f'Train data shape: {train.shape}\n'
          f'Eval data shape: {evals.shape}')

    tr_data = train.drop(['label'], axis=1)
    tr_label = train['label']
    ev_data = evals.drop(['label'], axis=1)
    ev_label = evals['label']

    tr_data = tr_data.reset_index(drop=True)
    tr_label = tr_label.reset_index(drop=True)
    ev_data = ev_data.reset_index(drop=True)
    ev_label = ev_label.reset_index(drop=True)
    if save:
        tr_data.to_csv('/data/processed_data/train_data.csv', index=False)
        tr_label.to_csv('/data/processed_data/train_label.csv', index=False)
        ev_data.to_csv('/data/processed_data/eval_data.csv', index=False)
        ev_label.to_csv('/data/processed_data/eval_label.csv', index=False)
        print('Data saved here /data/processed_data/')
    return tr_data, tr_label, ev_data, ev_label


def convert_to_onnx(model, x_test, y_test, light_preds, thres):
    """
    Convert model to onnx format and test outputs similarity
    :param model: lightgbm model
    :param x_test: eval data
    :param y_test: eval labels
    :param light_preds: predictions of lightgbm model
    :param thres: threshold for metric
    """
    initial_types = [("float_input", FloatTensorType([None, x_test.shape[1]]))]
    onnx_model = convert_lightgbm(model=model, name="onnx_lightgbm", initial_types=initial_types, target_opset=8)

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([label_name], {input_name: x_test.values.astype(np.float32)})[0][:, 0]

    percc = perc(pred_onnx, y_test.values.ravel(), thres)
    curr_mae_metr = mean_absolute_error(y_test, pred_onnx)
    print(f'MAE: {round(curr_mae_metr, 4)}')
    print(f'Accuraty +-{round(thres * 100)}%: {round(percc, 5)}')

    try:
        np.testing.assert_allclose(pred_onnx, light_preds, rtol=1e-03, atol=1e-03)
        print('Similarity test passed.\n')
        onnx.save(onnx_model, 'models/final_model.onnx')
    except AssertionError:
        print(f'Similarity test failed. Something went wrong.')
        onnx.save(onnx_model, 'models/final_model_similarity_failed.onnx')