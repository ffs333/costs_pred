from pyhocon import ConfigFactory
import pandas as pd

from data_pipeline.data_preprocess import prepare_data
from utils.utils import blockPrint
from core.train import training


def prepare_config(path: str):
    """
    Read config file
    :param path: path to config file
    :return tuple with configurations for necessary modules
    """
    conf = ConfigFactory.parse_file(path)
    return conf['data'], conf['train']


def pipeline(path, verbosity=True):
    """
    Aggregate configs and start data and training pipelines
    :param path: path to config file
    :param verbosity: show or hide prints
    """
    if not verbosity:
        blockPrint()
    data_conf, train_conf = prepare_config(path)

    if data_conf.load_prepared:
        print(f'Loading generated date without preprocessing.')
        train_data = pd.read_csv('./data/processed_data/train_data.csv')
        train_label = pd.read_csv('./data/processed_data/train_label.csv')
        eval_data = pd.read_csv('./data/processed_data/eval_data.csv')
        eval_label = pd.read_csv('./data/processed_data/eval_label.csv')
        print(f'Loaded.\n'
              f'Train data shape: {train_data.shape}\n'
              f'Eval data shape: {eval_data.shape}')
    else:
        train_data, train_label, eval_data, eval_label = prepare_data(data_conf)

    training(train_conf, train_data, train_label, eval_data, eval_label)

