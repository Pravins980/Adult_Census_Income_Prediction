# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
import numpy as np



def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    """
    dataset=pd.read_csv(input_filepath)
    dataset = dataset.replace('?', np.nan)
    # Checking null values
    columns_with_nan = ['workclass', 'occupation', 'native.country']
    for col in columns_with_nan:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    dataset.to_csv(output_filepath)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
