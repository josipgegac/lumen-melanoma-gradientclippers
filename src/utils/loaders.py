import warnings

from random import sample
from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd
import os
import cv2


def load_image_dataset(dataset_path: str, metadata_path: str, image_format: str = '.jpg', dtypes = None):
    df = pd.read_csv(metadata_path, dtype=dtypes)
    df = df.rename(columns={'image_name': 'image',
                                              'patient_id': 'patient',
                                              'anatom_site_general_challenge': 'anatom_site_general',
                                              })
    if not dataset_path.endswith(os.sep):
        dataset_path += os.sep
    if not image_format.startswith('.'):
        image_format = '.' + image_format
    df['image_path'] = dataset_path + df['image'] + image_format

    not_found_files = []
    for path in df['image_path']:
        if not os.path.exists(path):
            not_found_files.append(path)

    if len(not_found_files) == df.shape[0]:
        warnings.warn(f'No corresponding files found at {dataset_path}!')
    elif len(not_found_files) > 0:
        warnings.warn(f'{len(not_found_files)} files not found at {dataset_path}!')
        if len(not_found_files) < 100:
            for path in not_found_files:
                warnings.warn(f'File {path} not found')

    df.index.name = 'id'
    return df

def load_images(file_locations: Sequence[str], sample_size: Union[None, int] = None) -> Tuple[str, np.ndarray]:
    if sample_size is not None:
        file_locations = sample(file_locations, sample_size)

    for file in file_locations:
        img = cv2.imread(file, cv2.IMREAD_COLOR_RGB)
        if img is not None:
            yield file[file.rfind(os.sep) + 1 if os.sep in file else 0 : file.rfind('.') if '.' in file else -1], img

