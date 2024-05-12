import os
import config
from config import logger
import pandas as pd
from sklearn.utils import resample
from matplotlib import pyplot as plt


def fetch_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    logger.info(f"Successfully fetched {len(audio_files)} (.wav) audio files!")
    return audio_files


def get_user_input(prompt, choices):
    while True:
        user_input = input(prompt)
        if user_input.lower() in choices:
            return user_input.lower()
        else:
            logger.info('Invalid input. Please try again.')


def df_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    logger.info(f"Writing {file_path}...")


def add_train_scores(df):
    text_data = df
    logger.debug(text_data)
    scores_df = pd.read_csv(config.diagnosis_train_scores)
    scores_df = scores_df.rename(columns={'adressfname': 'addressfname', 'dx': 'diagnosis'})
    scores_df = binarize_labels(scores_df)
    logger.debug(scores_df)
    output = pd.merge(text_data,
                      scores_df[['addressfname', 'mmse', 'diagnosis']],  # We don't want the key column here
                      on='addressfname',
                      how='inner')

    logger.debug(output)
    return output


def binarize_labels(df):
    df['diagnosis'] = [1 if label == 'ad' else 0 for label in df['diagnosis']]
    df_majority = df[df['diagnosis'] == 1]  
    df_minority = df[df['diagnosis'] == 0]  
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  
                                       n_samples=len(df_minority), 
                                       random_state=42)  

    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    plt.show()
    return df_downsampled
