import config
from config import logger
from utils import fetch_audio_files, df_to_csv, add_train_scores
from pathlib import Path
import whisper
import os
import codecs
import pandas as pd


def transcribe():
    whisper_model = whisper.load_model(config.whisper_model_name)

    logger.info("Initiating transcription...")

    diagnosis_train_audio_files = fetch_audio_files(config.diagnosis_train_data)
    logger.debug(diagnosis_train_audio_files)
    diagnosis_test_audio_files = fetch_audio_files(config.diagnosis_test_data)
    logger.debug(diagnosis_test_audio_files)

    write_transcription(diagnosis_train_audio_files, config.diagnosis_train_transcription_dir, whisper_model)
    write_transcription(diagnosis_test_audio_files, config.diagnosis_test_transcription_dir, whisper_model)

    train_df = transcription_to_df(config.diagnosis_train_transcription_dir)
    train_df = add_train_scores(train_df)

    test_df = transcription_to_df(config.diagnosis_test_transcription_dir)

    df_to_csv(train_df, config.train_scraped_path)
    df_to_csv(test_df, config.test_scraped_path)

    logger.info("Transcription done.")


def write_transcription(audio_files, transcription_dir, whisper_model):
    for audio_file in audio_files:
        filename = Path(audio_file).stem
        transcription_file = (transcription_dir / filename).resolve()

        if not transcription_file.exists():
            result = whisper_model.transcribe(audio_file, fp16=False)
            transcription_str = str(result["text"])

            transcription_file.parent.mkdir(parents=True, exist_ok=True)

            transcription_file.write_text(transcription_str)
            logger.info(f"Transcribed {transcription_file}...")


def transcription_to_df(data_dir):
    texts = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            with codecs.open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                texts.append((file, text))

    df = pd.DataFrame(texts, columns=['addressfname', 'transcript'])
    df['transcript'] = df['transcript'].str.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ')
    df = df.sort_values(by='addressfname')
    df = df.reset_index(drop=True)
    logger.debug(df)

    return df
