import logging
import os
import sys
import typing
from pathlib import Path
from formatter import Formatter
import tiktoken

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(Formatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

dirname = Path(__file__).parent.resolve()

whisper_model_name = None
whisper_model = None

max_tokens = 500
embedding_engine = 'text-embedding-ada-002'

n_splits = 10

data_dir = (dirname / "ADReSSo").resolve()

diagnosis_train_data = (
        data_dir / "diagnosis-train" / "diagnosis" / "train" / "audio").resolve()  # Dementia and control group
diagnosis_test_data = (
        data_dir / "diagnosis-test" / "diagnosis" / "test-dist" / "audio").resolve()  # Test data
diagnosis_train_scores = (
        data_dir / "diagnosis-train" / "diagnosis" / "train" / "adresso-train-mmse-scores.csv").resolve()
empty_test_results_file = (data_dir / "diagnosis-test" / "diagnosis" / "test-dist" / "test_results_task1.csv").resolve()
test_results_task1 = (data_dir / "task1.csv").resolve()


decline_data = (
        data_dir / "progression-train" / "progression" / "train" / "audio" / "decline").resolve()  
no_decline_data = (
        data_dir / "progression-train" / "progression" / "train" / "audio" / "no_decline").resolve()  


transcription_dir = (dirname / "processed" / "transcription").resolve()
diagnosis_train_transcription_dir = (transcription_dir / "train").resolve()
diagnosis_test_transcription_dir = (transcription_dir / "test").resolve()
train_scraped_path = (dirname / "processed" / "train_scraped.csv").resolve()
test_scraped_path = (dirname / "processed" / "test_scraped.csv").resolve()
train_embeddings_path = (dirname / "processed" / "train_embeddings.csv").resolve()
test_embeddings_path = (dirname / "processed" / "test_embeddings.csv").resolve()

embedding_results_dir = (dirname / "results" / "embedding").resolve()
models_size_file = (embedding_results_dir / 'embedding_models_size.csv').resolve()


def set_up():
    logger.info("Loading cl100k_base tokenizer...")
    logger.info(f"Max tokens per embedding: {max_tokens}.")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logger.info(f"Loading GPT embedding engine {embedding_engine}...")

    Path(dirname / "processed").resolve().mkdir(exist_ok=True)
    Path(dirname / "results").resolve().mkdir(exist_ok=True)
    embedding_results_dir.mkdir(exist_ok=True)
    Path(embedding_results_dir / 'plots').resolve().mkdir(exist_ok=True)

    return tokenizer


def secret_key() -> typing.Optional[str]:
    value = os.environ.get('OPENAI_API_KEY', None)

    if not value:
        logger.warning("Optional environment variable 'OPENAI_API_KEY' is missing.")

    return value
