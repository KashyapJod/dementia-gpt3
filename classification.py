import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, make_scorer, recall_score, precision_score, f1_score
)
from sklearn.model_selection import (
    KFold, train_test_split, GridSearchCV, cross_validate
)
from sklearn.svm import SVC

import config
from config import logger

def embeddings_to_array(embeddings_file):
    df = pd.read_csv(embeddings_file)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    logger.debug(df.head())
    return df


def cross_validation(model, _X, _y, _cv):
    _scoring = {
        'accuracy': make_scorer(accuracy_score),  
        'precision': make_scorer(precision_score, average='weighted'), 
        'recall': make_scorer(recall_score, average='weighted'),        
        'f1_score': make_scorer(f1_score, average='macro')  
    }

    scores = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    result = {}

    for metric in metrics:
        train_scores = scores[f'test_{metric}']
        train_scores_mean = round(train_scores.mean(), 3)
        train_scores_std = round(train_scores.std(), 3)

        test_scores = scores[f'test_{metric}']
        test_scores_mean = round(test_scores.mean(), 3)

        result[f'train_{metric}'] = train_scores
        result[f'train_{metric}_mean'] = train_scores_mean
        result[f'train_{metric}_std'] = train_scores_std

        result[f'test_{metric}'] = test_scores
        result[f'test_{metric}_mean'] = test_scores_mean

    return result


def classify_embedding(train_data, test_data, _n_splits):
    logger.info("Initiating classification with GPT-3 text embeddings...")
    y_train = train_data['diagnosis'].values
    X_train = train_data['embedding'].to_list()
    X_test = test_data['embedding'].to_list()

    baseline_score = dummy_stratified_clf(X_train, y_train)
    logger.debug(f"Baseline performance of the dummy classifier: {baseline_score}")

    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    models_size_df = pd.DataFrame(columns=['Model', 'Size'])

    logger.info("Beginning to train models using GPT embeddings...")

    total_models_size = 0

    for model, name in zip(models, names):
        logger.info(f"Initiating {name}...")

        best_params = hyperparameter_optimization(X_train, y_train, cv, model, name)
        model.set_params(**best_params)
        scores = cross_validation(model, X_train, y_train, cv)
        results_df = results_to_df(name, scores, results_df)

        visualize_results(_n_splits, name, scores, (config.embedding_results_dir / "plots").resolve())

        model_size = len(pickle.dumps(model, -1))
        logger.debug(f"Model size of {name} before training: {model_size} bytes.")

        model.fit(X_train, y_train)

        model_size = len(pickle.dumps(model, -1))
        logger.debug(f"Model size of {name} after training: {model_size} bytes.")
        total_models_size += model_size

        models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': name,
                                                                   'Size': f"{model_size} B",
                                                                   }])], ignore_index=True)

        model_test_results = pd.read_csv(config.empty_test_results_file)

        model_predictions = model.predict(X_test)

        filename_to_prediction = {}

        for filename, prediction in zip(test_data['addressfname'], model_predictions):
            filename_to_prediction[filename] = 'ProbableAD' if prediction == 1 else 'Control'
        model_test_results['Prediction'] = model_test_results['ID'].map(filename_to_prediction)
        model_test_results_csv = (config.embedding_results_dir / f'task1_{name}.csv').resolve()
        model_test_results.to_csv(model_test_results_csv, index=False)
        logger.info(f"Writing {model_test_results_csv}...")
        evaluate_similarity(name, model_test_results)

    logger.info("Training using GPT embeddings done.")

    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': 'Dummy',
                                                       'Accuracy': baseline_score,
                                                       }])], ignore_index=True)

    embedding_results_file = (config.embedding_results_dir / 'embedding_results.csv').resolve()
    results_df.to_csv(embedding_results_file)
    logger.info(f"Writing {embedding_results_file}...")

    logger.debug(f"Total size of all models: {total_models_size}.")
    models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': 'Total',
                                                               'Size': f'{total_models_size} B',
                                                               }])], ignore_index=True)

    models_size_df.to_csv(config.models_size_file)
    logger.info(f"Writing {config.models_size_file}...")

    logger.info("Classification with GPT-3 text embeddings done.")


def evaluate_similarity(name, model_test_results):
    test_results_task1 = pd.read_csv(config.test_results_task1)
    real_diagnoses = test_results_task1['Dx']
    predicted_diagnoses = model_test_results['Prediction']
    matching_values = (real_diagnoses == predicted_diagnoses).sum()
    total_values = len(real_diagnoses)
    similarity_percentage = (matching_values / total_values) * 100
    logger.info(f"The similarity between the real and predicted diagnoses using model {name} "
                f"is {similarity_percentage:.2f}%.")


def hyperparameter_optimization(X_train, y_train, cv, model, name):
    lr_param_grid, rf_param_grid, svc_param_grid = param_grids()
    grid_search = None
    if name == 'SVC':
        grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'LR':
        grid_search = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'RF':
        grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=cv, n_jobs=-1, error_score=0.0)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


def param_grids():
    svc_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    lr_param_grid = [
        {'penalty': ['l1', 'l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['liblinear'],
         'max_iter': [100, 200, 500, 1000]},
        {'penalty': ['l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['lbfgs'],
         'max_iter': [200, 500, 1000]},
    ]
    rf_param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    return lr_param_grid, rf_param_grid, svc_param_grid


def visualize_results(_n_splits, name, results, save_dir):
    plot_accuracy_path = (save_dir / f'plot_accuracy_{name}.png').resolve()
    plot_precision_path = (save_dir / f'plot_precision_{name}.png').resolve()
    plot_recall_path = (save_dir / f'plot_recall_{name}.png').resolve()
    plot_f1_path = (save_dir / f'plot_f1_{name}.png').resolve()
    plot_result(name,
                "Accuracy",
                f"Accuracy scores in {_n_splits} Folds",
                results["train_accuracy"],
                results["test_accuracy"],
                plot_accuracy_path)
    plot_result(name,
                "Precision",
                f"Precision scores in {_n_splits} Folds",
                results["train_precision"],
                results["test_precision"],
                plot_precision_path)
    plot_result(name,
                "Recall",
                f"Recall scores in {_n_splits} Folds",
                results["train_recall"],
                results["test_recall"],
                plot_recall_path)
    plot_result(name,
                "F1",
                f"F1 Scores in {_n_splits} Folds",
                results["train_f1_score"],
                results["test_f1_score"],
                plot_f1_path)


def results_to_df(name, scores, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Train',
                                                       'Model': name,
                                                       'Accuracy': f"{scores['train_accuracy_mean']} "
                                                                   f"({scores['train_accuracy_std']})",
                                                       'Precision': f"{scores['train_precision_mean']} "
                                                                    f"({scores['train_precision_std']})",
                                                       'Recall': f"{scores['train_recall_mean']} "
                                                                 f"({scores['train_recall_std']})",
                                                       'F1': f"{scores['train_f1_score_mean']} "
                                                             f"({scores['train_f1_score_std']})",
                                                       }])], ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': name,
                                                       'Accuracy': scores['test_accuracy_mean'],
                                                       'Precision': scores['test_precision_mean'],
                                                       'Recall': scores['test_recall_mean'],
                                                       'F1': scores['test_f1_score_mean']
                                                       }])], ignore_index=True)
    return results_df

def plot_result(x_label, y_label, plot_title, train_data, val_data, savefig_path=None):
    fig = plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold",
              "9th Fold", "10th Fold"]
    X_axis = np.arange(len(labels))
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    if savefig_path is not None:
        fig.savefig(savefig_path, dpi=fig.dpi)


def dummy_stratified_clf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    stratified_clf = DummyClassifier(strategy='stratified').fit(X_train, y_train)

    score = round(stratified_clf.score(X_test, y_test), 3)

    return score
