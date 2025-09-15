import os
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import optuna
import random
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from Data_Preprocessing import preprocess_data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['SKLEARN_SEED'] = str(SEED)

TRAIN_FILE_PATH = 'your path'
TEST_FILE_PATH = 'your path'
MAPPING_FILE_PATH = 'your path'
STOPWORDS_PATH = 'your path'
MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'your path')
FIGURE_DIR = os.path.join(os.getcwd(), 'your path')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'y_pred': y_pred
    }

def hyperparameter_tuning_svm(X_train, y_train, pipeline, SEED):
    def objective(trial):
        params = {
            'svc__C': trial.suggest_categorical("svc__C", [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
            'svc__kernel': trial.suggest_categorical("svc__kernel", ['linear']),
            'svc__class_weight': trial.suggest_categorical("svc__class_weight", ['balanced'])
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

        def parallel_fit(train_idx, val_idx):
            try:
                X_train_fold = [X_train[i] for i in train_idx]
                X_val_fold = [X_train[i] for i in val_idx]
                y_train_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]

                fold_pipeline = clone(pipeline)
                fold_pipeline.set_params(**params)
                fold_pipeline.fit(X_train_fold, y_train_fold)

                y_val_prob = fold_pipeline.predict_proba(X_val_fold)[:, 1]
                y_val_pred = (y_val_prob > 0.5).astype(int)
                return f1_score(y_val_fold, y_val_pred)
            except Exception as e:
                print(f"Error in parallel_fit: {e}")
                return np.nan

        results = Parallel(n_jobs=-1, require='sharedmem')(
            delayed(parallel_fit)(train_idx, val_idx)
            for train_idx, val_idx in skf.split(X_train, y_train)
        )

        return np.nanmean(results)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    return best_params

def process_mode_svm(mode, train_df, test_df, MODEL_SAVE_DIR, FIGURE_DIR, SEED):
    print(f"\n--- Training with {mode.upper()} ---")

    X_train = train_df[f'{mode}_processed'].astype(str).values
    y_train = train_df['label'].astype(int).values
    X_test = test_df[f'{mode}_processed'].astype(str).values
    y_test = test_df['label'].astype(int).values

    print("X_train type:", type(X_train), "sample:", X_train[:2])
    print("y_train type:", type(y_train))

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        SVC(probability=True, random_state=SEED)
    )

    print("Pipeline steps:", pipeline.named_steps)

    best_params = hyperparameter_tuning_svm(X_train, y_train, pipeline, SEED)
    print(f"[{mode}] 最佳参数: {best_params}")

    final_pipeline = clone(pipeline)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train)

    print("Is SVC fitted?", hasattr(final_pipeline.named_steps['svc'], 'classes_'))

    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_test_prob, threshold=0.5)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recall, precision)

    print(f"[{mode}] Test Evaluation:")
    print(f"Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, "
          f"Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}, AUC: {results['auc']:.4f}, "
          f"PR-AUC: {pr_auc:.4f}")

    result_entry = {
        'Mode': mode,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1': results['f1'],
        'AUC': results['auc'],
        'PR-AUC': pr_auc
    }

    pr_curve_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall
    })
    pr_curve_df.to_excel(os.path.join(FIGURE_DIR, f'pr_curve_{mode}_svm.xlsx'), index=False)
    print(f"Precision-Recall curve data saved to pr_curve_{mode}_svm.xlsx")

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recall, precision)

    return {
        'roc': (fpr, tpr, results["auc"]),
        'pr': (recall, precision, pr_auc),
        'model_dict': {
            'pipeline': final_pipeline,
            'mode': mode,
            'params': best_params,
            'evaluation': {**results, 'pr_auc': pr_auc}
        },
        'result_entry': result_entry,
        'y_prob': y_test_prob
    }


if __name__ == "__main__":
    train_df, test_df = preprocess_data(TRAIN_FILE_PATH, TEST_FILE_PATH, MAPPING_FILE_PATH, STOPWORDS_PATH)

    all_results = []
    roc_data = []
    pr_data = []
    delong_data = {
        'y_true': None,
        'y_prob_text': None,
        'y_prob_text_with_emoji': None
    }

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for mode in ['text', 'text_with_emoji']:

        mode_processing = process_mode_svm(mode, train_df, test_df, MODEL_SAVE_DIR, FIGURE_DIR, SEED)

        joblib.dump(mode_processing['model_dict'], os.path.join(MODEL_SAVE_DIR, f"svm_{mode}.pkl"))

        pr_data.append(mode_processing['pr'])

        all_results.append(mode_processing['result_entry'])

    eval_df = pd.DataFrame(all_results)
    eval_df.to_excel(os.path.join(FIGURE_DIR, 'evaluation_metrics_SVM.xlsx'), index=False)
    print("All evaluation metrics saved to evaluation_metrics_SVM.xlsx")

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve - SVM')

    for i, mode in enumerate(['text', 'text_with_emoji']):
        recall, precision, pr_auc = pr_data[i]
        ax_pr.plot(recall, precision, label=f'{mode} (PR-AUC={pr_auc:.4f})')

    ax_pr.legend()
    fig_pr.savefig(os.path.join(FIGURE_DIR, 'pr_SVM.png'))

    for mode in ['text', 'text_with_emoji']:
        test_df[f'predicted_label_{mode}_svm'] = joblib.load(os.path.join(MODEL_SAVE_DIR, f"svm_{mode}.pkl"))[
            'pipeline'].predict(test_df[f'{mode}_processed'].astype(str).values)

    test_output_path = os.path.join(FIGURE_DIR, 'your path')
    test_df.to_excel(test_output_path, index=False)
    print(f"have reserve {test_output_path}")