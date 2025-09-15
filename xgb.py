import os
import joblib
import numpy as np
import pandas as pd
import optuna
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from Data_Preprocessing import preprocess_data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

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

def hyperparameter_tuning(X_train, y_train, pipeline, SEED):
    def objective(trial):
        params = {
            'xgbclassifier__learning_rate': trial.suggest_float("xgbclassifier__learning_rate", 0.0001, 0.1),
            'xgbclassifier__gamma': trial.suggest_float("xgbclassifier__gamma", 1, 10),
            'xgbclassifier__max_depth': trial.suggest_int("xgbclassifier__max_depth", 3, 7),
            'xgbclassifier__n_estimators': trial.suggest_categorical("xgbclassifier__n_estimators",
                                                                     [10, 100, 1000, 10000]),
            'xgbclassifier__colsample_bytree': trial.suggest_float("xgbclassifier__colsample_bytree", 0.1, 1),
            'xgbclassifier__scale_pos_weight': trial.suggest_float("xgbclassifier__scale_pos_weight", 0.1, 10)
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold = [X_train[i] for i in train_idx]
            X_val_fold = [X_train[i] for i in val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            fold_pipeline = pipeline.set_params(**params)
            fold_pipeline.fit(X_train_fold, y_train_fold)

            y_val_prob = fold_pipeline.predict_proba(X_val_fold)[:, 1]
            y_val_pred = (y_val_prob > 0.5).astype(int)
            f1_scores.append(f1_score(y_val_fold, y_val_pred))

        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    return best_params

def process_mode(mode, train_df, test_df, MODEL_SAVE_DIR, FIGURE_DIR, SEED):
    print(f"\n--- Training with {mode.upper()} ---")

    X_train = train_df[f'{mode}_processed'].astype(str).values
    y_train = train_df['label'].astype(int).values
    X_test = test_df[f'{mode}_processed'].astype(str).values
    y_test = test_df['label'].astype(int).values

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        XGBClassifier(random_state=SEED)
    )

    print("Optimizing hyperparameters...")
    best_params = hyperparameter_tuning(X_train, y_train, pipeline, SEED)
    print(f"[{mode}] 最佳参数: {best_params}")

    final_pipeline = pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train)

    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_test_prob, threshold=0.5)
    y_pred = results.pop('y_pred')

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

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    pr_curve_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall
    })
    pr_curve_df.to_excel(os.path.join(FIGURE_DIR, f'pr_curve_{mode}_xgb.xlsx'), index=False)
    print(f"Precision-Recall curve data saved to pr_curve_{mode}_xgb.xlsx")

    return {
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

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for mode in ['text', 'text_with_emoji']:

        mode_processing = process_mode(mode, train_df, test_df, MODEL_SAVE_DIR, FIGURE_DIR, SEED)

        joblib.dump(mode_processing['model_dict'], os.path.join(MODEL_SAVE_DIR, f"xgb_{mode}.pkl"))

        pr_data.append(mode_processing['pr'])

        all_results.append(mode_processing['result_entry'])

    eval_df = pd.DataFrame(all_results)
    eval_df.to_excel(os.path.join(FIGURE_DIR, 'evaluation_metrics_XGB.xlsx'), index=False)
    print("All evaluation metrics saved to evaluation_metrics_XGB.xlsx")

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve - XGB')

    for i, mode in enumerate(['text', 'text_with_emoji']):
        recall, precision, pr_auc = pr_data[i]
        ax_pr.plot(recall, precision, label=f'{mode} (PR-AUC={pr_auc:.4f})')

    ax_pr.legend()
    fig_pr.savefig(os.path.join(FIGURE_DIR, 'pr_XGB.png'))

    for mode in ['text', 'text_with_emoji']:
        test_df[f'predicted_label_{mode}_xgb'] = joblib.load(os.path.join(MODEL_SAVE_DIR, f"xgb_{mode}.pkl"))['pipeline'].predict(test_df[f'{mode}_processed'].astype(str).values)

    test_output_path = os.path.join(FIGURE_DIR, 'your path')
    test_df.to_excel(test_output_path, index=False)
    print(f"have reserve {test_output_path}")