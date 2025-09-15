import pandas as pd
import numpy as np
import os
import joblib
import optuna
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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

if __name__ == "__main__":
    train_df, test_df = preprocess_data(TRAIN_FILE_PATH, TEST_FILE_PATH, MAPPING_FILE_PATH, STOPWORDS_PATH)

    all_results = []
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for mode in ['text', 'text_with_emoji']:
        print(f"\n--- Training with {mode.upper()} ---")

        X_train = train_df[f'{mode}_processed'].astype(str).values
        y_train = train_df['label'].astype(int).values
        X_test = test_df[f'{mode}_processed'].astype(str).values
        y_test = test_df['label'].astype(int).values

        pipeline = make_pipeline(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            RandomForestClassifier(random_state=SEED)
        )

        print("Optimizing hyperparameters...")

        def objective(trial):
            params = {
                'randomforestclassifier__n_estimators': trial.suggest_int("randomforestclassifier__n_estimators", 100, 200),
                'randomforestclassifier__max_depth': trial.suggest_int("randomforestclassifier__max_depth", 5, 20),
                'randomforestclassifier__min_samples_split': trial.suggest_int("randomforestclassifier__min_samples_split", 2, 10),
                'randomforestclassifier__min_samples_leaf': trial.suggest_int("randomforestclassifier__min_samples_leaf", 1, 10),
                'randomforestclassifier__max_features': trial.suggest_categorical("randomforestclassifier__max_features", ['sqrt', 'log2', None]),
                'randomforestclassifier__class_weight': trial.suggest_categorical("randomforestclassifier__class_weight", ['balanced'])
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
        print(f"[{mode}] 最佳参数: {best_params}")

        final_pipeline = pipeline.set_params(**best_params)
        final_pipeline.fit(X_train, y_train)

        y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
        results = evaluate_model(y_test, y_test_prob, threshold=0.5)
        y_pred = results.pop('y_pred')


        if mode == 'text':
            test_df['predicted_label_text'] = y_pred
        elif mode == 'text_with_emoji':
            test_df['predicted_label_text_with_emoji'] = y_pred


        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        pr_auc = auc(recall, precision)

        print(f"[{mode}] Test Evaluation:")
        print(f"Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, "
              f"Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}, AUC: {results['auc']:.4f}, "
              f"PR-AUC: {pr_auc:.4f}")

        all_results.append({
            'Mode': mode,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1': results['f1'],
            'AUC': results['auc'],
            'PR-AUC': pr_auc
        })

        pr_curve_df = pd.DataFrame({'Precision': precision, 'Recall': recall})
        pr_curve_df.to_excel(os.path.join(FIGURE_DIR, f'pr_curve_{mode}.xlsx'), index=False)

        model_dict = {
            'pipeline': final_pipeline,
            'mode': mode,
            'params': best_params,
            'evaluation': {**results, 'pr_auc': pr_auc}
        }
        joblib.dump(model_dict, os.path.join(MODEL_SAVE_DIR, f"randomforest_{mode}.pkl"))

        ax_pr.plot(recall, precision, label=f'{mode} (PR-AUC={pr_auc:.4f})')

    eval_df = pd.DataFrame(all_results)
    eval_df.to_excel(os.path.join(FIGURE_DIR, 'evaluation_metrics_RF.xlsx'), index=False)

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    fig_pr.savefig(os.path.join(FIGURE_DIR, 'pr_RF.png'))

    test_output_path = os.path.join(FIGURE_DIR, 'your path')
    test_df.to_excel(test_output_path, index=False)
    print(f"have reserve{test_output_path}")
