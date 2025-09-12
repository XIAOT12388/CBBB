import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import optuna
import matplotlib.pyplot as plt
import os
import re
import joblib

# 设置随机种子以确保可重复性
SEED = 42

# 定义文件路径（建议使用相对路径或环境变量）
TRAIN_FILE_PATH = r'D:\桌面\Featurize上训练\train_set0403.xlsx'
TEST_FILE_PATH = r'D:\桌面\Featurize上训练\test_set0403.xlsx'
MAPPING_FILE_PATH = r'D:\桌面\建立映射\mapping_cleaned.csv'
FIGURE_DIR = os.path.join(os.getcwd(), 'FigureLR')
MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'Saved_ModelsLR')

# 确保Figure和模型保存文件夹存在
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 定义全局数据预处理函数
def load_mapping(mapping_file):
    """加载emoji映射文件，返回映射字典"""
    mapping_dict = {}
    df = pd.read_csv(mapping_file, encoding='utf-8')
    for _, row in df.iterrows():
        emoji = row['emoji']
        description = row['chinese']
        mapping_dict[emoji] = description
    return mapping_dict

def replace_emoji_with_special_token(text, mapping_dict):
    """将文本中的emoji替换为 [描述] 格式"""
    sorted_keys = sorted(mapping_dict.keys(), key=lambda x: (-len(x), x))
    pattern = re.compile('|'.join(re.escape(k) for k in sorted_keys))
    return pattern.sub(lambda x: f"[{mapping_dict[x.group()]}]", str(text))

def load_and_preprocess_data(train_file, test_file, mapping_file):
    """加载并预处理数据"""
    # 加载训练集和测试集
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    # 预处理：填充缺失值并转换为字符串
    for df in [train_df, test_df]:
        df['text-emoji'] = df['text-emoji'].fillna('')
        df['text'] = df['text'].fillna('')

    # 加载映射文件
    mapping_dict = load_mapping(mapping_file)

    # 替换emoji为特殊符号
    for df in [train_df, test_df]:
        df['text_with_emoji'] = df['text-emoji'].apply(lambda x: replace_emoji_with_special_token(x, mapping_dict))

    # 提取文本和标签
    train_texts = train_df['text'].astype(str).values
    train_texts_with_emoji = train_df['text_with_emoji'].astype(str).values
    train_labels = train_df['label'].values.astype(int)

    test_texts = test_df['text'].astype(str).values
    test_texts_with_emoji = test_df['text_with_emoji'].astype(str).values
    test_labels = test_df['label'].values.astype(int)

    return (train_texts, train_texts_with_emoji, train_labels,
            test_texts, test_texts_with_emoji, test_labels)

# 特征提取函数
def extract_features(texts, max_features=5000):
    """提取TF-IDF特征"""
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    return tfidf_vectorizer.fit_transform(texts), tfidf_vectorizer

# 定义训练函数
def train_lr(X, y, params):
    """训练逻辑回归模型"""
    # 移除 solver 参数（如果它在 params 中）
    solver = params.pop('solver', 'lbfgs')  # 默认值为 'lbfgs'
    # 确保 class_weight 被设置
    if 'class_weight' not in params:
        params['class_weight'] = 'balanced'
    model = LogisticRegression(**params, random_state=SEED, max_iter=5000, solver=solver)
    model.fit(X, y)
    return model

# 定义评估函数
def evaluate_model(model, X, y, threshold=0.5):
    """评估模型性能"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y, y_proba)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_proba),
        'fpr': fpr,
        'tpr': tpr,
        'proba': y_proba
    }

    return metrics

# 寻找最佳分类阈值
def find_best_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 81):
        y_pred = (y_proba >= threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    return best_threshold

# 定义Optuna的目标函数
def objective(trial, X, y):
    # 定义逻辑回归的超参数搜索空间
    params = {
        'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])  # 增加 solver 超参数
    }

    # 使用StratifiedKFold进行10折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 训练模型
        model = LogisticRegression(**params, random_state=SEED, max_iter=1000)
        model.fit(X_train, y_train)

        # 使用固定阈值 0.5 进行评估
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_proba),
        }

        scores.append(metrics['f1'])

    return np.mean(scores)

# 主函数
if __name__ == "__main__":
    # 加载和预处理数据
    (train_texts, train_texts_with_emoji, train_labels,
     test_texts, test_texts_with_emoji, test_labels) = load_and_preprocess_data(
        TRAIN_FILE_PATH, TEST_FILE_PATH, MAPPING_FILE_PATH
    )

    # 提取特征
    X_train_no_emoji, tfidf_vectorizer_no_emoji = extract_features(train_texts)
    X_train_with_emoji, tfidf_vectorizer_with_emoji = extract_features(train_texts_with_emoji)
    X_test_no_emoji = tfidf_vectorizer_no_emoji.transform(test_texts)
    X_test_with_emoji = tfidf_vectorizer_with_emoji.transform(test_texts_with_emoji)

    # 超参数调优
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: objective(trial, X_train_with_emoji, train_labels), n_trials=50)

    # 输出最佳超参数
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = trial.params

    # 明确确保在训练最终模型时使用 class_weight='balanced'
    if 'class_weight' not in best_params:
        best_params['class_weight'] = 'balanced'

    # 训练最终模型（带emoji）
    best_model_with_emoji = train_lr(X_train_with_emoji, train_labels, best_params.copy())
    best_threshold_with_emoji = find_best_threshold(best_model_with_emoji, X_train_with_emoji, train_labels)
    test_metrics_with_emoji = evaluate_model(best_model_with_emoji, X_test_with_emoji, test_labels, best_threshold_with_emoji)

    # 训练最终模型（不带emoji）
    best_model_no_emoji = train_lr(X_train_no_emoji, train_labels, best_params.copy())
    best_threshold_no_emoji = find_best_threshold(best_model_no_emoji, X_train_no_emoji, train_labels)
    test_metrics_no_emoji = evaluate_model(best_model_no_emoji, X_test_no_emoji, test_labels, best_threshold_no_emoji)

    # 输出测试集评估结果
    print("\nTest Set Evaluation (With Emoji - Best Params):")
    print(
        f"Accuracy: {test_metrics_with_emoji['accuracy']:.4f}, Precision: {test_metrics_with_emoji['precision']:.4f}, Recall: {test_metrics_with_emoji['recall']:.4f}, F1: {test_metrics_with_emoji['f1']:.4f}, AUC: {test_metrics_with_emoji['auc']:.4f}")

    print("\nTest Set Evaluation (No Emoji - Best Params):")
    print(
        f"Accuracy: {test_metrics_no_emoji['accuracy']:.4f}, Precision: {test_metrics_no_emoji['precision']:.4f}, Recall: {test_metrics_no_emoji['recall']:.4f}, F1: {test_metrics_no_emoji['f1']:.4f}, AUC: {test_metrics_no_emoji['auc']:.4f}")

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(test_metrics_with_emoji['fpr'], test_metrics_with_emoji['tpr'],
             label=f'With Emoji (AUC = {test_metrics_with_emoji["auc"]:.4f})')
    plt.plot(test_metrics_no_emoji['fpr'], test_metrics_no_emoji['tpr'],
             label=f'No Emoji (AUC = {test_metrics_no_emoji["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(FIGURE_DIR, 'roc_curve.png'))
    plt.show()

    # 保存最优模型
    joblib.dump(best_model_with_emoji, os.path.join(MODEL_SAVE_DIR, 'best_model_with_emoji_lr.pkl'))
    joblib.dump(best_model_no_emoji, os.path.join(MODEL_SAVE_DIR, 'best_model_no_emoji_lr.pkl'))