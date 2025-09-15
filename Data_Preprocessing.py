import pandas as pd
import numpy as np
import os
import re
import jieba
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def get_abs_path(relative_path):
    return os.path.abspath(relative_path)

# ==== 加载停用词表 ====
def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
            print(f"成功加载停用词表，包含 {len(stopwords)} 个停用词")
            return stopwords
    except Exception as e:
        print(f"加载停用词表失败: {e}")
        default_stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '在', '要',
                                 '为', '以', '或', '于', '有', '也', '但', '还', '又', '之', '这', '那', '着'])
        print(f"使用默认停用词表，包含 {len(default_stopwords)} 个停用词")
        return default_stopwords


def load_mapping(mapping_file):
    df = pd.read_csv(mapping_file, encoding='utf-8')
    return dict(zip(df['emoji'], df['chinese']))

def replace_emoji_with_special_token(text, mapping_dict):
    sorted_keys = sorted(mapping_dict.keys(), key=lambda x: (-len(x), x))
    pattern = re.compile('|'.join(re.escape(k) for k in sorted_keys))
    return pattern.sub(lambda x: f"[{mapping_dict[x.group()]}]", str(text))

def preprocess_text(text, stopwords=None):
    words = jieba.lcut(str(text))
    if stopwords:
        words = [word for word in words if word not in stopwords]
    words = [word for word in words if len(word) > 1]
    return ' '.join(words)

def preprocess_data(train_file, test_file, mapping_file, stopwords_path):

    stopwords = load_stopwords(stopwords_path)

    mapping_dict = load_mapping(mapping_file)
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    # 数据预处理
    for df in [train_df, test_df]:
        df['text'] = df['text'].fillna('')
        df['text-emoji'] = df['text-emoji'].fillna('')
        df['text_with_emoji'] = df['text-emoji'].apply(lambda x: replace_emoji_with_special_token(x, mapping_dict))

    # 文本预处理
    for mode in ['text', 'text_with_emoji']:
        train_df[f'{mode}_processed'] = train_df[mode].apply(lambda x: preprocess_text(x, stopwords))
        test_df[f'{mode}_processed'] = test_df[mode].apply(lambda x: preprocess_text(x, stopwords))

    return train_df, test_df