import pandas as pd
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import set_seed

set_seed(SEED)

figure_dir = os.path.join(os.getcwd(), 'Figure8')
model_save_dir = os.path.join(os.getcwd(), 'Saved_Models8')
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

def load_mapping(mapping_file):

    mapping_dict = {}
    df = pd.read_csv(mapping_file, encoding='utf-8')
    for _, row in df.iterrows():
        emoji = row['emoji']
        description = row['chinese']
        mapping_dict[emoji] = description
    return mapping_dict

def replace_emoji_with_special_token(text, mapping_dict):

    sorted_keys = sorted(mapping_dict.keys(), key=lambda x: (-len(x), x))
    pattern = re.compile('|'.join(re.escape(k) for k in sorted_keys))
    return pattern.sub(lambda x: f"[{mapping_dict[x.group()]}]", str(text))

def encode_texts(texts, tokenizer, batch_size=8, max_length=180):

    encodings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[i:i + batch_size]]
        encoding = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=max_length,
                             return_tensors='pt')
        encodings.append(encoding)
    input_ids = torch.cat([e['input_ids'] for e in encodings], dim=0)
    attention_masks = torch.cat([e['attention_mask'] for e in encodings], dim=0)
    return input_ids, attention_masks


train_file_path = 'your path'

df = pd.read_excel(train_file_path)

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df['text-emoji'] = df['text-emoji'].fillna('')

mapping_file = 'your path'
mapping_dict = load_mapping(mapping_file)

df['text_with_emoji'] = df['text-emoji'].apply(lambda x: replace_emoji_with_special_token(x, mapping_dict))


texts = df['text'].values
texts_with_emoji = df['text_with_emoji'].values
labels = df['label'].values.astype(int)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

all_special_tokens = []
for text in texts_with_emoji:
    tokens = re.findall(r'\[([^\[\]]+)\]', text)
    all_special_tokens.extend(tokens)

all_special_tokens = sorted(list(set(all_special_tokens)))
special_tokens_list = [f"[{token}]" for token in all_special_tokens]
tokenizer.add_tokens(special_tokens_list)

print(f"Tokenizer vocabulary size after adding special tokens: {len(tokenizer)}")

def train_model(model):
    model.train()

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

            outputs = model(b_input_ids, attention_mask=b_attention_mask)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            predictions.extend(probs.cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())
            predicted_labels.extend((probs >= 0.5).int().cpu().numpy())

    return predictions, true_labels, predicted_labels

def initialize_model():

    torch.manual_seed(SEED)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model.resize_token_embeddings(len(tokenizer))

    torch.manual_seed(SEED)
    if hasattr(model, 'classifier'):
        model.classifier.weight.data.normal_(mean=0.0, std=0.02)
        model.classifier.bias.data.zero_()
    return model

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def reset_model_weights(model):

    torch.manual_seed(SEED)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'reset_parameters'):
        model.classifier.reset_parameters()

best_f1_no_emoji = 0.0
best_f1_emoji = 0.0
best_model_path_no_emoji = os.path.join(model_save_dir, 'best_model_no_emoji.pt')
best_model_path_emoji = os.path.join(model_save_dir, 'best_model_emoji.pt')
best_fold_no_emoji = -1
best_fold_emoji = -1

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
epochs = 2

all_labels_no_emoji = []
all_preds_no_emoji = []
all_labels_emoji = []
all_preds_emoji = []

all_metrics_no_emoji = []
all_metrics_emoji = []

all_confusion_matrices = []

best_val_preds_no_emoji = None
best_val_preds_emoji = None

best_metrics_no_emoji = []
best_metrics_emoji = []

best_f1_no_emoji_fold = 0.0
best_f1_emoji_fold = 0.0

fold_counter = 0
for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print(f"\nFold {fold + 1}/{skf.n_splits}")
    fold_counter += 1

    def seed_worker(worker_id):  # 0421

        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


    train_texts_no_emoji = texts[train_idx]
    val_texts_no_emoji = texts[val_idx]
    train_texts_emoji = texts_with_emoji[train_idx]
    val_texts_emoji = texts_with_emoji[val_idx]
    train_labels_fold = labels[train_idx]
    val_labels_fold = labels[val_idx]

    train_inputs_no_emoji, train_masks_no_emoji = encode_texts(train_texts_no_emoji, tokenizer)
    val_inputs_no_emoji, val_masks_no_emoji = encode_texts(val_texts_no_emoji, tokenizer)
    train_inputs_emoji, train_masks_emoji = encode_texts(train_texts_emoji, tokenizer)
    val_inputs_emoji, val_masks_emoji = encode_texts(val_texts_emoji, tokenizer)

    train_data_no_emoji = TensorDataset(train_inputs_no_emoji, train_masks_no_emoji,
                                        torch.tensor(train_labels_fold, dtype=torch.long))
    val_data_no_emoji = TensorDataset(val_inputs_no_emoji, val_masks_no_emoji,
                                      torch.tensor(val_labels_fold, dtype=torch.long))
    train_data_emoji = TensorDataset(train_inputs_emoji, train_masks_emoji,
                                     torch.tensor(train_labels_fold, dtype=torch.long))
    val_data_emoji = TensorDataset(val_inputs_emoji, val_masks_emoji, torch.tensor(val_labels_fold, dtype=torch.long))

    train_dataloader_no_emoji = DataLoader(train_data_no_emoji, batch_size=8, shuffle=True, num_workers=4,
                                           pin_memory=True, worker_init_fn=seed_worker,
                                           generator=torch.Generator().manual_seed(SEED))
    val_dataloader_no_emoji = DataLoader(val_data_no_emoji, batch_size=8, num_workers=4, pin_memory=True,
                                         worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))
    train_dataloader_emoji = DataLoader(train_data_emoji, batch_size=8, shuffle=True, num_workers=4, pin_memory=True,
                                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))
    val_dataloader_emoji = DataLoader(val_data_emoji, batch_size=8, num_workers=4, pin_memory=True,
                                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))

    model_no_emoji = initialize_model().to(device)
    model_emoji = initialize_model().to(device)
    optimizer_no_emoji = AdamW(
        model_no_emoji.parameters(),
        lr=1e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.999)
    )
    optimizer_emoji = AdamW(
        model_emoji.parameters(),
        lr=1e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.999)
    )
    model_no_emoji.resize_token_embeddings(len(tokenizer))
    model_emoji.resize_token_embeddings(len(tokenizer))
    model_no_emoji.to(device)
    model_emoji.to(device)
    reset_model_weights(model_no_emoji)
    reset_model_weights(model_emoji)

    optimizer_no_emoji = AdamW(model_no_emoji.parameters(), lr=1e-5, weight_decay=1e-2, betas=(0.9, 0.999))
    optimizer_emoji = AdamW(model_emoji.parameters(), lr=1e-5, weight_decay=1e-2, betas=(0.9, 0.999))

    history_no_emoji = {'train_loss': [], 'val_loss': []}
    history_emoji = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):

        avg_train_loss_no_emoji = train_model(model_no_emoji, train_dataloader_no_emoji, optimizer_no_emoji, device)
        avg_train_loss_emoji = train_model(model_emoji, train_dataloader_emoji, optimizer_emoji, device)

        avg_val_loss_no_emoji = evaluate_loss(model_no_emoji, val_dataloader_no_emoji, device)
        avg_val_loss_emoji = evaluate_loss(model_emoji, val_dataloader_emoji, device)
        print(
            f"Fold {fold + 1} Epoch {epoch + 1} (No Emoji) - Train Loss: {avg_train_loss_no_emoji:.4f}, Val Loss: {avg_val_loss_no_emoji:.4f}")
        print(
            f"Fold {fold + 1} Epoch {epoch + 1} (With Emoji) - Train Loss: {avg_train_loss_emoji:.4f}, Val Loss: {avg_val_loss_emoji:.4f}")
        history_no_emoji['train_loss'].append(avg_train_loss_no_emoji)
        history_no_emoji['val_loss'].append(avg_val_loss_no_emoji)
        history_emoji['train_loss'].append(avg_train_loss_emoji)
        history_emoji['val_loss'].append(avg_val_loss_emoji)

        preds_no_emoji, labels_no_fold, predicted_labels_no_fold = evaluate(model_no_emoji, val_dataloader_no_emoji,
                                                                            device)
        preds_emoji, labels_emoji_fold, predicted_labels_emoji_fold = evaluate(model_emoji, val_dataloader_emoji,
                                                                               device)

        accuracy_no_emoji = accuracy_score(labels_no_fold, predicted_labels_no_fold)
        accuracy_emoji = accuracy_score(labels_emoji_fold, predicted_labels_emoji_fold)
        predicted_labels_no_emoji = [1 if p >= 0.5 else 0 for p in preds_no_emoji]
        predicted_labels_emoji = [1 if p >= 0.5 else 0 for p in preds_emoji]
        f1_no_emoji = f1_score(labels_no_fold, predicted_labels_no_emoji)
        f1_emoji = f1_score(labels_emoji_fold, predicted_labels_emoji)
        print(f"Fold {fold + 1} Epoch {epoch + 1} (No Emoji) - F1: {f1_no_emoji:.4f}")
        print(f"Fold {fold + 1} Epoch {epoch + 1} (With Emoji) - F1: {f1_emoji:.4f}")

        if f1_no_emoji > best_f1_no_emoji:
            best_f1_no_emoji = f1_no_emoji
            best_fold_no_emoji = fold + 1
            torch.save(model_no_emoji.state_dict(), best_model_path_no_emoji)
            print(f"Saved best no emoji model for fold {fold + 1} at epoch {epoch + 1} with F1: {best_f1_no_emoji:.4f}")
            val_metrics_no_emoji = {
                'fold': fold + 1,
                'accuracy': accuracy_no_emoji,
                'precision': precision_score(labels_no_fold, predicted_labels_no_emoji),
                'recall': recall_score(labels_no_fold, predicted_labels_no_emoji),
                'f1': f1_no_emoji,
                'auc': roc_auc_score(labels_no_fold, preds_no_emoji)
            }
            pd.DataFrame([val_metrics_no_emoji]).to_excel(
                os.path.join(figure_dir, 'your path'),
                index=False
            )
            best_metrics_no_emoji.append(val_metrics_no_emoji)

            best_val_preds_no_emoji = pd.DataFrame({
                'user_id_original': df.iloc[val_idx]['user ID original'].values,
                'count': df.iloc[val_idx]['count'].values,
                'text': val_texts_no_emoji,
                'true_label': val_labels_fold,
                'predicted_label': predicted_labels_no_emoji,
                'model_type': 'no_emoji',
                'fold': fold + 1
            })

        if f1_emoji > best_f1_emoji:
            best_f1_emoji = f1_emoji
            best_fold_emoji = fold + 1
            torch.save(model_emoji.state_dict(), best_model_path_emoji)
            print(f"Saved best with emoji model for fold {fold + 1} at epoch {epoch + 1} with F1: {best_f1_emoji:.4f}")

            val_metrics_emoji = {
                'fold': fold + 1,
                'accuracy': accuracy_emoji,
                'precision': precision_score(labels_emoji_fold, predicted_labels_emoji),
                'recall': recall_score(labels_emoji_fold, predicted_labels_emoji),
                'f1': f1_emoji,
                'auc': roc_auc_score(labels_emoji_fold, preds_emoji)
            }
            pd.DataFrame([val_metrics_emoji]).to_excel(
                os.path.join(figure_dir, 'your path'),
                index=False
            )
            best_metrics_emoji.append(val_metrics_emoji)

            best_val_preds_emoji = pd.DataFrame({
                'user_id_original': df.iloc[val_idx]['user ID original'].values,
                'count': df.iloc[val_idx]['count'].values,
                'text': val_texts_emoji,
                'true_label': val_labels_fold,
                'predicted_label': predicted_labels_emoji,
                'model_type': 'with_emoji',
                'fold': fold + 1
            })

    preds_no_emoji, labels_no_fold, predicted_labels_no_fold = evaluate(model_no_emoji, val_dataloader_no_emoji, device)
    preds_emoji, labels_emoji_fold, predicted_labels_emoji_fold = evaluate(model_emoji, val_dataloader_emoji, device)
    all_labels_no_emoji.extend(labels_no_fold)
    all_preds_no_emoji.extend(preds_no_emoji)
    all_labels_emoji.extend(labels_emoji_fold)
    all_preds_emoji.extend(preds_emoji)
    predicted_labels_no_emoji = [1 if p >= 0.5 else 0 for p in preds_no_emoji]
    predicted_labels_emoji = [1 if p >= 0.5 else 0 for p in preds_emoji]


    val_metrics_no_emoji = {
        'fold': fold + 1,
        'accuracy': accuracy_no_emoji,
        'precision': precision_score(labels_no_fold, predicted_labels_no_emoji),
        'recall': recall_score(labels_no_fold, predicted_labels_no_emoji),
        'f1': f1_no_emoji,
        'auc': roc_auc_score(labels_no_fold, preds_no_emoji)
    }
    all_metrics_no_emoji.append(val_metrics_no_emoji)

    val_metrics_emoji = {
        'fold': fold + 1,
        'accuracy': accuracy_emoji,
        'precision': precision_score(labels_emoji_fold, predicted_labels_emoji),
        'recall': recall_score(labels_emoji_fold, predicted_labels_emoji),
        'f1': f1_emoji,
        'auc': roc_auc_score(labels_emoji_fold, preds_emoji)
    }
    all_metrics_emoji.append(val_metrics_emoji)

def save_metrics_to_excel(metrics_list, model_type, figure_dir, best_fold):

    df = pd.DataFrame(metrics_list)
    df['best_fold'] = df['fold'] == best_fold

    output_path = os.path.join(figure_dir, f'cross_validation_metrics_{model_type}.xlsx')
    df.to_excel(output_path, index=False)
    print(f"交叉验证指标已保存到 {output_path}")

save_metrics_to_excel(all_metrics_no_emoji, 'no_emoji', figure_dir, best_fold_no_emoji)
save_metrics_to_excel(all_metrics_emoji, 'with_emoji', figure_dir, best_fold_emoji)

if best_val_preds_no_emoji is not None:
    best_val_preds_no_emoji.to_excel(os.path.join(figure_dir, 'your path'), index=False)
    print("...")

if best_val_preds_emoji is not None:
    best_val_preds_emoji.to_excel(os.path.join(figure_dir, 'your path'), index=False)
    print("...")

model_no_emoji = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model_no_emoji.resize_token_embeddings(len(tokenizer))
model_no_emoji.load_state_dict(torch.load(best_model_path_no_emoji))
model_no_emoji.to(device)
model_no_emoji.eval()


preds_no_emoji, labels_no_fold, predicted_labels_no_emoji = evaluate(model_no_emoji, val_dataloader_no_emoji, device)

model_emoji = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model_emoji.resize_token_embeddings(len(tokenizer))
model_emoji.load_state_dict(torch.load(best_model_path_emoji))
model_emoji.to(device)
model_emoji.eval()

preds_emoji, labels_emoji_fold, predicted_labels_emoji = evaluate(model_emoji, val_dataloader_emoji, device)


test_file_path = 'your path'
test_df = pd.read_excel(test_file_path)
test_df['text-emoji'] = test_df['text-emoji'].fillna('')

test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

test_df['text_with_emoji'] = test_df['text-emoji'].apply(lambda x: replace_emoji_with_special_token(x, mapping_dict))


test_inputs, test_masks = encode_texts(test_df['text_with_emoji'], tokenizer)
test_labels = test_df['label'].values.astype(int)
test_data = TensorDataset(test_inputs, test_masks, torch.tensor(test_labels, dtype=torch.long))

test_dataloader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(SEED)
)

test_inputs_no_emoji, test_masks_no_emoji = encode_texts(test_df['text'], tokenizer)
test_data_no_emoji = TensorDataset(test_inputs_no_emoji, test_masks_no_emoji,
                                   torch.tensor(test_labels, dtype=torch.long))

test_dataloader_no_emoji = DataLoader(
    test_data_no_emoji,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(SEED)
)


model_emoji = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model_emoji.resize_token_embeddings(len(tokenizer))
try:
    model_emoji.load_state_dict(torch.load(best_model_path_emoji, map_location=device))
except RuntimeError as e:
    print(f"Error loading weights for emoji model: {e}")
    print("...")
model_emoji.to(device)
model_emoji.eval()


test_preds_emoji, test_labels_emoji, test_predicted_labels_emoji = evaluate(model_emoji, test_dataloader,
                                                                            device)

test_accuracy_emoji = accuracy_score(test_labels_emoji, test_predicted_labels_emoji)
test_precision_emoji = precision_score(test_labels_emoji, test_predicted_labels_emoji)
test_recall_emoji = recall_score(test_labels_emoji, test_predicted_labels_emoji)
test_f1_emoji = f1_score(test_labels_emoji, test_predicted_labels_emoji)
test_auc_emoji = roc_auc_score(test_labels_emoji, test_preds_emoji)

print("Test Set Evaluation (With Emoji):")
print(
    f"Accuracy: {test_accuracy_emoji:.4f}, Precision: {test_precision_emoji:.4f}, Recall: {test_recall_emoji:.4f}, F1: {test_f1_emoji:.4f}")
test_metrics_emoji = {
    'accuracy': test_accuracy_emoji,
    'precision': test_precision_emoji,
    'recall': test_recall_emoji,
    'f1': test_f1_emoji,
    'auc': test_auc_emoji
}
test_metrics_df_emoji = pd.DataFrame([test_metrics_emoji])
test_metrics_df_emoji = pd.DataFrame([test_metrics_emoji])
test_metrics_df_emoji.to_excel(os.path.join(figure_dir, 'your path'), index=False)

model_no_emoji = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model_no_emoji.resize_token_embeddings(len(tokenizer))
try:
    model_no_emoji.load_state_dict(torch.load(best_model_path_no_emoji, map_location=device))
except RuntimeError as e:
    print(f"Error loading weights for no-emoji model: {e}")
    print("....")
model_no_emoji.to(device)
model_no_emoji.eval()

test_preds_no_emoji, test_labels_no_emoji, test_predicted_labels_no_emoji = evaluate(model_no_emoji,
                                                                                     test_dataloader_no_emoji, device)

test_accuracy_no_emoji = accuracy_score(test_labels_no_emoji, test_predicted_labels_no_emoji)
test_precision_no_emoji = precision_score(test_labels_no_emoji, test_predicted_labels_no_emoji)
test_recall_no_emoji = recall_score(test_labels_no_emoji, test_predicted_labels_no_emoji)
test_f1_no_emoji = f1_score(test_labels_no_emoji, test_predicted_labels_no_emoji)
test_auc_no_emoji = roc_auc_score(test_labels_no_emoji, test_preds_no_emoji)

print("Test Set Evaluation (No Emoji):")
print(
    f"Accuracy: {test_accuracy_no_emoji:.4f}, Precision: {test_precision_no_emoji:.4f}, Recall: {test_recall_no_emoji:.4f}, F1: {test_f1_no_emoji:.4f}")

test_metrics_no_emoji = {
    'accuracy': test_accuracy_no_emoji,
    'precision': test_precision_no_emoji,
    'recall': test_recall_no_emoji,
    'f1': test_f1_no_emoji,
    'auc': test_auc_no_emoji
}
test_metrics_df_no_emoji = pd.DataFrame([test_metrics_no_emoji])
test_metrics_df_no_emoji = pd.DataFrame([test_metrics_no_emoji])
test_metrics_df_no_emoji.to_excel(os.path.join(figure_dir, 'your path'), index=False)


test_df['predicted_label_no_emoji'] = test_predicted_labels_no_emoji
test_df['predicted_label_with_emoji'] = test_predicted_labels_emoji

test_output_path = os.path.join(figure_dir, 'your path')
test_df.to_excel(test_output_path, index=False)
print(f"have reserve {test_output_path}")



