import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pandas as pd
import numpy as np

import os

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, f1_score

from transformers import (BertTokenizerFast,
                          BertModel,
                          get_linear_schedule_with_warmup,
                          PreTrainedTokenizer,
                          AutoModel)

from tqdm.auto import tqdm

MODEL_NAME = 'bert-base-uncased'
EPOCHS = 3
LR = 0.0001
FREEZE_LAYERS = 10

BATCH_SIZE = 16
RANDOM_STATE = 42


PROJECT_FOLDER = '/content/drive/MyDrive/Fake_news_Bert/'
FAKE_DATA_PATH = os.path.join(PROJECT_FOLDER, 'Fake.csv')
TRUE_DATA_PATH = os.path.join(PROJECT_FOLDER, 'True.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class TextProcessor:
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.label2idx = {}
        self.idx2label = {}


    def encode_labels(self, labels: pd.Series, is_train: bool=True) -> torch.Tensor:
        if is_train:
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.label2idx = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        else:
            encoded_labels = self.label_encoder.transform(labels)
        return torch.tensor(encoded_labels, dtype=torch.long)


    def tokenize(self, text: str) -> dict:
        return self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

class CustomClassificationDataset(Dataset):
  def __init__(
               self,
               df: pd.DataFrame,
               text_col: str,
               label_col: str,
               processor: TextProcessor,
               is_train: bool
               ) -> None:

    self.df = df
    self.texts = df[text_col].tolist()
    self.processor = processor
    self.is_train = is_train
    self.label = self._encode_labels(df[label_col].astype(str), self.is_train)



  def _encode_labels(self, labels: pd.Series, is_train) -> torch.Tensor:
    return self.processor.encode_labels(labels, is_train)


  def __len__(self) -> int:
    return len(self.df)


  def __getitem__(self, idx: int) -> dict:
    text = str(self.texts[idx])
    encoding = self.processor.tokenize(text)

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label': self.label[idx].flatten()
    }

class CustomBertClassifier(nn.Module):
  def __init__(self, model_name: str, num_labels: int, freeze_layers: int=6, pooler_active: bool=True) -> None:
    super().__init__()
    self.bert = AutoModel.from_pretrained(model_name)
    self.num_labels = num_labels
    self.freeze_layers = self._set_freeze_layers(freeze_layers)


    for param in self.bert.parameters():
      param.requires_grad = False

    for layer in self.bert.encoder.layer[self.freeze_layers:]:
      for param in layer.parameters():
        param.requires_grad = True

    if pooler_active:
      for name, param in self.bert.pooler.named_parameters():
        if "dense" in name:
          param.requires_grad = True


    self.hidden_size = self.bert.pooler.dense.out_features
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(self.hidden_size, self.num_labels)


  def _set_freeze_layers(self, freeze_layers: int) -> int:
    return min(self.bert.encoder.config.num_hidden_layers, freeze_layers)


  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_embedding = outputs.pooler_output
    cls_embedding = self.dropout(cls_embedding)
    logits = self.classifier(cls_embedding)

    return logits

def train(
    train_dataloader,
    test_dataloader,
    device,
    model,
    loss_fn,
    optimizer,
    scheduler
    ):
    for _ in range(EPOCHS):
        train_metrics = train_one_epoch(
            train_dataloader,
            device,
            model,
            loss_fn,
            optimizer,
            scheduler
            )
        test_metrics = test(
            test_dataloader,
            device,
            loss_fn,
            model)
        scheduler.step()

        print(f'Train: {train_metrics}')
        print(f'Test: {test_metrics}')


def train_one_epoch(
    train_dataloader,
    device,
    model,
    loss_fn,
    optimizer,
    scheduler
    ):
  predictions_list = []
  targets_list = []
  total_loss = 0

  for batch in tqdm(train_dataloader, desc='Train'):
    optimizer.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].squeeze(-1).to(device)


    logits = model(input_ids, attention_mask)

    loss = loss_fn(logits, labels)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    optimizer.step()

    total_loss += loss.item()

    preds = torch.argmax(logits, dim=1).cpu().numpy()

    predictions_list.extend(preds)
    targets_list.extend(labels.detach().cpu().numpy())

  total_loss = total_loss / len(train_dataloader)

  return metrics_calc(total_loss, predictions_list, targets_list)


def test(
    test_dataloader,
    device,
    loss_fn,
    model
    ):
    predictions_list = []
    targets_list = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Test'):

          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['label'].squeeze(-1).to(device)


          logits = model(input_ids, attention_mask)

          loss = loss_fn(logits, labels)

          total_loss += loss.item()

          preds = torch.argmax(logits, dim=1).cpu().numpy()

          predictions_list.extend(preds)
          targets_list.extend(labels.cpu().numpy())

        total_loss = total_loss / len(test_dataloader)

    return metrics_calc(total_loss, predictions_list, targets_list)


def metrics_calc(loss, predictions, targets):

  return {
      'loss': loss,
      'accuracy': accuracy_score(targets, predictions),
      'f1_score': f1_score(targets, predictions, average='weighted')
  }

def shuffle_and_concat_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame
    ) -> pd.DataFrame:

    # shuffled_df1 = df1.sample(frac=1).reset_index(drop=True)
    # shuffled_df2 = df2.sample(frac=1).reset_index(drop=True)
    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    shuffled_concatenated_df = shuffle(concatenated_df)

    return shuffled_concatenated_df


def concat_columns(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    sep: str = ' ',
    na_rep: str = ''
    ) -> pd.Series:

    col1_clean = df[col1].astype(str).fillna(na_rep)
    col2_clean = df[col2].astype(str).fillna(na_rep)

    df['concat_col'] = col1_clean + sep + col2_clean

    return pd.DataFrame(df['concat_col'])


def add_label_column(
    df: pd.DataFrame,
    value: int=0
    ) -> pd.DataFrame:

    df['label'] = value
    return df

fake = pd.read_csv(FAKE_DATA_PATH,
                   sep=',',
                   on_bad_lines='skip',
                   quotechar='"',
                   escapechar='\\',
                   engine='python' )

true = pd.read_csv(TRUE_DATA_PATH,
                   sep=',',
                   on_bad_lines='skip',
                   quotechar='"',
                   escapechar='\\',
                   engine='python' )

fake = add_label_column(concat_columns(fake, 'title', 'text'), 0)
true = add_label_column(concat_columns(true, 'title', 'text'), 1)
shuffled_news = shuffle_and_concat_datasets(fake, true)
shuffled_news = shuffled_news.rename(columns={'concat_col': 'text'})
# shuffled_news = shuffled_news.head(1000)

shuffled_news.shape

train_df, temp_df = train_test_split(
    shuffled_news,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=shuffled_news['label']
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_df['label']
)


processor = TextProcessor(MODEL_NAME, 512)
train_dataset = CustomClassificationDataset(train_df, 'text', 'label', processor, is_train=True)
val_dataset = CustomClassificationDataset(val_df, 'text', 'label', processor, is_train=False)
test_dataset = CustomClassificationDataset(test_df, 'text', 'label', processor, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


num_labels = len(processor.label2idx)
model = CustomBertClassifier(MODEL_NAME, num_labels, freeze_layers=FREEZE_LAYERS)
model.to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

train(train_loader, val_loader, device, model, loss_fn, optimizer, scheduler)

test(test_loader, device, loss_fn, model)

def predict_news(
    text: str,
    model: torch.nn.Module,
    processor: TextProcessor,
    device: str = "cpu"
) -> dict:

    encoding = processor.tokenize(text)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)


    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        predicted_idx = torch.argmax(probabilities).item()


    predicted_class = processor.idx2label[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities.cpu().numpy().tolist(),
    }

model.eval()

processor = TextProcessor(tokenizer_name="bert-base-uncased", max_length=512)


text = """ONLY TODAY YOU CAN BECOME THE PRESIDENT OF UNITED STATES OF AMERICA"""
result = predict_news(text=text, model=model, processor=processor, device="cpu")

print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

text_truly_fake = shuffled_news.tail(10)
text_truly_fake = text_truly_fake.loc[text_truly_fake['label'] == 0].head(1).values.tolist()[0][0]

text_truly_true = shuffled_news.tail(10)
text_truly_true = text_truly_true.loc[text_truly_true['label'] == 1].head(1).values.tolist()[0][0]

result = predict_news(text=text_truly_true, model=model, processor=processor, device="cpu")

print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

torch.save(model.state_dict(), "model_weights.pth")