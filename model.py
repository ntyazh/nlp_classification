import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny')
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')
        self.n_classes = 8
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, self.n_classes)

    def forward(self, input_id, mask):
        return self.model(input_id, mask)


class BertDataset(Dataset):
    def __init__(self, df, mode='train'):
        tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')
        self.mode = mode
        if mode == 'test':
            self.tokens = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df]
        else:
            self.tokens = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df.text]
            self.labels = df['class'].to_numpy()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.tokens[idx]
        return self.tokens[idx], torch.tensor(self.labels[idx])


def get_result(text: pd.Series) -> pd.Series:
    model = BertClassifier()
    PATH = 'bert0.92948.pth'

    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    dataset = BertDataset(text, mode='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(text))
    with torch.no_grad():
        for val_input in tqdm(loader):
            mask = val_input['attention_mask']
            input_id = val_input['input_ids'].squeeze(1)
            output = model.forward(input_id, mask)
    classes = output[0].argmax(dim=1).numpy()
    inv_enc_dict = {0: "Вежливость сотрудников магазина", 1: "Время ожидания у кассы",
                    2: "Доступность персонала в магазине", 3: "Компетентность продавцов/ консультантов",
                    4: "Консультация КЦ", 5: "Обслуживание на кассе",
                    6: "Обслуживание продавцами/ консультантами", 7: "Электронная очередь"}
    answer = []
    for class_ in classes:
        answer.append(inv_enc_dict[class_])
    return pd.Series(answer, name="class_predicted")
