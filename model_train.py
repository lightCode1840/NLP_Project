import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

class LyricDataset(Dataset):
    def __init__(self, texts):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=128)
        return {'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask'])}

class LyricGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.rhyme_layer = torch.nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        rhyme_features = self.rhyme_layer(outputs[0])
        return outputs.logits, rhyme_features

def train():
    # 示例训练数据
    dataset = LyricDataset(["示例歌词1", "示例歌词2"])
    dataloader = DataLoader(dataset, batch_size=2)
    
    model = LyricGenerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in dataloader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = torch.nn.functional.cross_entropy(outputs[0].view(-1, outputs[0].size(-1)), batch['input_ids'].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Loss: {loss.item()}')

if __name__ == "__main__":
    train()