import torch
from model_train import LyricGenerator
from data_preprocess import LyricProcessor

class LyricInferencer:
    def __init__(self, model_path):
        self.model = LyricGenerator()
        self.model.load_state_dict(torch.load(model_path))
        self.processor = LyricProcessor()

    def generate_lyrics(self, source_text, num_lines=4):
        processed = self.processor.process_lyrics(source_text)
        inputs = self.model.tokenizer(processed, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            logits, _ = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    inferencer = LyricInferencer("./saved_model.pth")
    print(inferencer.generate_lyrics("示例源歌词"))