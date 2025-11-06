import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
        
    def encode(self, text):
        preprocessed= re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        preprocessed = [
            item if item in self.str_to_int else '<|unk|>' for item in preprocessed
        ]
        print(f"Preprocessed tokens: {preprocessed}")
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        # 在 token 之前添加空格
        text = ' '.join(self.int_to_str[i] for i in ids)
        # 删除标点符号之前多余的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
    