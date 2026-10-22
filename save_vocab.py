import os
from transformers import DistilBertTokenizer

os.makedirs("model", exist_ok=True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_vocabulary("model")
tokenizer.save_pretrained("tokenizer")
print(type(tokenizer))
print("Vocab file:", tokenizer.vocab_file)

print("Saved in /model folder")