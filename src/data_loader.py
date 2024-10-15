import os

def load_data(file_path):
    sentences, tags = [], []
    sentence, tag = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.split('\t')
                sentence.append(parts[1])  # Token
                tag.append(parts[2].strip())  # POS tag
            else:
                if sentence and tag:
                    sentences.append(sentence)
                    tags.append(tag)
                sentence, tag = [], []
    return sentences, tags

# Load your Uzbek dataset
data_path = os.path.join('data', 'uzb.txt')
sentences, tags = load_data(data_path)
