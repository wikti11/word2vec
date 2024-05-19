import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funkcja do wczytywania danych z pliku TSV
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != "":
                data.append(line.strip().split("\t"))
    return pd.DataFrame(data, columns=["label", "text"])

def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts if text.strip()]

# Wczytywanie danych treningowych
train_data = load_data("train/train.tsv")
train_texts = train_data["text"].values
train_labels = train_data["label"].values.astype(int)


# Wczytywanie danych walidacyjnych
dev_texts = load_texts("dev-0/in.tsv")

# Wczytywanie danych testowych
test_texts = load_texts("test-A/in.tsv")

# Trening modelu word2vec na tekstach treningowych
all_texts = np.concatenate((train_texts, dev_texts, test_texts))
tokenized_texts = [text.split() for text in all_texts]

word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Funkcja do konwersji tekstu na wektor
def text_to_vector(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Konwersja tekstów na wektory
X_train = np.array([text_to_vector(text, word2vec_model) for text in train_texts])
y_train = train_labels

X_dev = np.array([text_to_vector(text, word2vec_model) for text in dev_texts])
X_test = np.array([text_to_vector(text, word2vec_model) for text in test_texts])


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


input_dim = X_train.shape[1]
model = SimpleNN(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Konwersja danych do tensora
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

# Trening modelu
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
# Ewaluacja na zbiorze walidacyjnym
model.eval()
X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32).to(device)
with torch.no_grad():
    dev_outputs = model(X_dev_tensor)
    _, dev_preds = torch.max(dev_outputs, 1)

# Wczytanie expected.tsv dla zbioru walidacyjnego
expected_dev = load_texts("dev-0/expected.tsv")
expected_dev = np.array([int(label) for label in expected_dev])
accuracy = accuracy_score(expected_dev, dev_preds.cpu().numpy())
points = np.ceil(accuracy * 7.0)
print(f"Accuracy: {accuracy}, Points: {points}")
# Predykcja dla dev-0
with torch.no_grad():
    dev_outputs = model(X_dev_tensor)
    _, dev_preds = torch.max(dev_outputs, 1)

# Zapis wyników dla dev-0
dev_preds_df = pd.DataFrame(dev_preds.cpu().numpy())
dev_preds_df.to_csv("dev-0/out.tsv", sep='\t', index=False, header=False)

# Predykcja dla test-A
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_preds = torch.max(test_outputs, 1)

# Zapis wyników dla test-A
test_preds_df = pd.DataFrame(test_preds.cpu().numpy())
test_preds_df.to_csv("test-A/out.tsv", sep='\t', index=False, header=False)