import torch
import torch.nn as nn
import pandas as pd
import re
import logging
from torch.nn.utils.rnn import pad_sequence
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from specified paths
def load_data():
    logging.info("Loading data from CSV files...")
    item_map = pd.read_csv('/Users/drewrubendall/Documents/Fall 2024/DS 340/Project/Original Files/MUSE-main/data/mssd-3d-all-0/all/item_map.csv')
    seq_new = pd.read_csv('/Users/drewrubendall/Documents/Fall 2024/DS 340/Project/Original Files/MUSE-main/data/mssd-3d-all-0/all/seq_new.csv')
    sess_map = pd.read_csv('/Users/drewrubendall/Documents/Fall 2024/DS 340/Project/Original Files/MUSE-main/data/mssd-3d-all-0/all/sess_map.csv')
    
    # Convert list-like columns from string representations to actual lists
    for col in ['not_skipped', 'ItemId', 'hour_of_day', 'day']:
        seq_new[col] = seq_new[col].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
    logging.info("Data loaded and processed successfully.")
    
    return item_map, seq_new, sess_map

# Check the highest item ID in the dataset to set input_dim and output_dim accurately
def get_max_item_id(seq_new):
    max_id = seq_new['ItemId'].explode().max()  # Flatten list column and get max value
    logging.info(f"Max item ID in seq_new data: {max_id}")
    return max_id

# Define the Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Define the Session Encoder
class SessionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SessionEncoder, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

# Define the MUSE Model
class MUSE_Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(MUSE_Model, self).__init__()
        self.embedding = EmbeddingLayer(input_dim, embedding_dim)
        self.encoder = SessionEncoder(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        output = self.fc(encoded)
        return output

# Custom collate function for padding sequences
def collate_fn(batch):
    session_data, labels = zip(*batch)
    session_data = [torch.tensor(s, dtype=torch.long) for s in session_data]  # Convert to tensors if not already
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Pad session_data sequences to the same length within the batch
    session_data_padded = pad_sequence(session_data, batch_first=True, padding_value=0)
    return session_data_padded, labels

# Define a function to create a DataLoader with shuffled sessions
def create_data_loader(seq_new, batch_size=32, sample_size=10000, shuffle_rate=0.5):
    logging.info("Creating DataLoader for shuffle play environment...")
    data = []
    for _, row in seq_new.head(sample_size).iterrows():
        item_ids = row['ItemId']
        
        # Shuffle the session with probability based on `shuffle_rate`
        if random.random() < shuffle_rate:
            random.shuffle(item_ids)
        
        session_data = item_ids[:-1]  # All items except last as input
        label = item_ids[-1]  # Last item as label
        data.append((session_data, label))
    
    # Define DataLoader with custom collate function
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    logging.info(f"DataLoader created with {len(data)} samples (shuffle rate: {shuffle_rate * 100}%).")
    return data_loader

# Define a function to calculate metrics
def calculate_metrics(model, data_loader):
    logging.info("Calculating metrics...")
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            output = model(x_batch)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)
    accuracy = total_correct / total_samples
    logging.info(f"Calculated Accuracy: {accuracy:.4f}")
    return {"Accuracy": accuracy}

# Define a validation function
def validate(model, data_loader):
    metrics = calculate_metrics(model, data_loader)
    return metrics

# Training Loop
def train_model(model, data_loader, loss_fn, optimizer, epochs=4):
    logging.info("Starting training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
        
        # Validation at each epoch
        metrics = validate(model, data_loader)
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(data_loader):.4f}, Validation Metrics: {metrics}")

# Main function
def main():
    item_map, seq_new, sess_map = load_data()

    # Set input_dim and output_dim based on the maximum item ID in seq_new to prevent embedding index errors
    max_item_id = get_max_item_id(seq_new)
    input_dim = max_item_id + 1  # Ensure input_dim covers all item IDs
    output_dim = max_item_id + 1  # Ensure output_dim covers all item IDs
    embedding_dim = 64
    hidden_dim = 128

    data_loader = create_data_loader(seq_new, batch_size=32, shuffle_rate=0.5)  # 50% shuffle rate

    model = MUSE_Model(input_dim, embedding_dim, hidden_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, data_loader, loss_fn, optimizer)
    logging.info("Training Complete.")

    final_metrics = calculate_metrics(model, data_loader)
    logging.info(f"Final Metrics: {final_metrics}")

if __name__ == "__main__":
    main()

