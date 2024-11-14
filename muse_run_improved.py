import torch
import torch.nn as nn
import pandas as pd
import re
import logging
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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

# Define the Embedding Layer with skip prediction embedding
class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim, skip_embedding_dim=16):
        super(EmbeddingLayer, self).__init__()
        self.item_embedding = nn.Embedding(input_dim, embedding_dim)
        self.skip_embedding = nn.Embedding(2, skip_embedding_dim)  # 0 or 1 for skip status

    def forward(self, x, skip_status):
        item_embedded = self.item_embedding(x)
        
        # Ensure skip_status has the same length as x, else ignore skip embedding
        if skip_status.size(1) == x.size(1):
            skip_embedded = self.skip_embedding(skip_status)
            combined_embedded = torch.cat((item_embedded, skip_embedded), dim=2)
        else:
            combined_embedded = item_embedded  # Ignore skip embedding if lengths don’t match
            
        return combined_embedded

# Define the Session Encoder with correct combined embedding dimension
class SessionEncoder(nn.Module):
    def __init__(self, combined_embedding_dim, hidden_dim):
        super(SessionEncoder, self).__init__()
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

# Define the MUSE Model
class MUSE_Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, skip_embedding_dim=16):
        super(MUSE_Model, self).__init__()
        self.embedding = EmbeddingLayer(input_dim, embedding_dim, skip_embedding_dim)
        combined_embedding_dim = embedding_dim + skip_embedding_dim  # Now 64 + 16 = 80
        self.encoder = SessionEncoder(combined_embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, skip_status):
        embedded = self.embedding(x, skip_status)
        encoded = self.encoder(embedded)
        output = self.fc(encoded)
        return output

# Custom collate function for padding sequences and skip labels
def collate_fn(batch):
    session_data, labels, skip_statuses = zip(*batch)
    session_data = [torch.tensor(s, dtype=torch.long) for s in session_data]
    skip_statuses = [torch.tensor(s, dtype=torch.long) for s in skip_statuses]
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Pad session_data and skip_statuses sequences to the same length within the batch
    session_data_padded = pad_sequence(session_data, batch_first=True, padding_value=0)
    skip_statuses_padded = pad_sequence(skip_statuses, batch_first=True, padding_value=0)
    return session_data_padded, labels, skip_statuses_padded

# Create DataLoader for training, adding skip labels for each track in the session
def create_data_loader(seq_new, batch_size=32, sample_size=10000):
    logging.info("Creating DataLoader...")
    data = []
    for _, row in seq_new.head(sample_size).iterrows():
        item_ids = row['ItemId']
        session_data = item_ids[:-1]  # All but the last item for input
        label = item_ids[-1]          # Last item as the label for next prediction
        skip_statuses = [1 if not skip else 0 for skip in row['not_skipped'][:-1]]  # 1 if skipped, 0 otherwise
        data.append((session_data, label, skip_statuses))
    
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    logging.info(f"DataLoader created with {len(data)} samples.")
    return data_loader

# Define a function to calculate metrics
def calculate_metrics(model, data_loader):
    logging.info("Calculating metrics...")
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch, skip_statuses in data_loader:
            output = model(x_batch, skip_statuses)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)
    accuracy = total_correct / total_samples
    logging.info(f"Calculated Accuracy: {accuracy:.4f}")
    return {"Accuracy": accuracy}

# Training Loop
def train_model(model, data_loader, loss_fn, optimizer, epochs=5):
    logging.info("Starting training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x_batch, y_batch, skip_statuses) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(x_batch, skip_statuses)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
        
        # Validation at each epoch
        metrics = calculate_metrics(model, data_loader)
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

    data_loader = create_data_loader(seq_new, batch_size=32)

    model = MUSE_Model(input_dim, embedding_dim, hidden_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(model, data_loader, loss_fn, optimizer)
    logging.info("Training Complete.")

    final_metrics = calculate_metrics(model, data_loader)
    logging.info(f"Final Metrics: {final_metrics}")

if __name__ == "__main__":
    main()

import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Simplified DQN model with higher dropout to reduce overfitting
class SkipPredictionDQN(nn.Module):
    def __init__(self, behavior_dim, context_dim, content_dim, hidden_dim=100, output_dim=1):
        super(SkipPredictionDQN, self).__init__()
        
        # Dedicated layers for each feature type
        self.behavior_fc = nn.Linear(behavior_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim)
        self.content_fc = nn.Linear(content_dim, hidden_dim)
        
        # Combined layer and output
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)  # Increased dropout

    def forward(self, behavior_x, context_x, content_x):
        # Process each feature group with ReLU activation
        behavior_out = F.relu(self.behavior_fc(behavior_x))
        context_out = F.relu(self.context_fc(context_x))
        content_out = F.relu(self.content_fc(content_x))
        
        # Concatenate and process through combined layers
        combined = torch.cat((behavior_out, context_out, content_out), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        return torch.sigmoid(self.output(x))  # Sigmoid for binary classification

from sklearn.metrics.pairwise import cosine_similarity
import torch

class ColdStartPredictor:
    def __init__(self, threshold=0.6):
        """
        Initializes the ColdStartPredictor to handle new or infrequent items.
        :param threshold: Similarity threshold to classify items as "similar."
        """
        self.threshold = threshold

    def compute_similarity(self, item_embedding, known_embeddings):
        """
        Computes content similarity between a new item and known items.
        :param item_embedding: Embedding vector of the new item.
        :param known_embeddings: Embeddings of known items.
        :return: Array of similarity scores with known items.
        """
        item_embedding = item_embedding.reshape(1, -1)  # Reshape for compatibility
        similarities = cosine_similarity(item_embedding, known_embeddings)
        return similarities.flatten()

    def predict(self, item_embedding, known_embeddings, known_outputs):
        """
        Predicts based on the most similar known item if the similarity threshold is met.
        :param item_embedding: Embedding vector of the new item.
        :param known_embeddings: Embeddings of items with collaborative data.
        :param known_outputs: Model predictions for known items.
        :return: Prediction based on the most similar known item, or None if no match.
        """
        similarities = self.compute_similarity(item_embedding, known_embeddings)
        max_similarity = max(similarities)

        # Use the most similar known item's output if above threshold
        if max_similarity >= self.threshold:
            similar_index = torch.argmax(torch.tensor(similarities))
            return known_outputs[similar_index]
        else:
            return None  # Return None if no sufficiently similar item is found
