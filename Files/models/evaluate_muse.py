import torch
import numpy as np
from models.MUSE import MUSE_Trainer
from utils.data import load_data, CollateFn  # Ensure CollateFn is available
from models.metric import evaluate  # Assuming you have a metric evaluation function
import argparse
import os

def main(data_root, batch_size, n_epochs):
    # Load the data
    train_data, valid_data, n_item, sess_map = load_data(data_root)

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    args = argparse.Namespace(
        batch_size=batch_size,
        n_epochs=n_epochs,
        topk=[5, 10],  # Example top-k values
        embedder='MUSE',  # Model name
        device=device
    )
    embedder = MUSE_Trainer(args, n_item)
    
    # Load model
    embedder.load_model()

    # Instantiate CollateFn
    collate_fn_instance = CollateFn(args=args, n_items=n_item)

    # Set up DataLoader for training and validation
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_instance
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_instance
    )

    # Training phase: fit the model
    overall_performance = embedder.fit(train_loader, valid_loader)

    # Calculate metrics
    for key in overall_performance:
        recall = overall_performance[key]['recall']
        mrr = overall_performance[key]['mrr']
        ndcg = overall_performance[key]['ndcg']
        
        print(f'{key.capitalize()} Metrics:')
        print(f'Recall: {recall}')
        print(f'MRR: {mrr}')
        print(f'NDCG: {ndcg}\n')

if __name__ == '__main__':
    # Set data root and parameters
    data_root = '/path/to/your/data'  # Update this path accordingly
    batch_size = 512  # Example batch size
    n_epochs = 30  # Example number of epochs

    main(data_root, batch_size, n_epochs)
