from utils.argument import parse_args
from utils.utils import set_random_seeds, setup_logger, config2string
from utils.data import load_data, CollateFn  # Ensure CollateFn is imported
from datetime import datetime
import torch
import yaml
import os
from models.MUSE import MUSE_Trainer
import numpy as np  # Import numpy for calculations

def main():
    args = parse_args()
    torch.set_num_threads(4)
    model_name = args.embedder.lower()

    # Lists to store metrics across runs
    ns_recall_list, ns_mrr_list, ns_ndcg_list = [], [], []
    s_recall_list, s_mrr_list, s_ndcg_list = [], [], []
    all_recall_list, all_mrr_list, all_ndcg_list = [], [], []

    # Set up logging directory and logger
    if 'muse' in model_name:
        log_info_path = f'./logs_n_runs_{args.n_runs}/{args.dataset}'
        os.makedirs(log_info_path, exist_ok=True)

    logger = setup_logger(name=args.embedder, log_dir=log_info_path, filename=f'{args.embedder}.txt')
    logger_all = setup_logger(name=args.embedder, log_dir=log_info_path, filename='all_logs.txt')
    config_str = config2string(args)

    # Load dataset with load_full=False for testing purposes
    train_data, valid_data, n_item, sess_map = load_data(args.data_root, load_full=False)

    # Loop over seeds for multiple runs
    for seed in range(args.seed, args.seed + args.n_runs):
        print(f'Dataset: {args.dataset}, Inference: all, Seed: {seed}, Model: {model_name}')
        set_random_seeds(seed)
        args.seed = seed

        # Load hyperparameters if using a specific model configuration
        if 'best' in model_name:
            with open(f'config/{args.embedder[:-5]}_mssd.yaml', 'r') as f:
                hyperparams = yaml.safe_load(f)
                for k, v in hyperparams.items():
                    setattr(args, k, v)
            model_name = model_name[:-5]

        # Initialize the trainer with the actual MUSE_Trainer, passing n_items
        embedder = MUSE_Trainer(args, n_item)
        
        # Load the model
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

        # Training phase: pass train_loader and valid_loader to fit
        overall_performance = embedder.fit(train_loader, valid_loader)

        # Collect and log metrics
        ns_recall_list.append(overall_performance['nonshuffle']['recall'])
        ns_mrr_list.append(overall_performance['nonshuffle']['mrr'])
        ns_ndcg_list.append(overall_performance['nonshuffle']['ndcg'])

        s_recall_list.append(overall_performance['shuffle']['recall'])
        s_mrr_list.append(overall_performance['shuffle']['mrr'])
        s_ndcg_list.append(overall_performance['shuffle']['ndcg'])

        all_recall_list.append(overall_performance['all']['recall'])
        all_mrr_list.append(overall_performance['all']['mrr'])
        all_ndcg_list.append(overall_performance['all']['ndcg'])

        # Logging for individual runs
        st = f'{datetime.now()}\n'
        st += f'[Config] {config_str}\n'
        st += f'-------------------- Top K: {args.topk} --------------------\n'
        st += f'***** [Seed {args.seed}] Test Results (Non-Shuffle) *****\n'
        st += f'Recall: {overall_performance["nonshuffle"]["recall"]}\n'
        st += f'MRR: {overall_performance["nonshuffle"]["mrr"]}\n'
        st += f'NDCG: {overall_performance["nonshuffle"]["ndcg"]}\n'
        
        st += f'***** [Seed {args.seed}] Test Results (Shuffle) *****\n'
        st += f'Recall: {overall_performance["shuffle"]["recall"]}\n'
        st += f'MRR: {overall_performance["shuffle"]["mrr"]}\n'
        st += f'NDCG: {overall_performance["shuffle"]["ndcg"]}\n'

        st += f'***** [Seed {args.seed}] Test Results (All) *****\n'
        st += f'Recall: {overall_performance["all"]["recall"]}\n'
        st += f'MRR: {overall_performance["all"]["mrr"]}\n'
        st += f'NDCG: {overall_performance["all"]["ndcg"]}\n'

        st += f'================================='

        logger.info(st)
        logger_all.info(st)

    # After all runs, calculate and print average metrics
    avg_ns_recall = sum(ns_recall_list) / len(ns_recall_list) if ns_recall_list else 0
    avg_ns_mrr = sum(ns_mrr_list) / len(ns_mrr_list) if ns_mrr_list else 0
    avg_ns_ndcg = sum(ns_ndcg_list) / len(ns_ndcg_list) if ns_ndcg_list else 0

    avg_s_recall = sum(s_recall_list) / len(s_recall_list) if s_recall_list else 0
    avg_s_mrr = sum(s_mrr_list) / len(s_mrr_list) if s_mrr_list else 0
    avg_s_ndcg = sum(s_ndcg_list) / len(s_ndcg_list) if s_ndcg_list else 0

    avg_all_recall = sum(all_recall_list) / len(all_recall_list) if all_recall_list else 0
    avg_all_mrr = sum(all_mrr_list) / len(all_mrr_list) if all_mrr_list else 0
    avg_all_ndcg = sum(all_ndcg_list) / len(all_ndcg_list) if all_ndcg_list else 0

    # Print summary of average metrics
    print(f'Average Metrics Across Runs:')
    print(f'Non-Shuffle: Recall: {avg_ns_recall}, MRR: {avg_ns_mrr}, NDCG: {avg_ns_ndcg}')
    print(f'Shuffle: Recall: {avg_s_recall}, MRR: {avg_s_mrr}, NDCG: {avg_s_ndcg}')
    print(f'All: Recall: {avg_all_recall}, MRR: {avg_all_mrr}, NDCG: {avg_all_ndcg}')

if __name__ == '__main__':
    main()
