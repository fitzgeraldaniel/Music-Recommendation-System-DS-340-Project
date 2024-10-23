from utils.argument import parse_args
from utils.utils import set_random_seeds, setup_logger, config2string
from datetime import datetime
import numpy as np
import os
import torch
import yaml
import pandas as pd  # Import pandas for data handling


# Sample MUSE_Trainer class (for demonstration purposes)
class MUSE_Trainer:
    def __init__(self, args):
        self.args = args
        self.data = None  # Placeholder for dataset
        self.model = None  # Placeholder for model
        self.ckpt_path = './checkpoints'  # Example checkpoint path

    def load_dataset(self):
        print("Loading dataset...")
        self.data = pd.DataFrame({'data': range(100)})  # Replace with actual loading logic

    def load_model(self):
        print("Loading model...")
        self.model = 'dummy_model'  # Replace with actual model loading logic

    def fit(self):
        print("Fitting the model...")
        # Simulate returning some performance metrics
        return {
            'nonshuffle': {'recall': 0.8, 'mrr': 0.75, 'ndcg': 0.7},
            'shuffle': {'recall': 0.78, 'mrr': 0.74, 'ndcg': 0.69},
            'all': {'recall': 0.79, 'mrr': 0.76, 'ndcg': 0.71}
        }








def main():
    args = parse_args()
    torch.set_num_threads(4)
    model_name = args.embedder.lower()

    ns_recall_list = []
    ns_mrr_list = []
    ns_ndcg_list = []

    s_recall_list = []
    s_mrr_list = []
    s_ndcg_list = []

    all_recall_list = []
    all_mrr_list = []
    all_ndcg_list = []

    if 'muse' in args.embedder.lower():
        log_info_path = f'./logs_n_runs_{args.n_runs}/{args.dataset}'  # Logging

    os.makedirs(log_info_path, exist_ok=True)
    logger = setup_logger(name=args.embedder, log_dir=log_info_path, filename=f'./{args.embedder}.txt')
    logger_all = setup_logger(name=args.embedder, log_dir=log_info_path, filename='all_logs.txt')
    config_str = config2string(args)
    inference = 'all'

    # Load dataset and sample
    if hasattr(args, 'sample_size'):
        sample_size = args.sample_size  # Get sample size from args
    else:
        sample_size = 0.1  # Default to 10% if not specified

    for seed in range(args.seed, args.seed + args.n_runs):
        print(f'Dataset: {args.dataset}, Inference: {inference}, Seed: {seed}, Model: {model_name}')
        set_random_seeds(seed)
        args.seed = seed

        if 'best' in model_name:
            with open(f'config/{args.embedder[:-5]}_mssd.yaml', 'r') as f:
                hyperparams = yaml.safe_load(f)
                for k, v in hyperparams.items():
                    setattr(args, k, v)
            model_name = model_name[:-5]

        if model_name == 'muse':
            trainer = MUSE_Trainer
        else:
            raise NotImplementedError

        embedder = trainer(args)

        if args.inference:
            with open(f'config/{args.embedder}_mssd.yaml', 'r') as f:
                hyperparams = yaml.safe_load(f)
                for k, v in hyperparams.items():
                    setattr(args, k, v)
            # Load and sample dataset
            embedder.load_dataset()
            data = embedder.data  # Assuming this is where your dataset is loaded
            if isinstance(data, pd.DataFrame):  # Check if it's a pandas DataFrame
                data = data.sample(frac=sample_size, random_state=seed)  # Sample data
            embedder.load_model()
            embedder.model.load_state_dict(torch.load(f'{embedder.ckpt_path}/{args.embedder}_final_model.pt'))



        else:
            # Ensure the fit method exists
            if not hasattr(embedder, 'fit'):
                raise AttributeError(f"'{type(embedder).__name__}' object has no attribute 'fit'")

            # Call the fit method instead of train
            overall_performance = embedder.fit()  # Call the fit method





            

        _ns_recall = overall_performance['nonshuffle']['recall']
        _ns_mrr = overall_performance['nonshuffle']['mrr']
        _ns_ndcg = overall_performance['nonshuffle']['ndcg']

        ns_recall_list.append(_ns_recall)
        ns_mrr_list.append(_ns_mrr)
        ns_ndcg_list.append(_ns_ndcg)

        _s_recall = overall_performance['shuffle']['recall']
        _s_mrr = overall_performance['shuffle']['mrr']
        _s_ndcg = overall_performance['shuffle']['ndcg']

        s_recall_list.append(_s_recall)
        s_mrr_list.append(_s_mrr)
        s_ndcg_list.append(_s_ndcg)

        _all_recall = overall_performance['all']['recall']
        _all_mrr = overall_performance['all']['mrr']
        _all_ndcg = overall_performance['all']['ndcg']

        all_recall_list.append(_all_recall)
        all_mrr_list.append(_all_mrr)
        all_ndcg_list.append(_all_ndcg)

        st = f'{datetime.now()}\n'
        st += f'[Config] {config_str}\n'
        st += f'-------------------- Top K: {args.topk} --------------------\n'
        st += f'***** [Seed {args.seed}] Test Results (Non-Shuffle) *****\n'
        st += f'Recall: {_ns_recall}\n'
        st += f'MRR: {_ns_mrr}\n'
        st += f'NDCG: {_ns_ndcg}\n'
        
        st += f'***** [Seed {args.seed}] Test Results (Shuffle) *****\n'
        st += f'Recall: {_s_recall}\n'
        st += f'MRR: {_s_mrr}\n'
        st += f'NDCG: {_s_ndcg}\n'

        st += f'***** [Seed {args.seed}] Test Results (All) *****\n'
        st += f'Recall: {_all_recall}\n'
        st += f'MRR: {_all_mrr}\n'
        st += f'NDCG: {_all_ndcg}\n'

        st += f'================================='

        logger.info(st)
        logger_all.info(st)
    









if __name__ == '__main__':
    main()
