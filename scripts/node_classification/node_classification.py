'''    Node Classification    '''

import datetime

import torch

import data_modules
import functions
import utils


if __name__ == "__main__":

    # Load Config -c path/to/config.yaml
    config = utils.get_config("Node Classification")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'='*20} {config['data']['dataset_name']} - {start_time} - {device} {'='*20}")

    # Load the dataset
    data, shots =  data_modules.get_data(device=device, **config["data"])

    # Baseline
    functions.baseline(config, data, shots, device, results_path=config["results_path"])

    # Regularization Strategies
    functions.reg_baseline(config, data, shots, device, results_path=config["results_path"], csr=True)    # Cosine Similarity Regularization
    functions.reg_baseline(config, data, shots, device, results_path=config["results_path"], er=True)     # Entropy Regularization

    # Augmentation Strategies
    functions.drop_message(config, data, shots, device, results_path=config["results_path"])                # DropMessage
    functions.dropout(config, data, shots, device, "DropNode", results_path=config["results_path"])         # DropNode
    functions.dropout(config, data, shots, device, "DropEdge", results_path=config["results_path"])         # DropEdge
    functions.dropout(config, data, shots, device, "DropAttributes", results_path=config["results_path"])   # DropAttributes
    functions.augmentation(config, data, shots, device, results_path=config["results_path"])                # Augmentation
    functions.rewiring(config, data, shots, device, results_path=config["results_path"])                    # Rewiring

    # Pre-training and Fine-tuning
    functions.graphcl(config, data, shots, device, results_path=config["results_path"])        # GraphCL

    # Prototypical Networks
    functions.protonet(config, data, shots, device, results_path=config["results_path"])       # Prototypical Networks

    # Siamese Networks
    functions.siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='PW')        # Pairwise Loss
    functions.siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='PW_A')      # Pairwise Loss with Augmentation
    functions.siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='TPL')       # Triplet Loss
    functions.siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='TPL_A')     # Triplet Loss with Augmentation

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'='*20} {config['data']['dataset_name']} - {end_time} - {device} {'='*20}\n\n")
