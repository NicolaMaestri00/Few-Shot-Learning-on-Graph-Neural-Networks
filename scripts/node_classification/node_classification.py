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
    data, shots =  data_modules.nc_get_data(device=device, **config["data"])

    # Baseline
    functions.nc_baseline(config, data, shots, device, results_path=config["results_path"])

    # Regularization Strategies
    functions.nc_regularization(config, data, shots, device, results_path=config["results_path"], csr=True)    # Cosine Similarity Regularization
    functions.nc_regularization(config, data, shots, device, results_path=config["results_path"], er=True)     # Entropy Regularization

    # Augmentation Strategies
    functions.nc_drop_message(config, data, shots, device, results_path=config["results_path"])                # DropMessage
    functions.nc_dropblock(config, data, shots, device, "DropNode", results_path=config["results_path"])         # DropNode
    functions.nc_dropblock(config, data, shots, device, "DropEdge", results_path=config["results_path"])         # DropEdge
    functions.nc_dropblock(config, data, shots, device, "DropAttributes", results_path=config["results_path"])   # DropAttributes
    functions.nc_augmentation(config, data, shots, device, results_path=config["results_path"])                # Augmentation
    functions.nc_rewiring(config, data, shots, device, results_path=config["results_path"])                    # Rewiring

    # Pre-training and Fine-tuning
    functions.nc_graphcl(config, data, shots, device, results_path=config["results_path"])        # GraphCL

    # Prototypical Networks
    functions.nc_protonet(config, data, shots, device, results_path=config["results_path"])       # Prototypical Networks

    # Siamese Networks
    functions.nc_siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='PW')        # Pairwise Loss
    functions.nc_siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='PW_A')      # Pairwise Loss with Augmentation
    functions.nc_siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='TPL')       # Triplet Loss
    functions.nc_siamesenet(config, data, shots, device, results_path=config["results_path"], CL_Loss='TPL_A')     # Triplet Loss with Augmentation

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'='*20} {config['data']['dataset_name']} - {end_time} - {device} {'='*20}\n\n")
