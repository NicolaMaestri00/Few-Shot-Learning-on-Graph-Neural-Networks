'''    Graph Classification    '''

import datetime

import torch

import data_modules
import functions
import utils


if __name__ == "__main__":

    # Load Config -c path/to/config.yaml
    config = utils.get_config("Graph Classification")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'='*20} {config['data']['dataset_name']} - {start_time} - {device} {'='*20}")

    # Load the dataset
    dataset, shots, test_dataset, test_loader = data_modules.get_graphs(device=device, **config["data"])

    # Baseline
    functions.baseline_graph(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])

    # Regularization Strategies
    functions.regularized_graph(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], csr=True)
    functions.regularized_graph(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], er=True)

    # Dropout Strategies
    functions.dropout_graph(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])
    functions.dropmessage_graph(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])
    functions.dropstrategy_graph(config=config, shots=shots, test_loader=test_loader, drop_strategy="DropNode", device=device, dataset=dataset, results_path=config["results_path"])
    functions.dropstrategy_graph(config=config, shots=shots, test_loader=test_loader, drop_strategy="DropEdge", device=device, dataset=dataset, results_path=config["results_path"])
    functions.dropstrategy_graph(config=config, shots=shots, test_loader=test_loader, drop_strategy="DropAttributes", device=device, dataset=dataset, results_path=config["results_path"])
    functions.gc_augmentation(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])
    functions.gc_rewiring(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])

    # Prototypical Networks
    functions.gc_protonet(config=config, shots=shots, test_dataset=test_dataset, device=device, dataset=dataset, results_path=config["results_path"])

    # Siamese Networks
    functions.gc_siamesenet(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], CL_Loss='PW')
    functions.gc_siamesenet(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], CL_Loss='PW_A')
    functions.gc_siamesenet(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], CL_Loss='TPL')
    functions.gc_siamesenet(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"], CL_Loss='TPL_A')

    # Graph Contrastive Learning
    functions.gc_graphCL(config=config, shots=shots, test_loader=test_loader, device=device, dataset=dataset, results_path=config["results_path"])
    

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'='*20} {config['data']['dataset_name']} - {end_time} - {device} {'='*20}\n\n")
