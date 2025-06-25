import argparse
import yaml
import os
import torch

from utils import create_simulator, create_reduced_model, load_config, generate_dataset, save_config

def main():
    parser = argparse.ArgumentParser(description='Dataset Generation')
    parser.add_argument('--config', type=str, default='config/navier_stokes.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    save_config(config)
    
    experiment = config['experiment']
    viz_settings = config['viz_settings']
    data_settings = config['data_settings']
    simulator_settings = config['simulator_settings']
    reduced_model_settings = config['reduced_model_settings']
    
    seed = experiment['seed']
    simulator_type = experiment['simulator_type']
    reduced_model_type = experiment['reduced_model_type']

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    simulator = create_simulator(simulator_type, simulator_settings)
    reduced_model = create_reduced_model(reduced_model_type, reduced_model_settings, simulator_type)
    
    generate_dataset(simulator, reduced_model, data_settings, viz_settings, simulator_type, reduced_model_type)
    
if __name__ == "__main__":
    main()
