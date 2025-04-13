import os
import yaml
import logging
import datetime
import csv
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from omegaconf import OmegaConf, DictConfig

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

def get_model(model_type: str, model_settings: DictConfig) -> str:
    if model_type == "GCS":
        from models.gcs import GCSModel
        model = GCSModel() # Non Functional Model. Reimplement call if needed. (Removed due to vJp -> Jvp)
        return model
    elif model_type == "NS":
        from models.ns import NSModel
        model = NSModel(
            in_channels=model_settings.in_channels,
            out_channels=model_settings.out_channels,
            decoder_layer_size=model_settings.decoder_layer_size,
            num_fno_layers=model_settings.num_fno_layers,
            num_fno_modes=model_settings.num_fno_modes,
            padding=model_settings.padding,
            dimension=model_settings.dimension,
            latent_channels=model_settings.latent_channels,
            loss_type=model_settings.loss_type,
            train_eigen_count=model_settings.train_eigen_count,
            reg_param=model_settings.reg_param,
            scale_factor=model_settings.scale_factor,
            learning_rate=model_settings.learning_rate,
            weight_decay=model_settings.weight_decay,
            ckpt_path=model_settings.ckpt_path,
        )
        return model
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_dataset(dataset_type: str, data_settings: DictConfig) -> str:
    if dataset_type == "GCS":
        from datasets.gcs import GCSDataLoader
        data_loader = GCSDataLoader() # Non Functional Data Loader. Reimplement call if needed. (Removed due to vJp -> Jvp)
        return data_loader
    elif dataset_type == "NS":
        from datasets.ns import NSDataLoader
        data_loader = NSDataLoader(
            nx=data_settings.nx,
            ny=data_settings.ny,
            eigen_count=data_settings.eigen_count,
            sample_directories=data_settings.sample_directories,
            batch_size=data_settings.batch_size,
        )
        return data_loader
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

def setup_logging(log_dir: str = "logs", experiment_name: Optional[str] = None) -> Tuple[str, logging.Logger]:
    """
    Set up logging for the experiment.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        
    Returns:
        Tuple of log file path and logger
    """
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        experiment_name = f"GCS_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return log_file, logger


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration as DictConfig
    """
    config = OmegaConf.load(config_path)
    if 'base_config' in config:
        base_config_path = config.pop('base_config')
        if not os.path.isabs(base_config_path):
            config_dir = os.path.dirname(config_path)
            base_config_path = os.path.join(config_dir, base_config_path)
        
        base_config = load_config(base_config_path)  # Recursive loading 
        merged_config = OmegaConf.merge(base_config, config)
        return merged_config
    
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration
        save_path: Path to save file
    """
    with open(save_path, 'w') as f:
        OmegaConf.save(config=config, f=f)


def setup_neptune_logging(config: DictConfig) -> Optional[Any]:
    """
    Set up Neptune logging if enabled.
    
    Args:
        config: Configuration
        
    Returns:
        Neptune run object or None
    """
    if not NEPTUNE_AVAILABLE or not config.neptune.enabled:
        return None
    
    # Set up Neptune run
    neptune_run = neptune.init_run(
        project=config.neptune.project,
        name=config.experiment.name,
        tags=list(config.experiment.tags),
        capture_stdout=True,
        capture_stderr=True
    )
    
    # Log configuration
    neptune_run["parameters"] = OmegaConf.to_container(config, resolve=True)
    
    return neptune_run


def create_directories(config: DictConfig) -> Dict[str, str]:
    """
    Create necessary directories for the experiment.
    
    Args:
        config: Configuration
        
    Returns:
        Dictionary of directory paths
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.experiment.name}_{timestamp}"
    
    directories = {
        "log_dir": os.path.join(config.experiment.log_dir, experiment_name),
        "output_dir": os.path.join(config.experiment.output_dir, experiment_name),
        "checkpoint_dir": os.path.join(config.experiment.checkpoint_dir, experiment_name),
        "plot_dir": os.path.join(config.experiment.output_dir, experiment_name, "plots"),
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    directories["experiment"] = experiment_name
    return directories


def log_config_to_file(config: DictConfig, logger: logging.Logger) -> None:
    """
    Log configuration to file.
    
    Args:
        config: Configuration
        logger: Logger
    """
    logger.info("Configuration:")
    for section, section_config in OmegaConf.to_container(config).items():
        logger.info(f"[{section}]")
        for key, value in section_config.items():
            logger.info(f"  {key}: {value}")


def save_loss_history(
    loss_data: Dict[str, List[float]],
    filepath: str,
    epochs: Optional[List[int]] = None
) -> None:
    """
    Save loss history to CSV file.
    
    Args:
        loss_data: Dictionary of loss values
        filepath: Path to save file
        epochs: List of epoch indices (default: range(1, len(first_loss_array)+1))
    """
    if not loss_data:
        return
    
    first_loss = next(iter(loss_data.values()))
    if epochs is None:
        epochs = list(range(1, len(first_loss) + 1))
    
    with open(filepath, 'w', newline='') as file:
        fieldnames = ['Epoch'] + list(loss_data.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, epoch in enumerate(epochs):
            row = {'Epoch': epoch}
            for loss_name, loss_values in loss_data.items():
                if i < len(loss_values):
                    row[loss_name] = loss_values[i]
            writer.writerow(row)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
