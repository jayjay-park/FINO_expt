import os
import h5py
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)

def save_config(config: DictConfig):
    save_path = os.path.join(config['experiment']['output_dir'], 'config.yaml')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        OmegaConf.save(config=config, f=f)


def create_simulator(model_type, simulator_settings):
    if model_type == "NS":
        from simulators.NS import NavierStokesSimulator
        return NavierStokesSimulator(simulator_settings['s1'], 
                                     simulator_settings['s2'],
                                     simulator_settings['scale'],
                                     simulator_settings['T'], 
                                     simulator_settings['Re'],
                                     simulator_settings['adaptive'],
                                     simulator_settings['delta_t'],
                                     simulator_settings['nburn'],
                                     simulator_settings['nsteps'])
    elif model_type == "OldNS":
        from simulators.oldNS import OldNavierStokesSimulator
        return OldNavierStokesSimulator(simulator_settings['N'], 
                                     simulator_settings['L'], 
                                     simulator_settings['dt'], 
                                     simulator_settings['nu'],
                                     simulator_settings['nburn'],
                                     simulator_settings['nsteps'])
    elif model_type == "DARCY":
        from simulators.darcy import DarcySimulator
        return DarcySimulator(
            size=simulator_settings.size,
            T=simulator_settings.T
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_reduced_model(reduced_model_type, reduced_model_settings, simulator_type):
    if reduced_model_type == "FIM":
        from reduced_orders.fim import FIMReducedModel
        return FIMReducedModel(reduced_model_settings['eigen_value_fraction'],
                               reduced_model_settings['eigen_vector_count'], simulator_type)
    elif reduced_model_type == "RAND":
        from reduced_orders.random import RandomReducedModel
        return RandomReducedModel(reduced_model_settings['eigen_value_fraction'],
                                  reduced_model_settings['eigen_vector_count'], simulator_type)
    else:
        raise ValueError(f"Unsupported reduced model type: {reduced_model_type}")

def generate_dataset(simulator, reduced_model, data_settings, viz_settings, simulator_type):
    data_dir = data_settings['data_dir']
    plots_dir = viz_settings['plots_dir']

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    num_samples = data_settings['num_samples']
    plot_interval = viz_settings['plot_interval']
    plot_vector_count = viz_settings['plot_vector_count']
    
    # Get the number of workers from settings or default to CPU count - 1
    num_workers = data_settings.get('num_workers', max(1, os.cpu_count() - 1))
    
    # Define a worker function to process each sample
    def process_sample(i):
        print(f"Generating sample {i + 1} of {num_samples}")
        
        eigen_count = reduced_model.eigen_count(simulator)
        
        # TODO: Unify device handling code across the codebase
        x = simulator.sample()
        v, s = reduced_model.get_direction(simulator, x)
        Jvp = torch.zeros((simulator.range, eigen_count)).to(x.device)

        for e in range(eigen_count):
            print(f"Eigenvector {e + 1} of {eigen_count}")
            vector = v[:, e].reshape(x.shape).to(x.device)
            if simulator_type == "DARCY":
                f = np.zeros(x.shape)
                y = simulator.model.eval_fwd_op(f, x.cpu().numpy())
                y = torch.tensor(y)
                print("y", y.shape)

                plt.figure(figsize=(5, 5))
                plt.imshow(y, cmap='viridis')
                plt.title('Output')
                plt.savefig(f'Devito_output.png', bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(x.cpu().numpy(), cmap='viridis')
                plt.title('Input')
                plt.savefig(f'Devito_input.png', bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(vector.cpu().numpy(), cmap='viridis')
                plt.title('v')
                plt.savefig(f'Devito_v.png', bbox_inches='tight')
                plt.close()

                jvp_vector = simulator.model.compute_linearization(f, x.cpu().numpy(), vector.cpu().numpy())

                plt.figure(figsize=(5, 5))
                plt.imshow(jvp_vector, cmap='viridis')
                plt.title(r'Ground Truth $Jv$')
                plt.savefig(f'Devito_jvp.png', bbox_inches='tight')
                plt.close()
            else:
                y, jvp_vector = torch.func.jvp(simulator, (x,), (vector,))
            Jvp[:, e] = torch.tensor(jvp_vector).reshape(simulator.range)

        if i % plot_interval == 0:
            sample_dir = os.path.join(plots_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)

            for e in range(plot_vector_count):
                plot_path = os.path.join(sample_dir, f"vector_{e}.png")
                decay_path = os.path.join(sample_dir, f"decay_{e}.png")
                simulator.plot_data(x, y, v[:, e], Jvp[:, e], plot_path, f"Sample {i} Eigenvector {e}")
                reduced_model.plot_decay(s, decay_path, f"Sample {i} Eigenvector {e}")
        
        sample_path = os.path.join(data_dir, f"sample_{i}.h5")
        with h5py.File(sample_path, "w") as f:
            f.create_dataset("x", data=x.cpu().numpy())
            f.create_dataset("y", data=y.cpu().numpy())
            f.create_dataset("v", data=v.cpu().numpy())
            f.create_dataset("Jvp", data=Jvp.cpu().numpy())
        
        return i

    # Use ThreadPoolExecutor for I/O-bound operations
    import concurrent.futures
    from tqdm import tqdm
    
    print(f"Starting dataset generation with {num_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and create a dictionary of futures
        future_to_sample = {executor.submit(process_sample, i): i for i in range(num_samples)}
        
        # Process results as they complete
        completed = 0
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            for future in concurrent.futures.as_completed(future_to_sample):
                sample_idx = future_to_sample[future]
                result = future.result()
                completed += 1
                pbar.update(1)
        
        print(f"Completed {completed}/{num_samples} samples")
