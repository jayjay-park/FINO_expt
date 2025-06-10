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
                               reduced_model_settings['eigen_vector_count'], 
                               simulator_type,
                               reduced_model_settings['noise_std'],
                               reduced_model_settings['probe_size'])
    elif reduced_model_type == "RAND":
        from reduced_orders.random import RandomReducedModel
        return RandomReducedModel(reduced_model_settings['eigen_value_fraction'],
                                  reduced_model_settings['eigen_vector_count'], simulator_type)
    else:
        raise ValueError(f"Unsupported reduced model type: {reduced_model_type}")


def generate_dataset(simulator, reduced_model, data_settings, viz_settings, simulator_type):
    data_dir = data_settings['data_dir']
    plots_dir = viz_settings['plots_dir']
    num_samples = data_settings['num_samples']
    num_subsamples = data_settings['num_subsamples']
    plot_interval = viz_settings['plot_interval']
    plot_vector_count = viz_settings['plot_vector_count']
    num_workers = data_settings.get('num_workers', max(1, os.cpu_count() - 1))
    eigen_count = reduced_model.eigen_count(simulator)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ----------------------------------------------------
    # 0) COLUMN‑WISE jittered pick of m grid‑points once
    # ----------------------------------------------------

    # # 1) jittered pick of m columns once
    # d = int(np.sqrt(simulator.domain))
    # m = reduced_model.probe_size
    # bins = torch.linspace(0, d, m+1, device='cpu')
    # L = []
    # for k in range(m):
    #     s, e = int(bins[k].item()), int(bins[k+1].item())
    #     L.append(int(torch.randint(s, max(s+1, e), (), device='cpu').item()))
    # print("L", L)

    # print("Creating v ... ")
    # # 2) build and stack all Q_b ∈ R^{p×r}
    # Q_list = []
    # for b in range(num_subsamples):
    #     print("number of subsample", b)
    #     x_b = simulator.sample().detach().cpu().numpy()
    #     Qb = reduced_model.compute_score_matrix(simulator, x_b, L)  # shape [p, r]
    #     # print("Qb", Qb.shape) Qb torch.Size([16384, 25])
    #     Q_list.append(Qb)

    
    # ----------------------------------------------------
    # 1) BLOCK‑WISE jittered pick of m grid‑points once
    # ----------------------------------------------------
    d = int(np.sqrt(simulator.domain))       # assume domain = d*d
    m = reduced_model.probe_size             # total number of samples
    # factor m into a near‑square grid of blocks
    m_x = int(np.sqrt(m))
    m_y = m // m_x
    assert m_x * m_y == m, "probe_size must factor as m_x*m_y"

    bx = d // m_x   # block size in x
    by = d // m_y   # block size in y

    print(m_x, m_y, "bx", bx, by)

    L = []
    for px in range(m_x):
        for qy in range(m_y):
            # pick one random point (i,j) in block (px,qy)
            i = torch.randint(px*bx, min((px+1)*bx, d), (), device='cpu').item()
            j = torch.randint(qy*by, min((qy+1)*by, d), (), device='cpu').item()
            # flatten to a single index in [0, d*d)
            L.append(i * d + j)

    # ----------------------------------------------------
    # 2) build and stack all Q_b ∈ R^{p×r}
    # ----------------------------------------------------
    Q_list = []
    # Sigma = reduced_model.build_anisotropic_covariance(d, sigma_2=1.0, lx=0.2, ly=0.05, device='cpu')
    # coords = reduced_model.build_coords(d)
    # Sigma = matern_kernel(coords, nu=1.5, sigma2=1.0, lx=8.0, ly=4.0)


    for b in range(num_subsamples):
        print("subsample", b+1, "of", num_subsamples)
        x_b = simulator.sample().detach().cpu().numpy()
        Qb = reduced_model.compute_score_matrix(simulator, x_b, L)  # [p, r]
        Q_list.append(Qb)

    # 3) form Q_tilde ∈ R^{p × (r * num_subsamples)}
    Q_tilde = torch.cat(Q_list, dim=1) # 16384, 50
    print("Q_tilde", Q_tilde.shape)

    # 4) single SVD on Q_tilde
    U, S, Vh = torch.linalg.svd(Q_tilde, full_matrices=False)
    v, s = U[:, :eigen_count], S[:eigen_count]  # top‑r modes
    reduced_model.plot_decay(s.detach().cpu(), f"Averaged_Decay_over_{num_subsamples}_samples.png", f"Total Decay over {eigen_count}")
    
    # Define a worker function to process each sample
    def process_sample(i):
        print(f"Generating sample {i + 1} of {num_samples}")
            
        # TODO: Unify device handling code across the codebase
        x = simulator.sample()
        Jvp = torch.zeros((simulator.range, eigen_count)).to(x.device)

        for e in range(eigen_count):
            print(f"Eigenvector {e + 1} of {eigen_count}")
            vector = v[:, e].reshape(x.shape).to(x.device)
            if simulator_type == "DARCY":
                f = np.zeros(x.shape)
                y = simulator.model.eval_fwd_op(f, x.cpu().numpy(), simulator.T)
                y = torch.tensor(y)
                jvp_vector = simulator.model.compute_linearization(f, x.cpu().numpy(), vector.cpu().numpy(), simulator.T)
            else:
                y, jvp_vector = torch.func.jvp(simulator, (x,), (vector,))
            Jvp[:, e] = torch.tensor(jvp_vector).reshape(simulator.range)

        if i % plot_interval == 0:
            plot_sampling_on_field_simple_blocks(y.cpu().numpy(), L, m_x, m_y)
            sample_dir = os.path.join(plots_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)

            for e in range(plot_vector_count):
                print(e, plot_vector_count)
                plot_path = os.path.join(sample_dir, f"vector_{e}.png")
                decay_path = os.path.join(sample_dir, f"decay_{e}.png")
                simulator.plot_data(x, y, v[:, e], Jvp[:, e], plot_path, f"Sample {i} Eigenvector {e}")
                reduced_model.plot_decay(s.detach().cpu(), decay_path, f"Sample {i} Eigenvector {e}")
        
        sample_path = os.path.join(data_dir, f"sample_{i}.h5")
        with h5py.File(sample_path, "w") as f:
            f.create_dataset("x", data=x.cpu().numpy())
            f.create_dataset("y", data=y.cpu().numpy())
            f.create_dataset("v", data=v.cpu().numpy())
            f.create_dataset("Jvp", data=Jvp.cpu().numpy())
            f.create_dataset("L", data=np.array(L, dtype=np.int32))
        
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

def plot_sampling_on_field_simple_blocks(y, L, m_x, m_y, figsize=(6,6)):
    """
    Plot the 2D field 'y' with point-wise jittered samples and simple block boundaries.

    Parameters
    ----------
    y : np.ndarray, shape (d, d)
    L : list of int, length m
        Flattened indices of sampled points in the d*d grid.
    m_x, m_y : int
        Number of blocks in the x (rows) and y (columns) directions.
    """
    d = y.shape[0]
    b_x = d // m_x
    b_y = d // m_y

    rows = np.array([idx // d for idx in L])
    cols = np.array([idx % d for idx in L])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(y, cmap='jet', origin='upper')
    fig.colorbar(im, ax=ax, fraction=0.046)

    # horizontal lines at each block row
    for px in range(1, m_x):
        y_line = px * b_x - 0.5
        ax.hlines(y=y_line, xmin=-0.5, xmax=d-0.5, colors='white', linestyles='--')

    # vertical lines at each block column
    for qy in range(1, m_y):
        x_line = qy * b_y - 0.5
        ax.vlines(x=x_line, ymin=-0.5, ymax=d-0.5, colors='white', linestyles='--')

    ax.scatter(cols, rows,
               s=40, marker='o',
               facecolors='none', edgecolors='red',
               linewidths=1., label="Samples")

    ax.set_title("Block‑wise Jittered Samples")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Observation_O")