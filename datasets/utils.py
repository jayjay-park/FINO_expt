
import os
import h5py
import torch
import numpy as np
from torch.func import vmap, jvp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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
                                     simulator_settings['Re'],
                                     simulator_settings['T'], 
                                     simulator_settings['delta_t'])
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
    elif reduced_model_type == "AS":
        from reduced_orders.AS import ASReducedModel
        return ASReducedModel(reduced_model_settings['eigen_value_fraction'],
                               reduced_model_settings['eigen_vector_count'], 
                               simulator_type,
                               reduced_model_settings['noise_std'],
                               reduced_model_settings['probe_size'])
    else:
        raise ValueError(f"Unsupported reduced model type: {reduced_model_type}")


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


def generate_dataset(simulator, reduced_model, data_settings, viz_settings, simulator_type, reduced_model_type):
    data_dir       = data_settings['data_dir']
    plots_dir      = viz_settings['plots_dir']
    num_samples    = data_settings['num_samples']
    num_subsamples = data_settings['num_subsamples']
    plot_interval  = viz_settings['plot_interval']
    plot_vector_count = viz_settings['plot_vector_count']
    num_workers    = data_settings.get('num_workers', max(1, os.cpu_count() - 1))
    eigen_count    = reduced_model.eigen_count(simulator)

    os.makedirs(data_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # path for caching modes
    if reduced_model_type == "AS":
        svd_cache = 'datasets/AS_svd_modes.h5'
    elif reduced_model_type == "FIM": 
        if simulator_type == "DARCY":
            svd_cache = 'datasets/svd_modes.h5'
        elif simulator_type == "NS":
            svd_cache = 'datasets/svd_modes_NS.h5'

    # --- build L once (block-wise jitter) ---
    d = int(np.sqrt(simulator.domain))
    m = reduced_model.probe_size
    m_x = int(np.sqrt(m)); m_y = m // m_x
    bx, by = d // m_x, d // m_y

    L = []
    for px in range(m_x):
        for qy in range(m_y):
            i = torch.randint(px*bx, min((px+1)*bx, d), (), device='cpu').item()
            j = torch.randint(qy*by, min((qy+1)*by, d), (), device='cpu').item()
            L.append(i * d + j)

    # ────────────────────────────────────────────────────────────────────────────
    # 0) LOAD OR COMPUTE SVD MODES
    # ────────────────────────────────────────────────────────────────────────────
    if os.path.isfile(svd_cache):
        # we have cached modes: load and move to device
        with h5py.File(svd_cache, 'r') as f:
            v_np = f['v'][:]
            s_np = f['s'][:]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v = torch.from_numpy(v_np).to(device)
        s = torch.from_numpy(s_np).to(device)
        print(f"Loaded cached SVD modes (shape v={v.shape}, s={s.shape})", flush=True)
    else:
        Q_list = []
        for b in range(num_subsamples):
            print(f"Computing Q_b for subsample {b+1}/{num_subsamples}", flush=True)
            x_b = simulator.sample().detach().cpu().numpy()
            # simulator.plot_vorticity(x_b, -2)
            if reduced_model_type == "FIM":
                Qb  = reduced_model.compute_score_matrix(simulator, x_b, L)  # [p, r]
                Q_list.append(Qb)
            elif reduced_model_type == "AS":
                theta = simulator.sample().cpu().numpy()
                Gb = reduced_model.compute_active_subspace(simulator, theta, L) # [p, 1]
                Q_list.append(Gb)  # [p, B]
                # Qb  = reduced_model.compute_active_subspace(simulator, x_b, idx_list)  # [p, r]

        Q_tilde = torch.cat(Q_list, dim=1)  # [p, r * num_subsamples]
        with h5py.File(f"Q_tilde_{simulator_type}.h5", "w") as f:
            f.create_dataset("Q_tilde", data=Q_tilde.cpu().numpy())
        print("Q_tilde shape:", Q_tilde.shape)

        # --- SVD ---
        U, S_full, Vh = torch.linalg.svd(Q_tilde, full_matrices=False)
        v = U[:, :eigen_count]
        s = S_full[:eigen_count]

        # --- cache them ---
        with h5py.File(svd_cache, 'w') as f:
            f.create_dataset('v', data=v.cpu().numpy(), compression='gzip')
            f.create_dataset('s', data=s.cpu().numpy(), compression='gzip')
        print(f"Computed & cached SVD modes to {svd_cache}")

        # optional: plot decay on the full s
        reduced_model.plot_decay(
            s.detach().cpu(),
            os.path.join(plots_dir, f"Averaged_Decay_over_{num_subsamples}_samples.png"),
            f"Decay (r={eigen_count})"
        )

    # ────────────────────────────────────────────────────────────────────────────
    # 1) Now generate each sample (and save x,y,v,Jvp,L per sample)
    # ────────────────────────────────────────────────────────────────────────────
    def process_sample(i):
        print(f"Generating sample {i + 1}/{num_samples}", flush=True)
        x = simulator.sample()
        print("x", x.shape)
        device = x.device
        p = simulator.range
        r = eigen_count

        # preallocate Jvp
        Jvp = torch.zeros((simulator.range, eigen_count), device=device)

        # compute JvPs along each eigenvector
        # for e in range(eigen_count):
        #     vec = v[:, e].reshape(x.shape).to(device)
        #     if simulator_type == "DARCY":
        #         f = np.zeros(x.shape)
        #         y = simulator.model.eval_fwd_op(f, x.cpu().numpy(), simulator.T)
        #         jvp = simulator.model.compute_linearization(
        #             f, x.cpu().numpy(), vec.cpu().numpy(), simulator.T
        #         )
        #         y   = torch.tensor(y,  device=device)
        #         jvp = torch.tensor(jvp,device=device)
        #     else:
        #         y, jvp = torch.func.jvp(simulator, (x,), (vec,))
        #     Jvp[:, e] = jvp.reshape(-1)

        # 1) compute y once
        if simulator_type == "DARCY":
            f    = np.ones(x.shape) # computing y...
            y_np = simulator.model.eval_fwd_op(f, x.cpu().numpy())
            y    = torch.as_tensor(y_np, device=device)
        else:
            y = simulator(x)  # just forward

        # 2) build all r tangent‐vectors at once: [r, *x.shape]
        vecs = v[:, :r].T.reshape(r, *x.shape).to(device)

        # 3a) Non-Darcy: pure‐PyTorch vmap over jvp
        if simulator_type != "DARCY":
            def _single_jvp(vt):
                # returns flattened JvP for one vt
                return jvp(simulator, (x,), (vt,))[1].reshape(-1)

            # batched call!
            Jvp_rows = vmap(_single_jvp)(vecs)               # [r, p]
            Jvp = Jvp_rows.transpose(0,1).contiguous()       # [p, r]
            print("finished Jvp", Jvp[0])

        # 3b) Darcy: thread-parallel compute_linearization
        else:
            # bring your vecs to CPU‐numpy: [r, d, d]
            probes = vecs.cpu().numpy()                     

            # a simple helper
            def _darcy_jvp(probe_2d):
                out = simulator.model.compute_linearization(
                    f, x.cpu().numpy(), probe_2d, simulator.T
                )
                return out.reshape(-1)

            grads = np.empty((r, p), dtype=np.float32)
            with ThreadPoolExecutor(max_workers=num_workers) as exe:
                futures = { exe.submit(_darcy_jvp, probes[i]): i
                            for i in range(r) }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    grads[idx] = fut.result()

            Jvp = torch.from_numpy(grads).to(device).transpose(0,1)  # [p, r]

        # optional plotting
        if i % plot_interval == 0:
            print("creating plot")
            sample_dir = os.path.join(plots_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            for e in range(plot_vector_count):
                simulator.plot_data(
                    x, y, v[:, e], Jvp[:, e],
                    os.path.join(sample_dir, f"vector_{e}.png"),
                    title=f"Sample {i}, Eigen {e}"
                )
                reduced_model.plot_decay(
                    s.detach().cpu(),
                    os.path.join(sample_dir, f"decay_{e}.png"),
                    f"Sample {i}, Eigen {e}"
                )

        # write out HDF5 for this sample
        plot_sampling_on_field_simple_blocks(y.cpu().numpy(), L, m_x, m_y)
        sample_path = os.path.join(data_dir, f"sample_{i}.h5")
        print("created sample path")
        with h5py.File(sample_path, "w") as f:
            print("Creating h5 file", {i})
            f.create_dataset("x",   data=x.cpu().numpy())
            f.create_dataset("y",   data=y.cpu().numpy())
            f.create_dataset("v",   data=v.cpu().numpy())
            f.create_dataset("s",   data=s.cpu().numpy())
            f.create_dataset("Jvp", data=Jvp.cpu().numpy())
            f.create_dataset("L",   data=np.array(L, dtype=np.int32))

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
                                                                                                                                                                                                                                                                                                                                                      