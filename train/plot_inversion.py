import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE PATHS & PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
folder    = "." #"noise(0.01)_NS_smooth_partial(0.1)_lr=0.5_sample=10"#"prior_mean_noise(0.01)_tau(3)_partial(0.25)_correct" #"prior_mean_wonoise" #"naturalperturb_pert=0.8_30000_lr=0.5_ood"#'.'
dest_folder = folder
initial   = 'prior_mean'#'naturalperturb'
gd_type   = '_GD'               # e.g. "_NGD" if you used that suffix
max_iter  = 100            # match your data
every     = 10             # how often to draw markers / errorbars
expt_name = "input_outdist"
dim       = 128
data_type = "Darcy"

# CSVs for each method
csvs = {
    # r'FINO ($r = 50$)':  f'{folder}/loss_statistics_multiple_samples_JAC_50_{initial}{gd_type}.csv', #prior_mean_ noise(0.01)_tau(3)_partial(0.25)/loss_statistics_multiple_samples_JAC_50_prior_mean_GD.csv
    # r'FINO ($r = 200$)': f'{folder}/loss_statistics_multiple_samples_JAC_200_{initial}{gd_type}.csv',
    'MSE-FNO':          f'{folder}/loss_statistics_multiple_samples_MSE_{data_type}_{initial}{gd_type}.csv',
    'Numerical Simulator':       f'{folder}/loss_statistics_multiple_samples_Devito_{data_type}_{initial}{gd_type}.csv',
    # 'PINO': f'{folder}/loss_statistics_multiple_samples_PINO_{initial}{gd_type}.csv',
    r'FINO ($r = 400$)': f'{folder}/loss_statistics_multiple_samples_JAC_{data_type}_400_{initial}{gd_type}.csv',
}

# H5s for the two norm-error panels
h5s = {
    **csvs,  # reuse same keys and base filenames but with .h5 instead of .csv for the JAC methods
    # r'FINO ($r = 50$)':  f'{folder}/inversion_history_JAC_50_{initial}{gd_type}.h5',
    # r'FINO ($r = 200$)': f'{folder}/inversion_history_JAC_200_{initial}{gd_type}.h5',
    'MSE-FNO':          f'{folder}/inversion_history_MSE_{data_type}_{initial}{gd_type}.h5',
    'Numerical Simulator':       f'{folder}/inversion_history_Devito_{data_type}_{initial}{gd_type}.h5',
    # 'PINO':       f'{folder}/inversion_history_PINO_{initial}{gd_type}.h5',
    r'FINO ($r = 400$)': f'{folder}/inversion_history_JAC_{data_type}_400_{initial}{gd_type}.h5',
}

# Color-blind palette + styles
# cb = plt.get_cmap('Dark2').colors

# styles = {
#     # r'FINO ($r = 50$)':  dict(color=cb[0], marker='o', linestyle='-.'),
#     # r'FINO ($r = 200$)': dict(color=cb[1], marker='s', linestyle='-'),
#     r'FINO ($r = 400$)': dict(color=cb[2], marker='^', linestyle=':'),
#     'MSE-FNO':          dict(color=cb[3], marker='d', linestyle='--'),
#     'Numerical Simulator':       dict(color=cb[4], marker='v', linestyle=':'),
#     'PINO':       dict(color=cb[5], marker='s', linestyle='-.'),
# }

# unified palette + style (colors from HUSL, markers/linestyles as before)
# pal = sns.color_palette("husl", len(h5s))
pal = sns.color_palette("husl", 4)
method_list = ["MSE-FNO", "Numerical Simulator", "PINO", r"FINO ($r = 400$)"] #list(h5s.keys())

symbols = {
    "MSE-FNO":           dict(marker="d", linestyle="--"),
    "Numerical Simulator": dict(marker="v", linestyle=":"),
    "PINO":              dict(marker="s", linestyle="-."),
    r"FINO ($r = 400$)": dict(marker="^", linestyle=":"),
}

styles = {m: dict(color=pal[i], **symbols[m]) for i, m in enumerate(method_list)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD CSV DATA for each method
# ─────────────────────────────────────────────────────────────────────────────
# each CSV has 'iteration' plus the metrics columns
dfs = {m: pd.read_csv(path).query("iteration <= @max_iter")
       for m, path in csvs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. H5 LOADING & NORM-ERROR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def load_a(path):
    with h5py.File(path,'r') as f:
        A = f['a'][:]            # (samples, iters, H, W)
    s, t, h, w = A.shape
    return A.reshape(s, t, h*w)

def rel_stats(A_model, A_ref, p):
    s, t, d = A_model.shape # 1, 20000, 256*256
    rel = np.zeros((s, t))
    for i in range(s):
        for j in range(t):
            diff = A_model[i,j] - A_ref
            if p == np.inf:
                rel[i,j] = np.max(np.abs(diff)) / np.max(np.abs(A_ref))
            else:
                rel[i,j] = np.linalg.norm(diff,ord=p) / np.linalg.norm(A_ref,ord=p)
    return rel.mean(axis=0), rel.std(axis=0)

def compute_gradients(field, h, w):
    f2d = field.reshape(h, w)
    gx = np.zeros_like(f2d); gy = np.zeros_like(f2d)
    gx[:,1:-1] = (f2d[:,2:] - f2d[:,:-2]) / 2.0
    gy[1:-1,:] = (f2d[2:,:] - f2d[:-2,:]) / 2.0
    return gx, gy

def H1_norm(field, h, w):
    f2d = field.reshape(h, w)
    gx, gy = compute_gradients(field, h, w)
    return np.sqrt(np.sum(f2d**2) + np.sum(gx**2) + np.sum(gy**2))

def rel_H1_stats(A_model, A_ref, h, w):
    s, t, d = A_model.shape
    rel = np.zeros((s, t))
    for i in range(s):
        for j in range(t):
            diff = A_model[i,j] - A_ref
            rel[i,j] = H1_norm(diff, h, w) / H1_norm(A_ref, h, w)
    return rel.mean(axis=0), rel.std(axis=0)


# reference Devito trajectory
A_dev = load_a(h5s['Numerical Simulator'])

# compute rel2 and rel_inf stats
rel2, rel_inf = {}, {}
for name, path in h5s.items():
    A_mod = load_a(path)
    μ2, σ2 = rel_stats(A_mod, A_dev[0, -1], p=2)
    μ_inf, σ_inf = rel_stats(A_mod, A_dev[0, -1], p=np.inf)
    rel2[name]    = (μ2[:max_iter], σ2[:max_iter])
    rel_inf[name] = (μ_inf[:max_iter], σ_inf[:max_iter])

# compute rel2, rel_inf, rel_H1 stats
rel2, rel_inf, rel_H1 = {}, {}, {}
for name, path in h5s.items():
    A_mod = load_a(path)
    μ2, σ2 = rel_stats(A_mod, A_dev[0, -1], p=2)
    μ_inf, σ_inf = rel_stats(A_mod, A_dev[0, -1], p=np.inf)
    μ_H1, σ_H1 = rel_H1_stats(A_mod, A_dev[0, -1], dim, dim)
    rel2[name]    = (μ2[:max_iter], σ2[:max_iter])
    rel_inf[name] = (μ_inf[:max_iter], σ_inf[:max_iter])
    rel_H1[name]  = (μ_H1[:max_iter], σ_H1[:max_iter])

iters = np.arange(1, max_iter+1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT **ONE FIGURE PER PANEL**
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 10,
})

# (key_or_dict ,   title text,                    y-axis label,  stem for PNG)
panels = [
    ('loss',           'Data residual',            r'$\frac{1}{n}\sum^n_{i=1}(\mathbf{y}^\ast - \mathbf{y}_{nn})^2$',           f'mt_loss'),
    ('inversion_MSE',  r'Model Error in MSE', r'$\frac{1}{n}\sum^n_{i=1}(\mathbf{a}^\ast - \mathbf{a}_{nn})^2$',   f'mt_model_error'),
    ('SSIM',           'SSIM',                     'SSIM',                      f'mt_ssim'),
    ('loss',           'Data residual',            r'$\frac{1}{n}\sum^n_{i=1}(\mathbf{y}^\ast - \mathbf{y}_{nn})^2$',           f'mt_loss_woMSE'),
    ('rel_H1',         r'Model Error in Relative $H^1$',    r'$\frac{\|\mathbf{a}_{nn} - \mathbf{a}^\ast\|_{H^1}}{\|\mathbf{a}^\ast\|_{H^1}}$',           f'mt_relH'),
]

for key, title, ylabel, fname in panels:
    fig, ax = plt.subplots(figsize=(8, 5))

    # common cosmetic tweaks
    ax.minorticks_on()
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.grid(True, which='major', linestyle='-')

    # ─── draw the curves ─────────────────────────────────────────────────────
    if isinstance(key, str):  # CSV metrics
        for m, style in styles.items():
            if m not in dfs:  # skip if dataframe not found
                continue
            df = dfs[m]
            grouped = df.groupby("iteration")[key]
            μ = grouped.mean().values
            σ = grouped.std().values
            iters_csv = grouped.mean().index.values

            if key == "loss":
                ax.semilogy(iters_csv, μ, markevery=every, label=m, **style)    
            else:
                ax.plot(iters_csv, μ, markevery=every, label=m, **style)
            ax.fill_between(iters_csv, μ - σ, μ + σ, alpha=0.3, color=style["color"])

    else:  # dicts like rel2, rel_inf, rel_H1
        for m, style in styles.items():
            μ, σ = key[m]
            ax.plot(iters, μ, markevery=every, label=m, **style)
            ax.fill_between(iters, μ - σ, μ + σ, alpha=0.3, color=style["color"])


    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{dest_folder}/{fname}{expt_name}{gd_type}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)          # frees memory before the next panel
