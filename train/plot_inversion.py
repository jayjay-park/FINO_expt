import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE PATHS & PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
folder    = "prior_mean_noise(0.01)_ood(+5)" #"prior_mean_wonoise" #"naturalperturb_pert=0.8_30000_lr=0.5_ood"#'.'
initial   = 'prior_mean'#'naturalperturb'
gd_type   = '_GD'               # e.g. "_NGD" if you used that suffix
max_iter  = 1000            # match your data
every     = 100             # how often to draw markers / errorbars
expt_name = "input_outdist"
dest_folder = "prior_mean_noise(0.01)_ood(+5)"

# CSVs for each method
csvs = {
    r'FINO ($r = 50$)':  f'{folder}/loss_statistics_multiple_samples_JAC_50_{initial}{gd_type}.csv',
    r'FINO ($r = 200$)': f'{folder}/loss_statistics_multiple_samples_JAC_200_{initial}{gd_type}.csv',
    r'FINO ($r = 400$)': f'{folder}/loss_statistics_multiple_samples_JAC_400_{initial}{gd_type}.csv',
    'MSE-FNO':          f'{folder}/loss_statistics_multiple_samples_MSE_{initial}{gd_type}.csv',
    'Numerical Simulator':       f'{folder}/loss_statistics_multiple_samples_Devito_{initial}{gd_type}.csv',
}

# H5s for the two norm-error panels
h5s = {
    **csvs,  # reuse same keys and base filenames but with .h5 instead of .csv for the JAC methods
    r'FINO ($r = 50$)':  f'{folder}/inversion_history_JAC_50_{initial}{gd_type}.h5',
    r'FINO ($r = 200$)': f'{folder}/inversion_history_JAC_200_{initial}{gd_type}.h5',
    r'FINO ($r = 400$)': f'{folder}/inversion_history_JAC_400_{initial}{gd_type}.h5',
    'MSE-FNO':          f'{folder}/inversion_history_MSE_{initial}{gd_type}.h5',
    'Numerical Simulator':       f'{folder}/inversion_history_Devito_{initial}{gd_type}.h5',
}

# Color-blind palette + styles
cb = plt.get_cmap('Dark2').colors
styles = {
    r'FINO ($r = 50$)':  dict(color=cb[0], marker='o', linestyle='-.'),
    r'FINO ($r = 200$)': dict(color=cb[1], marker='s', linestyle='-'),
    r'FINO ($r = 400$)': dict(color=cb[2], marker='^', linestyle=':'),
    'MSE-FNO':          dict(color=cb[3], marker='d', linestyle='--'),
    'Numerical Simulator':       dict(color=cb[4], marker='v', linestyle=':'),
}

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

iters = np.arange(1, max_iter+1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT **ONE FIGURE PER PANEL**
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# (key_or_dict ,   title text,                    y-axis label,  stem for PNG)
panels = [
    ('loss',           'Data residual',            r'$\frac{1}{n}\sum^n_{i=1}(\mathbf{y}_\text{true} - \mathbf{y}_\text{pred})^2$',           f'mt_loss'),
    ('inversion_MSE',  r'Rel. Frobenius error to $a^{\ast}$', r'$\frac{\|\mathbf{a}^{\ast} - \mathbf{a}\|_F}{\|\mathbf{a}^{\ast}\|_F}$',   f'mt_fro_error'),
    ('SSIM',           'SSIM',                     'SSIM',                      f'mt_ssim'),
    (rel2,             r'Relative $L_2$-error',    r'$\frac{\|\mathbf{a}_{\text{model}} - \mathbf{a}_{\text{NS}}\|_2}{\|\mathbf{a}_{\text{NS}}\|_2}$',        f'mt_rel2'),
    (rel_inf,          r'Relative $L_\infty$-error', r'$\frac{\|\mathbf{a}_{\text{model}} - \mathbf{a}_{\text{NS}}\|_\infty}{\|\mathbf{a}_{\text{NS}}\|_\infty}$', f'mt_rel_inf'),
    ('loss',           'Data residual',            r'$\frac{1}{n}\sum^n_{i=1}(\mathbf{y}_\text{true} - \mathbf{y}_\text{pred})^2$',           f'mt_loss_woMSE'),
    ('rel_H1',         r'Relative $H^1$-error',    r'$\frac{\|\mathbf{a}_{\text{model}} - \mathbf{a}_{\text{NS}}\|_{H^1}}{\|\mathbf{a}_{\text{NS}}\|_{H^1}}$',           f'mt_relH'),
]

for key, title, ylabel, fname in panels:
    fig, ax = plt.subplots(figsize=(10, 6))

    # common cosmetic tweaks
    ax.minorticks_on()
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.grid(True, which='major', linestyle='-')

    # ─── draw the curves ─────────────────────────────────────────────────────
    if isinstance(key, str):                        # CSV metrics
        for m, style in styles.items():
            if key == 'loss':
                if fname == 'mt_loss_woMSE':
                    if m == "MSE-FNO":
                        continue
                    else:
                        ax.semilogy(dfs[m]['iteration'], dfs[m][key],
                            markevery=every, label=m, **style)
                else:
                    ax.semilogy(dfs[m]['iteration'], dfs[m][key],
                    markevery=every, label=m, **style)
            else:
                ax.plot(dfs[m]['iteration'], dfs[m][key],
                    markevery=every, label=m, **style)
    else:                                           # rel2 / rel_inf dicts
        for m, style in styles.items():
            μ, σ = key[m]
            # ax.errorbar(iters, μ, yerr=σ,
            #             capsize=3, alpha=0.8,
            #             markevery=every, label=m, **style)
            ax.plot(iters, μ, markevery=every, label=m, **style)

    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{dest_folder}/{fname}{expt_name}{gd_type}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)          # frees memory before the next panel
