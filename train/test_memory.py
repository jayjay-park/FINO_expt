import torch
import torch.nn.functional as F
from functorch import jvp
from physicsnemo.models.fno import FNO
import sys
sys.path.append("/net/slimdata/jayjaydata2/DeFINO_Richard/ConvolutionalNeuralOperator")
sys.path.append("/net/slimdata/jayjaydata2/DeFINO_Richard/ConvolutionalNeuralOperator/cno")

from cno.models import CNO



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Model & Data
# ---------------------------------------------------------------------
nx, ny = 512, 256
batch_size = 1

model = FNO(
    num_fno_modes=(16, 16),
    latent_channels=64,
    in_channels=1,
    out_channels=1,
    dimension=2,
).to(device)

u = torch.randn(batch_size, 1, nx, ny, device=device, requires_grad=True)
y_true = torch.randn(batch_size, 1, nx, ny, device=device)

# K probe directions for JvP misfit (you can increase for Monte-Carlo Fisher averaging)
K = 3
v_list = [torch.randn_like(y_true) for _ in range(K)]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# Helper: compute JvP misfit for several probes
# ---------------------------------------------------------------------
def jvp_misfit(model, u, v_list, y_true):
    """Compute mean JvP misfit ∑ₖ ||Jvₖ − J_true vₖ||² / K.
       If you only have synthetic data, you can use Jvₖ itself or residual-weighted variant."""
    misfit = 0.0
    # Forward (shared)
    def f(u_):
        return model(u_)

    # Base output and residual
    y_pred = f(u)
    residual = y_pred - y_true

    # For each probe direction v_k
    for v in v_list:
        # Compute Jv (Jacobian-vector product of model wrt input)
        y_pred_jvp, jvp_val = jvp(f, (u,), (v,))
        # JvP misfit as inner product between Jv and residual (like Fisher directional misfit)
        misfit += torch.mean((jvp_val * residual).pow(2))
    return misfit / len(v_list), y_pred

# ---------------------------------------------------------------------
# Training loop (test run)
# ---------------------------------------------------------------------
for it in range(3):
    optimizer.zero_grad()

    # Compute both MSE and JvP misfit
    jvp_loss, y_pred = jvp_misfit(model, u, v_list, y_true)
    mse_loss = F.mse_loss(y_pred, y_true)

    total_loss = mse_loss + 1e-2 * jvp_loss
    total_loss.backward()
    optimizer.step()

    print(f"[Iter {it}] total={total_loss.item():.4e}, mse={mse_loss.item():.4e}, jvp={jvp_loss.item():.4e}")
