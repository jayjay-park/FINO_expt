import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
import matplotlib.colors as colors
from skimage.metrics import structural_similarity as ssim

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

class BaseVisualizationCallback(Callback):
    """Base class for visualization callbacks."""
    
    def __init__(self, output_dir=None, save_to_disk=False, log_to_neptune=False, num_plots=1, plot_interval=1):
        """
        Initialize visualization callback.
        
        Args:
            output_dir: Directory to save visualizations locally
            save_to_disk: Whether to save to disk
            log_to_neptune: Whether to log to Neptune
            num_plots: Number of plots to generate
            plot_interval: Plot every N epochs
        """
        super().__init__()
        self.output_dir = output_dir
        self.save_to_disk = save_to_disk
        self.num_plots = num_plots
        self.plot_interval = plot_interval
        if output_dir and not os.path.exists(output_dir) and save_to_disk:
            os.makedirs(output_dir, exist_ok=True)
        
        self.log_to_neptune = log_to_neptune and NEPTUNE_AVAILABLE
    
    def save_figure(self, fig, filename):
        """Save figure locally and/or to Neptune."""
        if self.save_to_disk and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
        return fig
    
    def log_figure(self, trainer, pl_module, fig, tag, step=None):
        """Log figure to Neptune if available."""
        if self.log_to_neptune and hasattr(trainer.logger, 'experiment'):
            if step is None:
                step = trainer.global_step
            
            # Log to Neptune
            trainer.logger.experiment[f"visualizations/{tag}"].append(fig)


class SaturationVisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing saturation predictions."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        if (trainer.current_epoch - 1) % self.plot_interval == 0:
            self.plot_saturation_results(trainer, pl_module, trainer.train_dataloader, "train")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():
            self.plot_saturation_results(trainer, pl_module, trainer.val_dataloaders, "val")
    
    def plot_saturation_results(self, trainer, pl_module, dataloader, tag):
        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            # Get model predictions
            x = batch['x'].to(pl_module.device)
            y_true = batch['y']
            
            with torch.no_grad():
                y_pred = pl_module(x)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            channel_idx = 0
            
            # # Plot for each timestep
            # for timestep in range(y_true.shape[1]):
            #     fig = self.plot_results(
            #         y_true[batch_idx, channel_idx, timestep].squeeze().cpu().numpy(),
            #         y_pred[batch_idx, channel_idx, timestep].squeeze().detach().cpu().numpy(),
            #         f"Timestep {timestep+1}"
            #     )
            #     filename = f"saturation_timestep_{timestep+1}_epoch_{trainer.current_epoch}.png"
            #     self.save_figure(fig, filename)
            #     self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/saturation/timestep_{timestep+1}", trainer.current_epoch)
            #     plt.close(fig)

            # Plot all timesteps in a single figure
            fig = self.plot_multiple_timesteps(
                y_true[batch_idx, channel_idx].squeeze().cpu().numpy(),
                y_pred[batch_idx, channel_idx].squeeze().detach().cpu().numpy()
            )
            
            filename = f"saturation_{tag}_timesteps_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/saturation_all", trainer.current_epoch)
            plt.close(fig)
    
    def plot_results(self, true1, pred1, title_suffix=""):
        """
        Plot true saturation, predicted saturation, and error.
        
        Args:
            true1: True saturation field
            pred1: Predicted saturation field
            title_suffix: Suffix for plot title
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(19, 5))
        plt.rcParams.update({'font.size': 14})
        
        # True saturation
        plt.subplot(1, 3, 1)
        plt.imshow(true1.numpy(), cmap='jet', vmin=0.0, vmax=1.0)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"True Saturation {title_suffix}")
        
        # Predicted saturation
        plt.subplot(1, 3, 2)
        plt.imshow(pred1.numpy(), cmap='jet', vmin=0.0, vmax=1.0)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Predicted Saturation {title_suffix}")
        
        # Error
        plt.subplot(1, 3, 3)
        error = np.abs(true1.numpy() - pred1.numpy())
        vmax_error = error.max()
        plt.imshow(error, cmap='magma', vmin=0.0, vmax=vmax_error)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('Error')
        
        return fig
    
    def plot_multiple_timesteps(self, true, pred, cmap='jet'):
        """
        Plot multiple timesteps in a single figure.
        
        Args:
            true: True saturation fields (timesteps, height, width)
            pred: Predicted saturation fields (timesteps, height, width)
            cmap: Colormap to use
            
        Returns:
            matplotlib figure
        """
        num_timesteps = true.shape[0]
        fig, axes = plt.subplots(3, num_timesteps, figsize=(4*num_timesteps, 8))
        plt.rcParams.update({'font.size': 12})
        
        # Plot true saturation (top row)
        for t in range(num_timesteps):
            im = axes[0, t].imshow(true[t], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[0, t].set_title(f'True t={t+1}')
            if t == 0:
                axes[0, t].set_ylabel('True')
            fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
        
        # Plot predicted saturation (bottom row)
        for t in range(num_timesteps):
            im = axes[1, t].imshow(pred[t], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[1, t].set_title(f'Pred t={t+1}')
            if t == 0:
                axes[1, t].set_ylabel('Predicted')
            fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.04)
            
        # Plot error (third row)
        for t in range(num_timesteps):
            im = axes[2, t].imshow(np.abs(true[t] - pred[t]), cmap='magma', vmin=0.0, vmax=1.0)
            axes[2, t].set_title(f'Error t={t+1}')
            if t == 0:
                axes[2, t].set_ylabel('Error')
            fig.colorbar(im, ax=axes[2, t], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig


class JacobianVisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing vector-Jacobian products."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        if (trainer.current_epoch - 1) % self.plot_interval == 0:
            original_shuffle = trainer.train_dataloader.dataset.shuffle
            trainer.train_dataloader.dataset.shuffle = False
            self.plot_jacobian_results(trainer, pl_module, trainer.train_dataloader, "train")
            trainer.train_dataloader.dataset.shuffle = original_shuffle
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():   
            self.plot_jacobian_results(trainer, pl_module, trainer.val_dataloaders, "val")
    
    def plot_jacobian_results(self, trainer, pl_module, dataloader, tag):
        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            # Only perform Jacobian visualization if model has VJP computation
            if not hasattr(pl_module, 'compute_vjps'):
                return
            
            x = batch['x'].to(pl_module.device)
            eigvec = batch['eigvec'].to(pl_module.device)
            true_vjp = batch['vjp'].to(pl_module.device)
            
            # Skip if no eigenvectors in batch
            if 'eigvec' not in batch or 'vjp' not in batch:
                trainer.logger.experiment[f"errors/{tag}"].append(f"No eigenvectors or VJPs in batch")
                return

            # Compute predicted VJPs
            pred_vjp = pl_module.compute_vjps(x, eigvec, clear_memory=True)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            vec_idx = 0    # Visualize first vector
            
            # # Plot for each timestep
            # for timestep in range(eigvec.shape[1]):
            #     # Plot VJP results for current timestep
            #     fig = self.plot_vjp_results(
            #         eigvec[batch_idx, timestep, vec_idx].detach().cpu(),
            #         true_vjp[batch_idx, timestep, vec_idx].detach().cpu(),
            #         pred_vjp[batch_idx, timestep, vec_idx].detach().cpu(),
            #         f"Timestep {timestep+1}"
            #     )
            #     filename = f"vjp_timestep_{timestep+1}_epoch_{trainer.current_epoch}.png"
            #     self.save_figure(fig, filename)
            #     self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/jacobian/timestep_{timestep+1}", trainer.current_epoch)
            #     plt.close(fig)
                
            # Plot all timesteps in a single figure
            fig = self.plot_multiple_timesteps(
                eigvec[batch_idx, :, vec_idx].squeeze().cpu().numpy(),
                true_vjp[batch_idx, :, vec_idx].squeeze().cpu().numpy(),
                pred_vjp[batch_idx, :, vec_idx].squeeze().detach().cpu().numpy()
            )
            
            filename = f"jacobian_{tag}_timesteps_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/jacobian_all", trainer.current_epoch)
            plt.close(fig)
    
    def plot_multiple_timesteps(self, eigvec, true_vjp, pred_vjp, cmap='seismic'):
        """
        Plot multiple timesteps in a single figure.
        
        Args:
            eigvec: Eigenvectors (timesteps, height, width)
            true_vjp: True vector-Jacobian products (timesteps, height, width)
            pred_vjp: Predicted vector-Jacobian products (timesteps, height, width)
            cmap: Colormap to use
            
        Returns:
            matplotlib figure
        """
        num_timesteps = eigvec.shape[0]
        fig, axes = plt.subplots(4, num_timesteps, figsize=(4*num_timesteps, 12))
        plt.rcParams.update({'font.size': 10})
        
        # Plot eigenvector
        for t in range(num_timesteps):
            eigvec_np = eigvec[t]
            eigvec_range = max(abs(eigvec_np.min()), abs(eigvec_np.max()))
            norm = SymLogNorm(linthresh=0.1 * eigvec_range, vmin=-eigvec_range, vmax=eigvec_range)
            im = axes[0, t].imshow(eigvec_np, cmap=cmap, norm=norm)
            axes[0, t].set_title(f'Eigenvector t={t+1}')
            if t == 0:
                axes[0, t].set_ylabel('Eigenvector')
            fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
        
        # Plot true vJp
        for t in range(num_timesteps):
            true_vjp_np = true_vjp[t]
            true_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max()))
            norm_true = SymLogNorm(linthresh=0.1 * true_range, vmin=-true_range, vmax=true_range)
            im = axes[1, t].imshow(true_vjp_np, cmap=cmap, norm=norm_true)
            axes[1, t].set_title(f'True VJP t={t+1}')
            if t == 0:
                axes[1, t].set_ylabel('True VJP')
            fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.04)
            
        # Plot predicted vJp
        for t in range(num_timesteps):
            pred_vjp_np = pred_vjp[t]
            pred_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max())) # scale according to true vjp
            norm_pred = SymLogNorm(linthresh=0.1 * pred_range, vmin=-pred_range, vmax=pred_range)
            im = axes[2, t].imshow(pred_vjp_np, cmap=cmap, norm=norm_pred)
            axes[2, t].set_title(f'Predicted VJP t={t+1}')
            if t == 0:
                axes[2, t].set_ylabel('Predicted VJP')
            fig.colorbar(im, ax=axes[2, t], fraction=0.046, pad=0.04)
            
        # Plot error
        for t in range(num_timesteps):
            error_np = np.abs(true_vjp[t] - pred_vjp[t])
            error_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max())) # scale according to true vjp
            norm_error = SymLogNorm(linthresh=0.1 * error_range, vmin=-error_range, vmax=error_range)
            im = axes[3, t].imshow(error_np, cmap='magma', norm=norm_error)
            axes[3, t].set_title(f'Error t={t+1}')
            if t == 0:
                axes[3, t].set_ylabel('Error')
            fig.colorbar(im, ax=axes[3, t], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def plot_vjp_results(self, vector, true_vjp, pred_vjp, title_suffix=""):
        """
        Plot vector, true VJP, and predicted VJP.
        
        Args:
            vector: Input vector
            true_vjp: True vector-Jacobian product
            pred_vjp: Predicted vector-Jacobian product
            title_suffix: Suffix for plot title
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(15, 5))
        plt.rcParams.update({'font.size': 14})
        
        # Input vector
        plt.subplot(1, 3, 1)
        vector_np = vector.numpy()
        vector_range = max(abs(vector_np.min()), abs(vector_np.max()))
        norm = SymLogNorm(linthresh=0.1 * vector_range, vmin=-vector_range, vmax=vector_range)
        plt.imshow(vector_np, cmap='seismic', norm=norm)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Vector {title_suffix}")
        
        # True VJP
        plt.subplot(1, 3, 2)
        true_vjp_np = true_vjp.numpy()
        true_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max()))
        norm_true = SymLogNorm(linthresh=0.1 * true_range, vmin=-true_range, vmax=true_range)
        plt.imshow(true_vjp_np, cmap='seismic', norm=norm_true)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"True VJP {title_suffix}")
        
        # Predicted VJP
        plt.subplot(1, 3, 3)
        pred_vjp_np = pred_vjp.numpy()
        pred_range = max(abs(pred_vjp_np.min()), abs(pred_vjp_np.max()))
        norm_pred = SymLogNorm(linthresh=0.1 * pred_range, vmin=-pred_range, vmax=pred_range)
        plt.imshow(pred_vjp_np, cmap='seismic', norm=norm_pred)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Predicted VJP {title_suffix}")
        
        return fig

class NSVisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing NS predictions."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        if (trainer.current_epoch - 1) % self.plot_interval == 0:
            train_loader = trainer.train_dataloader

            # Re-create the dataloader with shuffle=False for plotting same samples
            new_train_loader = DataLoader(
                dataset=train_loader.dataset,
                batch_size=train_loader.batch_size,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                collate_fn=train_loader.collate_fn,
                drop_last=train_loader.drop_last,
                shuffle=False,
            )

            self.plot_ns_results(trainer, pl_module, new_train_loader, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():
            self.plot_ns_results(trainer, pl_module, trainer.val_dataloaders, "val")
            
    def plot_ns_data(self, x, y, pred):
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        
        # Convert tensors to numpy arrays
        x_np = x.squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy() 
        pred_np = pred.squeeze().cpu().numpy()
        
        # Calculate global min/max
        vmin = min(np.min(y_np), np.min(pred_np))
        vmax = max(np.max(y_np), np.max(pred_np))
        
        # Input
        im0 = axes[0].imshow(x_np, cmap='jet', vmin=np.min(x_np), vmax=np.max(x_np))
        axes[0].set_title('Input Vorticity')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Output
        im1 = axes[1].imshow(y_np, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title('Output Vorticity')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Prediction
        im2 = axes[2].imshow(pred_np, cmap='jet', vmin=vmin, vmax=vmax)
        axes[2].set_title('Predicted Vorticity')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Error
        im3 = axes[3].imshow(np.abs(y_np - pred_np), cmap='magma')
        axes[3].set_title('Error')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
            
    def plot_ns_results(self, trainer, pl_module, dataloader, tag):

        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            x = batch['x'].to(pl_module.device)
            y = batch['y'].to(pl_module.device)
            Jvp = batch['Jvp'].to(pl_module.device)
            v = batch['v'].to(pl_module.device)
            
            with torch.no_grad():
                pred_Jvp = pl_module(x)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            vec_idx = 0    # Visualize first vector
            
            # Plot input, output, and predicted vorticity
            fig = self.plot_ns_data(x, y, pred_Jvp)
            filename = f"ns_{tag}_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/forward", trainer.current_epoch)
            plt.close(fig)

class NS_JVP_VisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing JVPs."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        if (trainer.current_epoch - 1) % self.plot_interval == 0:
            self.plot_jvp_results(trainer, pl_module, trainer.train_dataloader, "train")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():
            self.plot_jvp_results(trainer, pl_module, trainer.val_dataloaders, "val")
            
    def plot_jvp_data(self, v, Jvp, pred_Jvp):
        
        v = v.squeeze(0).cpu().numpy()
        Jvp = Jvp.squeeze(0).cpu().numpy()
        pred_Jvp = pred_Jvp.squeeze(0).cpu().numpy()

        vmin = min(np.min(Jvp), np.min(pred_Jvp))
        vmax = max(np.max(Jvp), np.max(pred_Jvp))
        
        cols = min(4, Jvp.shape[2])    
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))    
        
        for eig_idx in range(cols):
            x_np = v[:, :, eig_idx]
            y_np = Jvp[:, :, eig_idx] 
            pred_np = pred_Jvp[:, :, eig_idx]
            
            # Input
            im0 = axes[0, eig_idx].imshow(x_np, cmap='jet', vmin=np.min(v), vmax=np.max(v))
            axes[0, eig_idx].set_title(f'Vector no. {eig_idx+1}')
            fig.colorbar(im0, ax=axes[0, eig_idx], fraction=0.046, pad=0.04)

            # Output
            im1 = axes[1, eig_idx].imshow(y_np, cmap='jet', vmin=vmin, vmax=vmax)
            axes[1, eig_idx].set_title(f'Jvp')
            fig.colorbar(im1, ax=axes[1, eig_idx], fraction=0.046, pad=0.04)
            
            # Prediction
            im2 = axes[2, eig_idx].imshow(pred_np, cmap='jet', vmin=vmin, vmax=vmax)
            axes[2, eig_idx].set_title(f'Pred Jvp')
            fig.colorbar(im2, ax=axes[2, eig_idx], fraction=0.046, pad=0.04)
            
            # Error
            im3 = axes[3, eig_idx].imshow(np.abs(y_np - pred_np), cmap='magma')
            axes[3, eig_idx].set_title(f'Error')
            fig.colorbar(im3, ax=axes[3, eig_idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
        
        
    def plot_jvp_results(self, trainer, pl_module, dataloader, tag):

        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            x = batch['x'].to(pl_module.device)
            y = batch['y'].to(pl_module.device)
            Jvp = batch['Jvp'].to(pl_module.device)
            v = batch['v'].to(pl_module.device)
            
            with torch.no_grad():
                pred_Jvp = pl_module(x)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            vec_idx = 0    # Visualize first vector
            
            Jvp_pred = pl_module.compute_Jvp(x, v).detach()
            fig = self.plot_jvp_data(v, Jvp, Jvp_pred)
            filename = f"jvp_{tag}_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/jvp", trainer.current_epoch)
            plt.close(fig)

class NS_Inversion_VisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing NS inversion."""
    
    def on_train_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        
        self.invert_ns(trainer, pl_module, trainer.train_dataloader, "train")
        self.invert_ns(trainer, pl_module, trainer.val_dataloaders, "val")
        
    def plot_ns_data(self, x0, x, y, x_pred):
        x0 = x0.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        x_pred = x_pred.squeeze().cpu().numpy()
        
        vmin = min(np.min(x0), np.min(x), np.min(x_pred))
        vmax = max(np.max(x0), np.max(x), np.max(x_pred))
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        
        # Row 1
        im0 = axes[0].imshow(x0, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title('Starting Vorticity')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Row 2
        im0 = axes[1].imshow(y, cmap='jet', vmin=np.min(y), vmax=np.max(y))
        axes[1].set_title('Final Vorticity')
        fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Row 3
        im1 = axes[2].imshow(x, cmap='jet', vmin=vmin, vmax=vmax)
        axes[2].set_title('Input Vorticity')
        fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Row 4
        im2 = axes[3].imshow(x_pred, cmap='jet', vmin=vmin, vmax=vmax)
        axes[3].set_title('Surrogate Inverted V0')
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
        
        # Row 5
        im3 = axes[4].imshow(np.abs(x - x_pred), cmap='magma')
        axes[4].set_title('Error')
        fig.colorbar(im3, ax=axes[4], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
        
    def invert_ns(self, trainer, pl_module, dataloader, tag):
        """Invert NS."""

        plot_counter = 0
        l2_final = []
        ssim_final = []

        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            x = batch['x'].to(pl_module.device)
            y = batch['y'].to(pl_module.device)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            vec_idx = 0    # Visualize first vector
            
            iters = 500
            x0 = torch.zeros_like(x)
            x0_init = x0.clone().detach()
            x0.requires_grad = True # Enable gradients for x0

            for iteration in range(iters):
                x_pred = pl_module(x0)
                loss = 0.5 * torch.sum((x_pred - y) ** 2)
                
                loss.backward()
                with torch.no_grad():
                    x0.data -= pl_module.hparams.learning_rate * x0.grad.data # Use .data to modify tensor in-place
                    x0.grad.zero_() # Zero out gradients
                    
                x0_np = x0.detach().cpu().numpy()[0, 0]
                x_np = x.detach().cpu().numpy()[0, 0]
                ssim_val = ssim(x0_np, x_np, data_range=x0_np.max()-x0_np.min())
                l2_val = np.linalg.norm(x0_np - x_np) / np.linalg.norm(x_np)
                
                trainer.logger.experiment[f"{tag}/sample_{batch['idx'][batch_idx]}/metrics_chart"].append(
                    {
                        "SSIM": ssim_val,
                        "L2": l2_val,
                    }
                )
                    
            l2_final.append(l2_val)
            ssim_final.append(ssim_val)
                
            fig = self.plot_ns_data(x0_init, x, y, x0.detach())
            filename = f"ns_inversion_{tag}_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/ns_inversion", trainer.current_epoch)
            plt.close(fig)

        trainer.logger.experiment[f"{tag}/metrics_chart"].append(
            {
                "L2": np.mean(l2_final),
                "SSIM": np.mean(ssim_final),
            }
        )
