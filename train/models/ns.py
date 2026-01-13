import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from physicsnemo.models.fno import FNO
from typing import Dict, Any, Optional, Tuple
import gc  # for garbage collection

class NSModel(pl.LightningModule):
    """PyTorch Lightning module for NS saturation prediction with optional Jacobian regularization."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        decoder_layer_size: int = 128,
        num_fno_layers: int = 3,
        num_fno_modes: list = [2, 15, 15],
        padding: int = 3,
        dimension: int = 3,
        latent_channels: int = 64,
        loss_type: str = "L2",
        train_eigen_count: int = 8,
        reg_param: float = 0.01,
        scale_factor: float = 5500.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ckpt_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # use manual optimization for chunked backward
        self.automatic_optimization = False
        
        self.model = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_layer_size=decoder_layer_size,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            dimension=dimension,
            latent_channels=latent_channels
        )
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model_state_dict = checkpoint['state_dict']
            corrected_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('model.'):
                    corrected_state_dict[key[6:]] = value
            self.model.load_state_dict(corrected_state_dict)
        
        self.loss_type = loss_type
        self.reg_param = reg_param
        self.scale_factor = scale_factor
        self.train_eigen_count = train_eigen_count
        self.train_rel_l2_loss = 0.0
        self.train_jac_loss    = 0.0
        self.val_rel_l2_loss   = 0.0
        self.val_jac_loss      = 0.0

    def forward(self, x):
        return self.model(x)
    
    def relative_l2_loss(self, true, pred):
        return torch.norm(true - pred) / torch.norm(true)
        # return torch.nn.functional.mse_loss(target=true, input=pred)

    def compute_Jvp(self, x, v, create_graph=False):
        Jvp = torch.zeros_like(v)
        for eig_idx in range(v.shape[-1]):
            _, jvp_value = torch.autograd.functional.jvp(self.forward, x, v[:, :, :, eig_idx].unsqueeze(1), create_graph=create_graph)
            Jvp[:, :, :, eig_idx] = jvp_value.squeeze()
        x.requires_grad_(False)
        return Jvp

    def training_step(self, batch, batch_idx):
        # manual optimization
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x, y = batch['x'], batch['y']

        # x = x.reshape(-1, 2, 512, 256)
        # y = y.reshape(-1, 512, 256)
        # print("x", x.shape, y.shape)
        
        output = self.forward(x)
        rel_l2 = self.relative_l2_loss(y.squeeze(), output.squeeze())
        self.train_rel_l2_loss = rel_l2.detach()
        self.log('train_rel_l2_loss', rel_l2, prog_bar=True, on_step=True, on_epoch=True)

        # backward the rel_l2 loss
        self.manual_backward(rel_l2)

        # Jacobian regularization
        jac_loss = torch.tensor(0.0, device=x.device)
        K = self.train_eigen_count
        if 'v' in batch:
            v = batch['v'][..., :K]
            true_Jvp = batch['Jvp'][..., :K]
            for k in range(K):
                v_dir = v[..., k].unsqueeze(-1)
                _, jval = torch.autograd.functional.jvp(
                    self.forward, x, v_dir.reshape_as(x), create_graph=True
                )
                target = true_Jvp[..., k]
                loss_k = self.relative_l2_loss(jval, target)
                # accumulate for logging
                jac_loss += loss_k.detach()
                # backprop scaled to average
                scaled = self.reg_param * loss_k / K
                self.manual_backward(scaled)

                # free fragment
                del jval, loss_k, scaled
                torch.cuda.empty_cache()

            jac_loss /= K
            self.train_jac_loss = jac_loss
            self.log('train_jac_loss', jac_loss, prog_bar=True, on_step=True, on_epoch=True)

        # optimizer step
        optimizer.step()

        total_loss = rel_l2 + self.reg_param * jac_loss
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        print("Validation x y", x.shape, y.shape)
        y_pred = self.forward(x)
        val_l2 = self.relative_l2_loss(y.squeeze(), y_pred.squeeze())
        if 'v' in batch:
            v = batch['v'][..., :self.train_eigen_count]
            true_Jvp = batch['Jvp'][..., :self.train_eigen_count]
            jac_loss = torch.tensor(0.0, device=x.device)
            K = v.shape[-1]
            for k in range(K):
                _, jval = torch.autograd.functional.jvp(
                    self.forward, x, v[..., k].unsqueeze(-1).reshape_as(x), create_graph=False
                )
                jac_loss += self.relative_l2_loss(jval, true_Jvp[..., k]).detach()
            jac_loss /= K
            self.val_jac_loss    = jac_loss.detach()
            self.log('val_jac_loss', jac_loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            v = None
            true_Jvp = None
            # self.val_jac_loss = None
        self.val_rel_l2_loss = val_l2.detach()
        self.log('val_rel_l2_loss', val_l2, prog_bar=True, on_step=True, on_epoch=True)
        return {'val_loss': val_l2}

    def _clear_memory(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_end(self): self._clear_memory()
    def on_validation_epoch_end(self): self._clear_memory()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.99
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'interval': 'step'}}

