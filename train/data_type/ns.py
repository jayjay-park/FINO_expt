import os
import torch
import numpy as np
import h5py
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nx,
        ny,
        eigen_count,
        sample_paths: List[str]):
        self.nx = nx
        self.ny = ny
        self.eigen_count = eigen_count
        self.sample_paths = sample_paths
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.sample_paths[idx], 'r') as f:
            x = f['x'][:].astype(np.float32)
            y = f['y'][:].astype(np.float32)
            v = f['v'][:].astype(np.float32)
            Jvp = f['Jvp'][:].astype(np.float32)
            L = f['L'][:].astype(np.float32)
        # print("L", L)
            
        # extract the batch and sample number from the path
        # ex path: /home/ubuntu/DeFINO/datasets/dataset_NS_batch2/samples/sample_6.h5
        path_parts = self.sample_paths[idx].split('/')
        batch_num = path_parts[-3]
        sample_num = path_parts[-1].split('.')[0]
        id_str = f"b={batch_num}_s={sample_num}"
            
        item = {
            'x': x.reshape(1, self.nx, self.ny),
            'y': y.reshape(1, self.nx, self.ny),
            'Jvp': Jvp.reshape(self.nx, self.ny, -1)[:, :, :self.eigen_count],
            'v': v.reshape(self.nx, self.ny, -1)[:, :, :self.eigen_count],
            'L': L,
            'sample_path': self.sample_paths[idx],
            'idx': id_str
        }
        return item

class NSDataLoader:
    def __init__(
        self,
        nx: int,
        ny: int,
        eigen_count: int,
        sample_directories: List[str],
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True
    ):
        self.nx = nx
        self.ny = ny
        self.eigen_count = eigen_count
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_paths = []

        for directory in sample_directories:
            for file in os.listdir(os.path.join(directory, 'samples')):
                if file.endswith('.h5'):
                    self.sample_paths.append(os.path.join(directory, 'samples', file))
        
    def get_dataloader(self, offset: int, limit: int, shuffle: bool = True):
        dataset = CustomDataset(self.nx, self.ny, self.eigen_count, self.sample_paths[offset:offset+limit])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)        

# ============================================================
#  Velocity Model Building (VMB) Dataset Loader
# ============================================================

class VMBCustomDataset(Dataset):
    """
    VMB dataset with normalization:
      - Velocity fields divided by global v_max (positive)
      - RTM images divided by global max(abs(y))
    """
    def __init__(self, nx, ny, sample_paths, norm_path=None, eigen_count: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        self.sample_paths = sample_paths
        self.eigen_count = eigen_count

        # Load normalization constants
        if norm_path is not None:
            stats = np.load(norm_path)
            self.v_max = float(stats["v_max"])
            self.y_max = float(stats["y_max"])
            print(f"🔹 Loaded normalization constants: v_max={self.v_max:.6f}, y_max={self.y_max:.6f}")
        else:
            self.v_max = None
            self.y_max = None


    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        with h5py.File(path, "r") as f:
            x = f["x"][:].astype(np.float32)  # (nx, ny, 2)
            y = f["y"][:].astype(np.float32) # y = f["y_ext_13"][:].astype(np.float32)  # (nx, ny)
            
            bundle_id = f.attrs.get("bundle_id", -1)
            background_tag = f.attrs.get("background_tag", "unknown")
        
        # derive a reproducible id string (b=batch_dir, s=sample_name)
        path_parts = self.sample_paths[idx].split('/')
        batch_num = path_parts[-3]
        sample_num = path_parts[-1].split('.')[0]
        id_str = f"b={batch_num}_s={sample_num}"

        # -----------------------------
        # 🔸 Apply normalization
        # -----------------------------
        if self.v_max is not None:
            x = x / self.v_max
        if self.y_max is not None:
            y = y / self.y_max #* 500

        # Convert to torch tensors and ensure channel-first layout (C, H, W)
        x = torch.from_numpy(x)
        if x.ndim != 3:
            x = x.unsqueeze(0)

        y = torch.from_numpy(y)
        if y.ndim == 2:
            y = y.unsqueeze(0)  # (1, H, W)

        # Attempt to load v and Jvp (multiple possible key names)
        v_tensor = None
        Jvp_tensor = None
        with h5py.File(path, "r") as f2:
            if "v" in f2:
                v_np = f2["v"][:].astype(np.float32)
                # print("v_np", v_np.shape)
                # expected saved shape: (H, W, r) -> convert to (r, H, W)
                if v_np.ndim == 3:
                    v_np = np.transpose(v_np, (1, 2, 0))
                elif v_np.ndim == 2:
                    v_np = v_np[np.newaxis, ...]
                # apply eigen_count if set
                if self.eigen_count is not None:
                    v_np = v_np[: self.eigen_count]
                v_tensor = torch.from_numpy(v_np)
                # print("v_tensor", v_tensor.shape)

            # 'Jvp' or 'Jv' may be present depending on how files were written
            if "Jvp" in f2:
                Jvp_np = f2["Jvp"][:].astype(np.float32)
            elif "Jv" in f2:
                Jvp_np = f2["Jv"][:].astype(np.float32)
            else:
                Jvp_np = None

            if Jvp_np is not None:
                if Jvp_np.ndim == 3:
                    # Jvp_np = np.transpose(Jvp_np)
                    Jvp_np = np.transpose(Jvp_np, (1, 2, 0)) / self.y_max
                elif Jvp_np.ndim == 2:
                    Jvp_np = Jvp_np[np.newaxis, ...]
                if self.eigen_count is not None:
                    Jvp_np = Jvp_np[: self.eigen_count]
                Jvp_tensor = torch.from_numpy(Jvp_np)

        return {
            "x": x,
            "y": y,
            "v": v_tensor,
            "Jvp": Jvp_tensor,
            "bundle_id": bundle_id,
            "background_tag": background_tag,
            "path": path,
            "idx": id_str,
        }


class VMBDataLoader:
    """
    DataLoader wrapper for VMB dataset.
    Automatically finds .h5 files and loads (x, y) tensors.
    """
    def __init__(self, nx, ny, sample_directories, batch_size=1, num_workers=1, pin_memory=True, norm_path=None):
        self.nx = nx
        self.ny = ny
        self.sample_directories = sample_directories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_paths = []
        self.norm_path = norm_path

        # Collect all .h5 files
        for directory in sample_directories:
            for file in os.listdir(directory):
                if file.endswith(".h5"):
                    self.sample_paths.append(os.path.join(directory, file))
        self.sample_paths.sort()

        print(f"🧩 Found {len(self.sample_paths)} samples in {len(sample_directories)} directories.")

    def get_dataloader(self, offset=0, limit=None, shuffle=True):
        paths = self.sample_paths[offset:offset + limit] if limit else self.sample_paths
        dataset = VMBCustomDataset(self.nx, self.ny, paths, self.norm_path)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
