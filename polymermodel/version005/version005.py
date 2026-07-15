import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile

class DenseCiliaModel3D(nn.Module):
    def __init__(self, num_segments, segment_length, width, height, known_root, dt_frame=1.0):
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.W = width
        self.H = height
        self.dt = dt_frame
        
        # Lock the known root position (not optimized)
        self.register_buffer("root_pos", torch.tensor(known_root, dtype=torch.float32))
        
        # --- GLOBAL KINEMATIC PARAMETERS ---
        self.omega = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.base_amplitude = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.spatial_lag = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, N_frames, sigma=1.5):
        """
        Renders the entire [Frames, Height, Width] volume in a single, 
        fully-vectorized execution pass without loops.
        """
        device = self.root_pos.device
        
        # 1. Create spatial coordinate grids
        # Shape: [1, Height, Width, 1]
        y_grid = torch.arange(self.H, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        x_grid = torch.arange(self.W, dtype=torch.float32, device=device).view(1, 1, -1, 1)
        
        # 2. Setup backbone sampling parameters
        samples_per_seg = 12
        total_samples = self.num_segments * samples_per_seg
        # Shape: [1, total_samples]
        s_mesh = torch.linspace(0, self.num_segments, total_samples, device=device).view(1, -1)
        
        norm_height = s_mesh / self.num_segments
        scaled_amp = self.base_amplitude * (norm_height ** 1.5)
        
        # 3. Compute continuous wave kinematics across all frames simultaneously
        t_vec = torch.arange(N_frames, dtype=torch.float32, device=device).view(-1, 1) * self.dt
        
        # Broadcast times and spatial positions to compute phases: [N_frames, total_samples]
        phase = (self.omega * t_vec) - (self.spatial_lag * s_mesh)
        angles = scaled_amp * torch.sin(phase)
        
        # 4. Integrate the structural backbone coordinates over time
        dx = self.L * torch.sin(angles) / samples_per_seg
        dy = self.L * torch.cos(angles) / samples_per_seg
        
        # Shape: [N_frames, total_samples]
        cx = self.root_pos[0] + torch.cumsum(dx, dim=1)
        cy = self.root_pos[1] - torch.cumsum(dy, dim=1)
        
        # Reshape backbone for spatial broadcasting: [N_frames, 1, 1, total_samples]
        cx = cx.view(N_frames, 1, 1, total_samples)
        cy = cy.view(N_frames, 1, 1, total_samples)
        
        # 5. Compute Euclidean distance field from grid to backbone
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        min_dist_sq, _ = torch.min(dist_sq, dim=3)  # [N_frames, Height, Width]
        
        # Render Gaussian intensity profile
        volume = torch.exp(-min_dist_sq / (2 * (sigma ** 2)))
        
        # Zero out any rendering artifacts below the root base
        mask = (y_grid.squeeze(-1) <= self.root_pos[1]) # [1, Height, Width]
        volume = volume * mask
        
        return volume


def save_volume_hyperstack(volume_tensor, filepath):
    """
    Saves a [Frames, Height, Width] tensor as a standardized Fiji hyperstack:
    [Slices (Y), Frames (T), Channels (C), Width (X)]
    """
    # Permute from [T, Y, X] to [Y, T, X]
    arr = volume_tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 0, 2))
    # Add dummy channel dimension for standard ImageJ hyperstack parsing
    arr = np.expand_dims(arr, axis=2).astype(np.float32)
    
    tifffile.imwrite(filepath, arr, imagej=True)
    print(f"Exported dense 3D stack to '{filepath}' (Shape: {arr.shape})")


if __name__ == "__main__":
    FRAMES = 200
    WIDTH = 64
    HEIGHT = 64  
    KNOWN_ROOT = [32.0, 62.0]  # Locked coordinate
    NUM_SEGMENTS = 12
    SEGMENT_LENGTH = 4.0
    
    device = torch.device("cpu")
    print(f"Running on device: {device}")
    
    # ==========================================
    # 1. GENERATE DENSE GROUND TRUTH
    # ==========================================
    print("\n--- 1. Generating Ground Truth Continuous Volume ---")
    gt_model = DenseCiliaModel3D(NUM_SEGMENTS, SEGMENT_LENGTH, WIDTH, HEIGHT, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        gt_model.omega.copy_(torch.tensor(0.16))
        gt_model.base_amplitude.copy_(torch.tensor(0.60))
        gt_model.spatial_lag.copy_(torch.tensor(0.40))
        
        gt_volume = gt_model(N_frames=FRAMES, sigma=1.5)
        save_volume_hyperstack(gt_volume, "dense_ground_truth.tif")

# ==========================================
    # 2. OPTIMIZATION LOOP (ANTI-HARMONIC TRAP)
    # ==========================================
    print("\n--- 2. Optimization Loop ---")
    fit_model = DenseCiliaModel3D(NUM_SEGMENTS, SEGMENT_LENGTH, WIDTH, HEIGHT, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        fit_model.omega.copy_(torch.tensor(0.07))
        fit_model.base_amplitude.copy_(torch.tensor(0.25))
        fit_model.spatial_lag.copy_(torch.tensor(0.15))
        
    # We use a slightly higher learning rate for omega to escape local harmonic traps
    optimizer = optim.Adam([
        {'params': [fit_model.base_amplitude, fit_model.spatial_lag], 'lr': 0.02},
        {'params': [fit_model.omega], 'lr': 0.04} 
    ])

    for step in range(121):  # Increased steps slightly to ensure full convergence
        optimizer.zero_grad()
        
        # Coarse-to-fine rendering: start with a slightly softer blur if step < 30
        # to expand the temporal gradient overlap, then tighten to 1.5
        current_sigma = 2.5 if step < 30 else 1.5
        
        est_volume = fit_model(N_frames=FRAMES, sigma=current_sigma)
        
        # If we changed sigma, evaluate ground truth at that temporary scale too
        if current_sigma != 1.5:
            with torch.no_grad():
                gt_volume_scaled = gt_model(N_frames=FRAMES, sigma=current_sigma)
            loss = F.mse_loss(est_volume, gt_volume_scaled)
        else:
            loss = F.mse_loss(est_volume, gt_volume)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step:03d} | Total Loss: {loss.item():.6f} | "
                  f"Omega: {fit_model.omega.item():.4f} (GT: 0.16) | "
                  f"Amp: {fit_model.base_amplitude.item():.4f} (GT: 0.60) | "
                  f"Lag: {fit_model.spatial_lag.item():.4f} (GT: 0.40)")

    # ==========================================
    # 3. EXPORT RECONSTRUCTION
    # ==========================================
    print("\n--- 3. Exporting Fitted Reconstruction ---")
    with torch.no_grad():
        fitted_volume = fit_model(N_frames=FRAMES, sigma=1.5)
        save_volume_hyperstack(fitted_volume, "dense_fitted_reconstruction.tif")
        
    print("\nFinished! Open both files in Fiji. Because the roots match, "
          "the physical structures will align perfectly upon convergence.")