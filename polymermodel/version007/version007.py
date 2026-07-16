import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile

class RigidRollingCiliaModel(nn.Module):
    def __init__(self, num_segments, segment_length, width, height, root_pos, num_harmonics=3, max_relative_angle=0.8):
        """
        Model that optimizes kinematics with built-in self-avoidance and maximum bending limits.
        
        Parameters:
            max_relative_angle (float): Hard limit on angle difference between adjacent segments (in radians).
        """
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.W = width
        self.H = height
        self.K = num_harmonics
        self.max_relative_angle = max_relative_angle
        
        # Known, fixed root coordinates
        self.register_buffer("root_pos", torch.tensor(root_pos, dtype=torch.float32))
        
        # Learnable temporal parameters
        self.omega = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.t0_rolling = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t1_dense = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # Row phase shifts
        self.row_phases = nn.Parameter(torch.zeros(height, dtype=torch.float32)) 
        
        # Shape kinematics parameters (represented as relative angle differences between segments)
        self.a0_diff = nn.Parameter(torch.zeros(num_segments, dtype=torch.float32))
        self.ak_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        self.bk_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        
        with torch.no_grad():
            self.ak_diff.normal_(0.0, 0.02)
            self.bk_diff.normal_(0.0, 0.02)
            self.row_phases.normal_(0.0, 0.1)

    def _get_bounded_angles(self, t_base, phase_offsets):
        """
        Computes segment angles from relative angle differences, enforcing hard bounds 
        via tanh to prevent unphysical looping or self-intersection.
        """
        device = self.root_pos.device
        samples_per_seg = 12
        total_samples = self.num_segments * samples_per_seg
        
        s_mesh = torch.linspace(0, self.num_segments - 1, total_samples, device=device)
        idx_low = torch.clamp(torch.floor(s_mesh).long(), 0, self.num_segments - 1)
        idx_high = torch.clamp(idx_low + 1, 0, self.num_segments - 1)
        weight_high = s_mesh - idx_low.float()
        weight_low = 1.0 - weight_high
        
        # Linearly interpolate relative angle parameters along the continuous spatial mesh
        mesh_a0_diff = (self.a0_diff[idx_low] * weight_low + self.a0_diff[idx_high] * weight_high).view(1, 1, -1)
        mesh_ak_diff = (self.ak_diff[:, idx_low] * weight_low + self.ak_diff[:, idx_high] * weight_high)
        mesh_bk_diff = (self.bk_diff[:, idx_low] * weight_low + self.bk_diff[:, idx_high] * weight_high)
        
        phase_grid = (self.omega * t_base + phase_offsets).unsqueeze(-1)
        
        # Accumulate Fourier relative terms
        rel_angles = mesh_a0_diff.expand(self.H, t_base.shape[1], -1).clone()
        for k in range(1, self.K + 1):
            cos_kt = torch.cos(k * phase_grid)
            sin_kt = torch.sin(k * phase_grid)
            rel_angles += (cos_kt * mesh_ak_diff[k-1].view(1, 1, -1)) + (sin_kt * mesh_bk_diff[k-1].view(1, 1, -1))
            
        # Hard Rigidity Constraint: Enforce strict relative limits per unit length using scaled tanh
        # This physically blocks the cilium from bending sharper than max_relative_angle
        rel_angles_bounded = self.max_relative_angle * torch.tanh(rel_angles)
        
        # Integrate relative angles along the filament to get absolute angles
        # Integrating step-by-step prevents the tip from making wild 360 loops
        angles = torch.cumsum(rel_angles_bounded, dim=2)
        
        # Soften movement close to the basal attachment point
        #root_clamp = torch.clamp(torch.linspace(0.0, 2.0, total_samples, device=device), 0.0, 1.0).view(1, 1, -1)
        return angles #* root_clamp

    def _compute_coordinates(self, angles):
        """Computes the continuous 2D joint coordinates [H, T, S, 2] of the filament."""
        samples_per_seg = 12
        dx = self.L * torch.sin(angles) / samples_per_seg
        dy = self.L * torch.cos(angles) / samples_per_seg
        
        cx = self.root_pos[0] + torch.cumsum(dx, dim=2)
        cy = self.root_pos[1] - torch.cumsum(dy, dim=2)
        return torch.stack([cx, cy], dim=-1)

    def _render_image_grid(self, coords, sigma):
        """Generates Gaussian intensity projections [T, H, W] from the 2D coordinates."""
        device = self.root_pos.device
        cx, cy = coords[..., 0], coords[..., 1]
        
        y_grid = torch.arange(self.H, dtype=torch.float32, device=device).view(self.H, 1, 1, 1)
        x_grid = torch.arange(self.W, dtype=torch.float32, device=device).view(1, 1, self.W, 1)
        
        dist_sq = (x_grid - cx.unsqueeze(2))**2 + (y_grid - cy.unsqueeze(2))**2
        min_dist_sq, _ = torch.min(dist_sq, dim=3)
        
        intensity = torch.exp(-min_dist_sq / (2 * (sigma ** 2)))
        return intensity.permute(1, 0, 2)

    def forward_dense_timelapse(self, N_frames, dt_frame, sigma, override_phi_zero=False):
        device = self.root_pos.device
        t_frame_starts = self.t1_dense + torch.arange(N_frames, dtype=torch.float32, device=device).unsqueeze(0) * dt_frame
        t_base = t_frame_starts.expand(self.H, -1)
        
        phase_offsets = torch.zeros_like(self.row_phases).view(-1, 1) if override_phi_zero else self.row_phases.view(-1, 1)
            
        angles = self._get_bounded_angles(t_base, phase_offsets)
        coords = self._compute_coordinates(angles)
        return self._render_image_grid(coords, sigma), coords

    def forward_initial_rolling_shutter(self, dt_slice, sigma):
        device = self.root_pos.device
        t_base = self.t0_rolling + (torch.arange(self.H, dtype=torch.float32, device=device).unsqueeze(1) * dt_slice)
        phase_offsets = torch.zeros_like(self.row_phases).view(-1, 1)
        
        angles = self._get_bounded_angles(t_base, phase_offsets)
        coords = self._compute_coordinates(angles)
        return self._render_image_grid(coords, sigma).squeeze(0), coords


def compute_self_avoidance_loss_optimized(coords, num_segments, safety_dist=3.0, time_stride=4):
    """
    Optimized, memory-efficient self-avoidance loss.
    Extracts only the actual skeletal joint coordinates instead of all interpolated pixels,
    and strides through time to prevent OOM errors on large 3D timelapse inputs.
    """
    device = coords.device
    H, T, total_samples, _ = coords.shape
    samples_per_seg = total_samples // num_segments
    
    # 1. Downsample spatially to skeletal joint nodes only (drastically reduces coordinates checks)
    joint_indices = torch.arange(0, total_samples, samples_per_seg, device=device)
    coords_joints = coords[:, ::time_stride, joint_indices, :]  # Shape: [H, T_strided, S_joints, 2]
    
    H, T_stride, S_j, _ = coords_joints.shape
    if S_j < 4:
        return torch.tensor(0.0, device=device)
    
    # 2. Reshape to flat batch representation for fast vectorized distance computation: [Batch, S_j, 2]
    coords_flat = coords_joints.reshape(-1, S_j, 2)
    
    # 3. Calculate pairwise Euclidean distances between joints: [Batch, S_j, S_j]
    diff = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    dist = torch.sqrt(dist_sq + 1e-8)
    
    # 4. Mask out immediate neighbors (adjacent segments on the filament chain shouldn't push away)
    mask = torch.ones((S_j, S_j), device=device)
    for i in range(S_j):
        mask[i, max(0, i-1):min(S_j, i+2)] = 0.0 
        
    violation = torch.clamp(safety_dist - dist, min=0.0)
    masked_violation = violation * mask.unsqueeze(0)
    
    return torch.mean(masked_violation ** 2)


def load_and_normalize_tiff(file_path, device):
    img = tifffile.imread(file_path).astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)
    return torch.tensor(img_normalized, dtype=torch.float32, device=device)


if __name__ == "__main__":
    device = torch.device("cuda")
    os.makedirs("outputs", exist_ok=True)
    
    # Paths to inputs
    path_rolling_snapshot = "code/version007/2D_rolling_shutter.tif"  # Input 1 (2D)
    path_async_timelapse = "code/version007/2D_async.tif"        # Input 2 (3D stack)
    
    print("\n--- 1. Loading Datasets ---")
    try:
        rolling_target = load_and_normalize_tiff(path_rolling_snapshot, device)
        async_timelapse_target = load_and_normalize_tiff(path_async_timelapse, device)
    except FileNotFoundError as e:
        print(f"Missing file error: {e}. Please ensure input data exists.")
        exit(1)
        
    HEIGHT, WIDTH = rolling_target.shape
    T_frames, H_check, W_check = async_timelapse_target.shape
    print(f"Loaded snapshot: {rolling_target.shape} | Timelapse stack: {async_timelapse_target.shape}")

    # Geometry & Rigidity Constants
    NUM_SEGMENTS = 19
    SEGMENT_LENGTH = 4.0
    SIGMA_LINEWIDTH = 2.0
    DT_SLICE = 1.0
    DT_FRAME = 1.0
    
    # Weights for regularizers
    ANGLE_MAX = 0.06
    LAMBDA_RIGIDITY = 0.05       # Smoothness cost
    LAMBDA_AVOIDANCE = 1.0       # Anti-knotting repulsion weight
    SAFETY_DISTANCE = 4.0        # Keep non-adjacent segments at least 4 pixels apart
    TIME_STRIDE = 4              # Apply avoidance on every 4th frame (keeps memory footprint very low)

    ROLLING_WEIGHT = 0.01
    
    HARM_K = 2

    KNOWN_ROOT = [WIDTH / 2.0 - 1.0, HEIGHT - 23.0] 
    INI_FREQ = 0.2

    START_SIGMA = 8.0     # Large basin to capture global frequency and phase
    END_SIGMA = 2.0       # Tight line-width to capture fine shape details
    DECAY_STEPS = 400     # Number of steps to transition to fine details

    # Initialize model with strict 0.1 rad (~5.7 deg) maximum relative bend limit (strict rigidity)
    model = RigidRollingCiliaModel(
        num_segments=NUM_SEGMENTS,
        segment_length=SEGMENT_LENGTH,
        width=WIDTH,
        height=HEIGHT,
        root_pos=KNOWN_ROOT,
        num_harmonics = HARM_K,
        max_relative_angle=ANGLE_MAX
    ).to(device)
    model.omega.data = torch.tensor(INI_FREQ, dtype=torch.float32)
    
    optimizer = optim.Adam([
        {'params': [model.a0_diff, model.ak_diff, model.bk_diff], 'lr': 0.03},
        {'params': [model.row_phases], 'lr': 0.05},
        {'params': [model.t0_rolling, model.t1_dense], 'lr': 0.01},
        {'params': [model.omega], 'lr': 0.005}
    ])

    print("\n--- 2. Dual-Input Optimization (With Anti-Knotting Guards) ---")
    for step in range(1561):
        optimizer.zero_grad()

        # 0. Dynamically decay sigma
        if step < DECAY_STEPS:
            # Linear decay from START_SIGMA down to END_SIGMA
            current_sigma = START_SIGMA - (step / DECAY_STEPS) * (START_SIGMA - END_SIGMA)
        else:
            current_sigma = END_SIGMA
        
        # 1. Forward model outputs
        est_rolling, coords_rolling = model.forward_initial_rolling_shutter(DT_SLICE, sigma=current_sigma)
        est_timelapse, coords_timelapse = model.forward_dense_timelapse(T_frames, DT_FRAME, sigma=current_sigma, override_phi_zero=False)
        
        # 2. Physics Losses
        loss_rolling = F.mse_loss(est_rolling, rolling_target)
        loss_timelapse = F.mse_loss(est_timelapse, async_timelapse_target)
        
        # 3. Soft Penalties (Smoothness & Spatial Repulsion)
        loss_rigidity = torch.mean((model.a0_diff)**2) + torch.mean((model.ak_diff)**2) + torch.mean((model.bk_diff)**2)
        
        # Call optimized version to prevent CUDA OOM
       # loss_avoidance = compute_self_avoidance_loss_optimized(
      #      coords_timelapse, 
      #      num_segments=NUM_SEGMENTS, 
      #      safety_dist=SAFETY_DISTANCE,
      #      time_stride=TIME_STRIDE
       # )
        # --- PHASED LOSS STRATEGY ---
       # if step < 250:
       #     # Phase 1: Force the model to find the true temporal frequency using only the movie stack
       #     total_loss = loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity) + (LAMBDA_AVOIDANCE * loss_avoidance)
       # else:
       #     # Phase 2: Bring back the snapshot to resolve sub-pixel structural details
        #    total_loss = loss_rolling + loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity) + (LAMBDA_AVOIDANCE * loss_avoidance)
        #total_loss = loss_rolling*ROLLING_WEIGHT + loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity) + (LAMBDA_AVOIDANCE * loss_avoidance)
        
        total_loss = loss_rolling*ROLLING_WEIGHT + loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity)# + (LAMBDA_AVOIDANCE * loss_avoidance)
        #total_loss = loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity) + (LAMBDA_AVOIDANCE * loss_avoidance)

        total_loss.backward()
        optimizer.step()
        
        # Log Metrics
        if step % 50 == 0:
            print(f"Step {step:03d} | Rigidity: {LAMBDA_RIGIDITY * loss_rigidity.item():.5f} | "
#                  f"Timelapse Fit: {loss_timelapse.item():.5f} | Avoidance Cost: {loss_avoidance.item():.5f} | Freq: {model.omega.item():.5f}")
                  f"Rolling: {ROLLING_WEIGHT * loss_rolling.item():.5f} | "
                  f"Timelapse: {loss_timelapse.item():.5f} | "
        #          f"Avoidance: {loss_avoidance.item():.5f} | "
                  f"Freq: {model.omega.item():.5f}")
            
            # Save intermediate TIFF snapshot reconstructions for progress tracking
            with torch.no_grad():
                # Reconstruct perfect global shutter clean projection
                est_global, _ = model.forward_dense_timelapse(T_frames, DT_FRAME, sigma=SIGMA_LINEWIDTH, override_phi_zero=True)
                
                # Write individual multi-page or single files for debugging
                tifffile.imwrite(f"outputs/step_{step:03d}_rolling_fit.tif", est_rolling.cpu().numpy().astype(np.float32))
                tifffile.imwrite(f"outputs/step_{step:03d}_dense_timelapse.tif", est_timelapse.cpu().numpy().astype(np.float32))
                tifffile.imwrite(f"outputs/step_{step:03d}_clean_global.tif", est_global.cpu().numpy().astype(np.float32))
                
    print("\nOptimization Finished! All checkpoints saved under 'outputs/' folder.")