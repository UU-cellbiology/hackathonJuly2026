import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile

import math
import torch
import torch.nn as nn

class RigidRollingCiliaModel3D(nn.Module):
    def __init__(self, num_segments, segment_length, depth, height, width, root_pos, 
                 voxel_size_xy=1.0, voxel_size_z=1.0, num_harmonics=3):
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.D = depth
        self.H = height
        self.W = width
        self.K = num_harmonics
        
        self.z_scale = float(voxel_size_z / voxel_size_xy)
        self.register_buffer("root_pos", torch.tensor(root_pos, dtype=torch.float32))
        
        self.omega = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        
        # --- FIX 1: Parameterize phases per DEPTH slice D instead of height H ---
        self.depth_phases = nn.Parameter(torch.zeros(depth, dtype=torch.float32)) 
        
        self.theta_a0_diff = nn.Parameter(torch.zeros(num_segments, dtype=torch.float32))
        self.theta_ak_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        self.theta_bk_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        
        self.phi_a0_diff = nn.Parameter(torch.zeros(num_segments, dtype=torch.float32))
        self.phi_ak_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        self.phi_bk_diff = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        
        with torch.no_grad():
            self.theta_ak_diff.normal_(0.0, 0.02)
            self.theta_bk_diff.normal_(0.0, 0.02)
            self.phi_ak_diff.normal_(0.0, 0.02)
            self.phi_bk_diff.normal_(0.0, 0.02)
            self.depth_phases.normal_(0.0, 0.1)

        self.base_theta = 0.0
        self.base_phi = 0.0

    def _get_bounded_angles(self, t_base, phase_offsets, max_angle_current):
        """
        t_base shape: [D, T_frames]
        phase_offsets shape: [D, 1]
        """
        device = self.root_pos.device
        samples_per_seg = 12
        total_samples = self.num_segments * samples_per_seg
        
        s_mesh = torch.linspace(0, self.num_segments - 1, total_samples, device=device)
        idx_low = torch.clamp(torch.floor(s_mesh).long(), 0, self.num_segments - 1)
        idx_high = torch.clamp(idx_low + 1, 0, self.num_segments - 1)
        weight_high = s_mesh - idx_low.float()
        weight_low = 1.0 - weight_high
        
        m_theta_a0 = (self.theta_a0_diff[idx_low] * weight_low + self.theta_a0_diff[idx_high] * weight_high).view(1, 1, -1)
        m_theta_ak = (self.theta_ak_diff[:, idx_low] * weight_low + self.theta_ak_diff[:, idx_high] * weight_high)
        m_theta_bk = (self.theta_bk_diff[:, idx_low] * weight_low + self.theta_bk_diff[:, idx_high] * weight_high)
        
        m_phi_a0 = (self.phi_a0_diff[idx_low] * weight_low + self.phi_a0_diff[idx_high] * weight_high).view(1, 1, -1)
        m_phi_ak = (self.phi_ak_diff[:, idx_low] * weight_low + self.phi_ak_diff[:, idx_high] * weight_high)
        m_phi_bk = (self.phi_bk_diff[:, idx_low] * weight_low + self.phi_bk_diff[:, idx_high] * weight_high)
        
        phase_grid = (self.omega * t_base + phase_offsets).unsqueeze(-1) # [D, T_frames, 1]
        
        # --- FIX 2: Expand across self.D instead of self.H ---
        rel_theta = m_theta_a0.expand(self.D, t_base.shape[1], -1).clone()
        rel_phi = m_phi_a0.expand(self.D, t_base.shape[1], -1).clone()
        
        for k in range(1, self.K + 1):
            cos_kt = torch.cos(k * phase_grid)
            sin_kt = torch.sin(k * phase_grid)
            
            rel_theta += (cos_kt * m_theta_ak[k-1].view(1, 1, -1)) + (sin_kt * m_theta_bk[k-1].view(1, 1, -1))
            rel_phi += (cos_kt * m_phi_ak[k-1].view(1, 1, -1)) + (sin_kt * m_phi_bk[k-1].view(1, 1, -1))
            
        mag = torch.sqrt(rel_theta**2 + rel_phi**2 + 1e-8)

        # Arc-length gain factor: distal end (tip) gets up to 2x higher max angle allowance
        #s_norm = torch.linspace(0.0, 1.0, total_samples, device=device).view(1, 1, -1)
        #distal_multiplier = 1.0 + 1.5 * (s_norm ** 1.5)
        
        #mag_bounded = (max_angle_current * distal_multiplier) * torch.tanh(mag)

        mag_bounded = max_angle_current * torch.tanh(mag)
        
        scale_factor = mag_bounded / mag
        rel_theta_bounded = rel_theta * scale_factor
        rel_phi_bounded = rel_phi * scale_factor
        
        theta = torch.cumsum(rel_theta_bounded, dim=2) # [D, T_frames, total_samples]
        phi = torch.cumsum(rel_phi_bounded, dim=2)     # [D, T_frames, total_samples]
        
        return theta, phi

    def _compute_coordinates(self, theta, phi):
        samples_per_seg = 12
        step_len = self.L / samples_per_seg
        
        theta_mounted = theta + self.base_theta
        phi_mounted = phi + self.base_phi
        
        dz = step_len * torch.cos(theta_mounted)
        dy = step_len * torch.sin(theta_mounted) * torch.sin(phi_mounted)
        dx = step_len * torch.sin(theta_mounted) * torch.cos(phi_mounted)
        
        cz = self.root_pos[0] - torch.cumsum(dz, dim=2)
        cy = self.root_pos[1] - torch.cumsum(dy, dim=2)
        cx = self.root_pos[2] + torch.cumsum(dx, dim=2)
        
        return torch.stack([cz, cy, cx], dim=-1) # Output shape: [D, T_frames, total_samples, 3]

    def _render_image_grid(self, coords, sigma_xy, sigma_z, n_sigma_cutoff=3.5):
        """
        Renders volume by looping over depth slices D, while parallelizing 
        the continuous temporal dimension T using vectorized PyTorch operations.
        coords shape: [D, T_frames, total_samples, 3]
        """
        device = self.root_pos.device
        D, H, W = self.D, self.H, self.W
        T_frames = coords.shape[1]
        
        cz, cy, cx = coords[..., 0], coords[..., 1], coords[..., 2]
        out_volume = torch.zeros((T_frames, D, H, W), dtype=torch.float32, device=device)
        
        cutoff_xy = n_sigma_cutoff * sigma_xy

        for d in range(D):
            z_val = float(d) * self.z_scale
            
            # Extract coordinates for depth slice d across ALL time steps T
            cz_d = cz[d] * self.z_scale  # [T_frames, samples]
            cy_d = cy[d]                 # [T_frames, samples]
            cx_d = cx[d]                 # [T_frames, samples]

            # 1. Compute bounding box spanning the filament's trajectory across ALL T
            cy_min, cy_max = cy_d.min().item(), cy_d.max().item()
            cx_min, cx_max = cx_d.min().item(), cx_d.max().item()

            y_min = max(0, int(np.floor(cy_min - cutoff_xy)))
            y_max = min(H, int(np.ceil(cy_max + cutoff_xy)) + 1)
            x_min = max(0, int(np.floor(cx_min - cutoff_xy)))
            x_max = min(W, int(np.ceil(cx_max + cutoff_xy)) + 1)

            if y_min >= y_max or x_min >= x_max:
                continue

            # 2. Local 2D subgrid shapes: [1, y_sub, 1, 1] and [1, 1, x_sub, 1]
            y_grid = torch.arange(y_min, y_max, dtype=torch.float32, device=device).view(1, -1, 1, 1)
            x_grid = torch.arange(x_min, x_max, dtype=torch.float32, device=device).view(1, 1, -1, 1)

            # Reshape coordinates for temporal broadcasting: [T_frames, 1, 1, samples]
            cz_dt = cz_d.unsqueeze(1).unsqueeze(1)
            cy_dt = cy_d.unsqueeze(1).unsqueeze(1)
            cx_dt = cx_d.unsqueeze(1).unsqueeze(1)

            # 3. Vectorized 3D Euclidean distances across T simultaneously
            dist_z_sq = (z_val - cz_dt) ** 2                   # [T_frames, 1, 1, samples]
            dist_y_sq = (y_grid - cy_dt) ** 2                  # [T_frames, y_sub, 1, samples]
            dist_x_sq = (x_grid - cx_dt) ** 2                  # [T_frames, 1, x_sub, samples]

            total_dist_sq = dist_z_sq + dist_y_sq + dist_x_sq # [T_frames, y_sub, x_sub, samples]

            # 4. Minimum distance squared across sample points
            min_dist_sq, _ = torch.min(total_dist_sq, dim=-1) # [T_frames, y_sub, x_sub]

            # 5. Gaussian intensity patch for all T
            intensity_patch = torch.exp(-min_dist_sq / (2 * (sigma_xy ** 2)))

            # 6. Direct tensor slice assignment across all T frames
            out_volume[:, d, y_min:y_max, x_min:x_max] = intensity_patch

        return out_volume

    def forward_async_stack(self, N_frames, dt_frame, sigma_xy, sigma_z, max_angle_current, override_phi_zero=False, n_sigma_cutoff=3.5):
        device = self.root_pos.device
        # --- FIX 4: Expand t_base across D depth slices ---
        t_frame_starts = torch.arange(N_frames, dtype=torch.float32, device=device).unsqueeze(0) * dt_frame
        t_base = t_frame_starts.expand(self.D, -1) # [D, N_frames]
        
        phase_offsets = torch.zeros_like(self.depth_phases).view(-1, 1) if override_phi_zero else self.depth_phases.view(-1, 1)
            
        theta, phi = self._get_bounded_angles(t_base, phase_offsets, max_angle_current)
        coords = self._compute_coordinates(theta, phi)
        return self._render_image_grid(coords, sigma_xy, sigma_z, n_sigma_cutoff=n_sigma_cutoff), coords


def load_and_normalize_tiff(file_path, device):
    img = tifffile.imread(file_path).astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)
    return torch.tensor(img_normalized, dtype=torch.float32, device=device)

def get_current_max_angle(step, total_steps, start_angle=0.05, end_angle=0.8):
    """
    Gradually relaxes the max relative bending angle from a rigid baseline 
    to a flexible polymer over training steps.
    """
    progress = min(1.0, max(0.0, step / total_steps))
    # Linear relaxation (can also use cosine or exponential)
    return start_angle + progress * (end_angle - start_angle)

def get_cosine_max_angle(step, total_steps, start_angle=0.05, end_angle=0.8):
    progress = min(1.0, max(0.0, step / total_steps))
    # Smooth S-curve transition
    cosine_decay = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_angle + cosine_decay * (end_angle - start_angle)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs009", exist_ok=True)
    
    path_async_timelapse = "data/3D_data_async.tif"
    
    print("\n--- 1. Loading 3D + T async datasets ---")
    try:
        async_timelapse_target = load_and_normalize_tiff(path_async_timelapse, device)
    except FileNotFoundError as e:
        print(f"Missing file error: {e}. Falling back to random data for testing.")
        async_timelapse_target = torch.rand(10, 16, 128, 128, device=device)
        
    T_frames, DEPTH, HEIGHT, WIDTH = async_timelapse_target.shape
    print(f"Loaded async timelapse: {async_timelapse_target.shape}")

    # Geometry Constants
    NUM_SEGMENTS = 19
    SEGMENT_LENGTH = 4.0
    DT_SLICE = 1.0
    DT_FRAME = 1.0
    
    # Voxel Size Setup
    VOXEL_XY = 1.0   # 160 nm per pixel in XY
    VOXEL_Z = 1.0    # 450 nm step slicing interval in Z
    
    # Target Gaussian blur spread parameters
    SIGMA_XY_FINAL = 2.0
    SIGMA_Z_FINAL = 2.0 * (VOXEL_XY / VOXEL_Z) # Scaled based on anisotropy
    
    # Regularizers
    #ANGLE_MAX = 0.04
    START_MAX_ANGLE = 0.01
    END_MAX_ANGLE = 0.05
    LAMBDA_RIGIDITY = 0.05
    ROLLING_WEIGHT = 0.01
    HARM_K = 2

    KNOWN_ROOT = [22.0, 5.0, 49] 
    INI_FREQ = 0.2

    BASE_THETA_DEG = 180.0  
    BASE_PHI_DEG = 0.0     

    START_SIGMA = 8.0     
    DECAY_STEPS = 400     

    model = RigidRollingCiliaModel3D(
        num_segments=NUM_SEGMENTS,
        segment_length=SEGMENT_LENGTH,
        depth=DEPTH,
        height=HEIGHT,
        width=WIDTH,
        root_pos=KNOWN_ROOT,
        voxel_size_xy=VOXEL_XY,    
        voxel_size_z=VOXEL_Z,      
        num_harmonics=HARM_K
    ).to(device)
    
    model.omega.data = torch.tensor(INI_FREQ, dtype=torch.float32)
    
    model.base_theta = torch.tensor(BASE_THETA_DEG * (np.pi / 180.0), device=device)
    model.base_phi = torch.tensor(BASE_PHI_DEG * (np.pi / 180.0), device=device)
    
    optimizer = optim.Adam([
        {'params': [model.theta_a0_diff, model.theta_ak_diff, model.theta_bk_diff,
                    model.phi_a0_diff, model.phi_ak_diff, model.phi_bk_diff], 'lr': 0.03},
        {'params': [model.depth_phases], 'lr': 0.05},
        {'params': [model.omega], 'lr': 0.005}
    ])
    TOTAL_STEPS = 2501

    print("\n--- 2. 3D Optimization ---")
    for step in range(TOTAL_STEPS):
        optimizer.zero_grad()

        #current_max_angle = get_cosine_max_angle(
        current_max_angle = get_current_max_angle(        
        step, TOTAL_STEPS * 0.2, START_MAX_ANGLE, END_MAX_ANGLE)
        #step, TOTAL_STEPS, START_MAX_ANGLE, END_MAX_ANGLE)

        if step < DECAY_STEPS:
            factor = (step / DECAY_STEPS)
            current_sigma_xy = START_SIGMA - factor * (START_SIGMA - SIGMA_XY_FINAL)
            current_sigma_z = (START_SIGMA * (VOXEL_XY / VOXEL_Z)) - factor * ((START_SIGMA * (VOXEL_XY / VOXEL_Z)) - SIGMA_Z_FINAL)
        else:
            current_sigma_xy = SIGMA_XY_FINAL
            current_sigma_z = SIGMA_Z_FINAL
        
        est_timelapse, coords_timelapse = model.forward_async_stack(T_frames, DT_FRAME, sigma_xy=current_sigma_xy, sigma_z=current_sigma_z, max_angle_current=current_max_angle, override_phi_zero=False)
        
        loss_timelapse = F.mse_loss(est_timelapse, async_timelapse_target)
        
        loss_rigidity = (torch.mean((model.theta_a0_diff)**2) + torch.mean((model.theta_ak_diff)**2) + torch.mean((model.theta_bk_diff)**2) +
                         torch.mean((model.phi_a0_diff)**2) + torch.mean((model.phi_ak_diff)**2) + torch.mean((model.phi_bk_diff)**2))
        
        total_loss =  loss_timelapse + (LAMBDA_RIGIDITY * loss_rigidity)

        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:03d} | Rigidity: {LAMBDA_RIGIDITY * loss_rigidity.item():.5f} | "
                  f"ANGLEMAX: {current_max_angle:.5f} | "
                  f"Timelapse Fit: {loss_timelapse.item():.5f} | Freq: {model.omega.item():.5f}")
            
            with torch.no_grad():
                est_global, _ = model.forward_async_stack(
                    T_frames, DT_FRAME, sigma_xy=SIGMA_XY_FINAL, sigma_z=SIGMA_Z_FINAL, max_angle_current = current_max_angle, override_phi_zero=True
                )
                
                # Convert PyTorch Tensors to NumPy float32
                timelapse_np = est_timelapse.cpu().numpy().astype(np.float32) # Shape: (T, D, H, W)
                global_np = est_global.cpu().numpy().astype(np.float32)       # Shape: (T, D, H, W)
                
                # 2. Dense Timelapse (4D volume over time: TZYX)
                tifffile.imwrite(
                    f"outputs009/step_{step:03d}_async_stack.tif",
                    timelapse_np,
                    imagej=True,
                    metadata={'axes': 'TZYX'}
                )
                
                # 3. Clean Global Frame (4D volume over time: TZYX)
                tifffile.imwrite(
                    f"outputs009/step_{step:03d}_clean_global.tif",
                    global_np,
                    imagej=True,
                    metadata={'axes': 'TZYX'}
                )                
    print("\nOptimization Finished!")