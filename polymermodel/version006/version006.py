import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile

class RollingCiliaModel(nn.Module):
    def __init__(self, num_segments, segment_length, width, height, num_harmonics=3):
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.W = width
        self.H = height
        self.K = num_harmonics
        
        known_root = [width / 2.0, height - 5.0]
        self.register_buffer("root_pos", torch.tensor(known_root, dtype=torch.float32))
        
        # Learnable frequency (angular frequency: rad per time unit)
        self.omega = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        
        # Timing constants (t0 and t1 are base times)
        self.t0_rolling = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t1_dense = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # Learnable phase offsets expressed directly in RADIANS for each row y.
        # Initialize with a small standard deviation to break symmetry.
        self.row_phases = nn.Parameter(torch.zeros(height, dtype=torch.float32)) 
        
        # Kinematic parameters
        self.a0 = nn.Parameter(torch.zeros(num_segments, dtype=torch.float32))
        self.ak = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        self.bk = nn.Parameter(torch.zeros(self.K, num_segments, dtype=torch.float32))
        
        with torch.no_grad():
            self.ak.normal_(0.0, 0.02)
            self.bk.normal_(0.0, 0.02)
            self.row_phases.normal_(0.0, 0.2) # ~11 degrees std deviation to initialize

    def _compute_angles_for_phase_grid(self, t_base, phase_offsets):
        """
        Computes cilia segment angles using a base time matrix and row-wise phase offsets.
        
        t_base shape: [H, T] (represents the global clock time of measurement)
        phase_offsets shape: [H, 1] or [H, T] (represents local phase delay in RADIANS)
        
        Output shape: [H, T, Total_Samples]
        """
        device = self.root_pos.device
        samples_per_seg = 12
        total_samples = self.num_segments * samples_per_seg
        
        s_mesh = torch.linspace(0, self.num_segments - 1, total_samples, device=device)
        idx_low = torch.clamp(torch.floor(s_mesh).long(), 0, self.num_segments - 1)
        idx_high = torch.clamp(idx_low + 1, 0, self.num_segments - 1)
        weight_high = s_mesh - idx_low.float()
        weight_low = 1.0 - weight_high
        
        # Spatial segment interpolation
        mesh_a0 = (self.a0[idx_low] * weight_low + self.a0[idx_high] * weight_high).view(1, 1, -1)
        mesh_ak = (self.ak[:, idx_low] * weight_low + self.ak[:, idx_high] * weight_high)
        mesh_bk = (self.bk[:, idx_low] * weight_low + self.bk[:, idx_high] * weight_high)
        
        root_clamp = torch.clamp(torch.linspace(0.0, 2.0, total_samples, device=device), 0.0, 1.0).view(1, 1, -1)
        
        # Core physical argument scaling: theta(t) = w * t + phi
        # Shape: [H, T, 1]
        phase_grid = (self.omega * t_base + phase_offsets).unsqueeze(-1)
        
        angles = mesh_a0.expand(self.H, t_base.shape[1], -1).clone()
        
        for k in range(1, self.K + 1):
            cos_kt = torch.cos(k * phase_grid)
            sin_kt = torch.sin(k * phase_grid)
            
            ak_mesh = mesh_ak[k-1].view(1, 1, -1)
            bk_mesh = mesh_bk[k-1].view(1, 1, -1)
            
            angles += (cos_kt * ak_mesh) + (sin_kt * bk_mesh)
            
        return angles * root_clamp

    def _render_image_grid(self, angles, sigma=1.5):
        """
        Converts angular kinematics [H, T, S] into [T, H, W] frames
        """
        device = self.root_pos.device
        samples_per_seg = 12
        
        dx = self.L * torch.sin(angles) / samples_per_seg
        dy = self.L * torch.cos(angles) / samples_per_seg
        
        cx = self.root_pos[0] + torch.cumsum(dx, dim=2)
        cy = self.root_pos[1] - torch.cumsum(dy, dim=2)
        
        cx_5d = cx.unsqueeze(2)
        cy_5d = cy.unsqueeze(2)
        
        y_grid = torch.arange(self.H, dtype=torch.float32, device=device).view(self.H, 1, 1, 1)
        x_grid = torch.arange(self.W, dtype=torch.float32, device=device).view(1, 1, self.W, 1)
        
        dist_sq = (x_grid - cx_5d)**2 + (y_grid - cy_5d)**2
        min_dist_sq, _ = torch.min(dist_sq, dim=3)
        
        intensity = torch.exp(-min_dist_sq / (2 * (sigma ** 2)))
        return intensity.permute(1, 0, 2)

    def forward_dense_timelapse(self, N_frames, dt_frame, sigma=1.5, override_phi_zero=False):
        """
        Generates N_frames of XY images with frame-start step dynamics.
        """
        device = self.root_pos.device
        
        # Base global frames timeline: shape [1, T]
        t_frame_starts = self.t1_dense + torch.arange(N_frames, dtype=torch.float32, device=device).unsqueeze(0) * dt_frame
        t_base = t_frame_starts.expand(self.H, -1) # Broadcaster shape [H, T]
        
        if override_phi_zero:
            # Ideal global shutter: all rows read with phase phase offset = 0
            phase_offsets = torch.zeros_like(self.row_phases).view(-1, 1)
        else:
            # Asynchronous local shutter delays: [H, 1]
            phase_offsets = self.row_phases.view(-1, 1)
            
        angles = self._compute_angles_for_phase_grid(t_base, phase_offsets)
        return self._render_image_grid(angles, sigma), angles

    def forward_initial_rolling_shutter(self, dt_slice, sigma=1.5):
        """
        Generates a rolling shutter 2D snapshot where row y is measured at global clock t0 + y * dt_slice.
        Because this is raw physical rolling shutter, the delay is linearly dependent on time 
        (and thus is implicitly scaled by omega through the calculation).
        """
        device = self.root_pos.device
        
        # Linear sweep times: shape [H, 1]
        t_base = self.t0_rolling + (torch.arange(self.H, dtype=torch.float32, device=device).unsqueeze(1) * dt_slice)
        
        # No extra phase shift variance during rolling snapshot generation
        phase_offsets = torch.zeros_like(self.row_phases).view(-1, 1)
        
        angles = self._compute_angles_for_phase_grid(t_base, phase_offsets)
        return self._render_image_grid(angles, sigma).squeeze(0)


if __name__ == "__main__":
    HEIGHT, WIDTH = 64, 64 
    NUM_SEGMENTS = 15
    SEGMENT_LENGTH = 2.0
    DT_SLICE = 0.05
    DT_FRAME = 1.0
    TIMEPOINTS = 100
    LAMBDA_RIGIDITY = 0.01
    
    device = torch.device("cpu")
    print(f"Target Execution Device: {device}")
    
    # =============================================================
    # 1. Initialize True Model (Generates Ground Truth Targets)
    # =============================================================
    print("--- 1. Simulating and Saving Ground Truth Targets ---")
    true_model = RollingCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, WIDTH, HEIGHT).to(device)
    
    with torch.no_grad():
        true_model.omega.copy_(torch.tensor(0.135))
        true_model.t0_rolling.copy_(torch.tensor(0.5))
        true_model.t1_dense.copy_(torch.tensor(0.0))
        
        # Phase offsets generated directly in radians.
        # Let's use a wide spread: [-1.2, 1.2] radians (~ +/- 70 degrees of phase variance)
        np.random.seed(42)
        mock_phases_rad = np.random.uniform(-1.2, 1.2, HEIGHT).astype(np.float32)
        true_model.row_phases.copy_(torch.tensor(mock_phases_rad))
        
        # Setup structural kinematics
        true_model.a0.fill_(-0.2)
        true_model.ak[0].fill_(0.5)
        true_model.bk[0].fill_(0.3)

    with torch.no_grad():
        # Async row-phase shifted 2D+T dense timelapse
        dense_4d_gt, _ = true_model.forward_dense_timelapse(TIMEPOINTS, DT_FRAME, override_phi_zero=False)
        print(f"Generated Dense Async Movie Shape: {dense_4d_gt.shape}")
        
        # Raw single-frame rolling shutter 2D snapshot
        rolling_gt = true_model.forward_initial_rolling_shutter(DT_SLICE)
        print(f"Generated Rolling Shutter Snapshot Shape: {rolling_gt.shape}")
        
        # Perfect phase-aligned Global Shutter reference movie
        clean_movie_gt, _ = true_model.forward_dense_timelapse(TIMEPOINTS, DT_FRAME, override_phi_zero=True)

    # Save outputs
    tifffile.imwrite("gt_raw_async_timelapse.tif", dense_4d_gt.cpu().numpy().astype(np.float32), imagej=True, metadata={'axes': 'TYX'})
    tifffile.imwrite("gt_rolling_shutter_snapshot.tif", rolling_gt.cpu().numpy().astype(np.float32), imagej=True, metadata={'axes': 'YX'})
    tifffile.imwrite("gt_clean_global_shutter_beating.tif", clean_movie_gt.cpu().numpy().astype(np.float32), imagej=True, metadata={'axes': 'TYX'})
    print("Saved target files to disk successfully.\n")

    # =============================================================
    # 2. Setup Trainable Model (Optimizing parameters to match target)
    # =============================================================
    print("--- 2. Setting up Multi-Phase Optimization Solver ---")
    model = RollingCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, WIDTH, HEIGHT).to(device)
    
    # Optimizer with tuned learning rates for phase recovery
    optimizer = optim.Adam([
        {'params': [model.a0, model.ak, model.bk], 'lr': 0.02},
        {'params': [model.row_phases], 'lr': 0.05}, # Direct updates in radian space
        {'params': [model.t0_rolling, model.t1_dense], 'lr': 0.01},
        {'params': [model.omega], 'lr': 0.005}
    ])
    
    # =============================================================
    # 3. Fitting Loop
    # =============================================================
    print("\n--- 3. Running Unified Row-Phase Optimization Loop ---")
    for step in range(321):
        optimizer.zero_grad()
        
        est_dense, angles = model.forward_dense_timelapse(TIMEPOINTS, DT_FRAME, override_phi_zero=False)
        loss_dense = F.mse_loss(est_dense, dense_4d_gt)
        
        est_rolling = model.forward_initial_rolling_shutter(DT_SLICE)
        loss_rolling = F.mse_loss(est_rolling, rolling_gt)
        
        dtheta_ds = angles[:, :, 1:] - angles[:, :, :-1]
        loss_rigidity = torch.mean(dtheta_ds ** 2)
        
        total_loss = loss_dense + (1.0 * loss_rolling) + (LAMBDA_RIGIDITY * loss_rigidity)
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Movie Loss: {loss_dense.item():.6f} | "
                  f"Snapshot Loss: {loss_rolling.item():.6f} | "
                  f"Est Omega: {model.omega.item():.4f}")

    # =============================================================
    # 4. Save Final Reconstruction Outputs
    # =============================================================
    print("\n--- 4. Exporting Reconstruction Results ---")
    with torch.no_grad():
        reconstructed_global_shutter, _ = model.forward_dense_timelapse(TIMEPOINTS, DT_FRAME, override_phi_zero=True)
        tifffile.imwrite(
            "output_reconstructed_global_shutter.tif", 
            reconstructed_global_shutter.cpu().numpy().astype(np.float32), 
            imagej=True, 
            metadata={'axes': 'TYX'}
        )
    print("Reconstruction complete! Check output_reconstructed_global_shutter.tif in your workspace.")