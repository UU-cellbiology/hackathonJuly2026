import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# Assuming this helper function is already in your script
def save_float32_tiff(tensor, filename):
    """
    Saves a float32 tensor to disk. Handles both single 2D frames [H, W] 
    and multi-page batches/volumes [K, H, W].
    """
    import tifffile
    # Ensure it's a decoupled numpy array on host memory
    data = tensor.detach().cpu().numpy().astype(np.float32)
    tifffile.imwrite(filename, data, photometric='minisblack')
    print(f"Saved volume stack successfully: '{filename}' (Shape: {data.shape})")

class MultiAcquisitionCiliaFitter(nn.Module):
    def __init__(self, num_acquisitions, num_segments, segment_length, img_size, known_start_pos, time_delay_per_column=1.0):
        super().__init__()
        self.K = num_acquisitions
        self.num_segments = num_segments
        self.L = segment_length
        self.H, self.W = img_size
        self.dt = time_delay_per_column
        
        self.register_buffer("start_pos", torch.tensor(known_start_pos, dtype=torch.float32))
        
        # SHARED PARAMETERS OVER ALL ACQUISITIONS
        self.omega = nn.Parameter(torch.tensor(0.04, dtype=torch.float32))
        self.base_a = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.base_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.relative_a = nn.Parameter(torch.zeros(num_segments - 1, dtype=torch.float32))
        self.relative_b = nn.Parameter(torch.zeros(num_segments - 1, dtype=torch.float32))
        
        # PER-ACQUISITION CHRONO INITIAL PHASES [K]
        self.phases = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))

    def get_joint_coordinates(self):
        columns = torch.arange(self.W, dtype=torch.float32, device=self.start_pos.device)
        
        # 🟢 FIX: Calculate independent global timestamps per column for each frame
        # Each frame k starts at its unique absolute phase time, then advances column-by-column
        # Shape: [K, W]
        times = self.phases.unsqueeze(1) + (columns.unsqueeze(0) * self.dt)
        
        # Base angles now sample distinct temporal cycles per frame: [K, W]
        base_angles = self.base_a * torch.sin(self.omega * times) + self.base_b * torch.cos(self.omega * times)
        
        sin_part = self.relative_a.view(-1, 1, 1) * torch.sin(self.omega * times).unsqueeze(0)
        cos_part = self.relative_b.view(-1, 1, 1) * torch.cos(self.omega * times).unsqueeze(0)
        rel_angles = torch.tanh(sin_part + cos_part) * math.radians(45.0)
        
        cum_rel_angles = torch.cumsum(rel_angles, dim=0)
        all_abs_angles = torch.cat([base_angles.unsqueeze(0), base_angles.unsqueeze(0) + cum_rel_angles], dim=0)
        
        dx = self.L * torch.cos(all_abs_angles)
        dy = self.L * torch.sin(all_abs_angles)
        
        # Keep roots structurally locked to spatial position for all frames
        joint_x = [torch.full((self.K, self.W), self.start_pos[0], device=self.start_pos.device)]
        joint_y = [torch.full((self.K, self.W), self.start_pos[1], device=self.start_pos.device)]
        
        for i in range(self.num_segments):
            joint_x.append(joint_x[-1] + dx[i])
            joint_y.append(joint_y[-1] + dy[i])
            
        return torch.stack(joint_x, dim=1), torch.stack(joint_y, dim=1)

    def forward(self, sigma=2.0):
        joint_x, joint_y = self.get_joint_coordinates()
        samples_per_segment = max(5, int(self.L * 2))
        t_samples = torch.linspace(0, 1, samples_per_segment, device=self.start_pos.device).view(-1, 1, 1, 1)
        
        dense_x, dense_y = [], []
        for i in range(self.num_segments):
            x_start = joint_x[:, i, :].unsqueeze(0)
            x_end = joint_x[:, i+1, :].unsqueeze(0)
            y_start = joint_y[:, i, :].unsqueeze(0)
            y_end = joint_y[:, i+1, :].unsqueeze(0)
            
            dense_x.append(x_start + t_samples * (x_end - x_start))
            dense_y.append(y_start + t_samples * (y_end - y_start))
            
        pts_x = torch.cat(dense_x, dim=0).view(-1, self.K, self.W)
        pts_y = torch.cat(dense_y, dim=0).view(-1, self.K, self.W)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=self.start_pos.device),
            torch.arange(self.W, dtype=torch.float32, device=self.start_pos.device),
            indexing='ij'
        )
        
        y_grid = y_grid.view(1, 1, self.H, self.W)
        x_grid = x_grid.view(1, 1, self.H, self.W)
        pts_y = pts_y.unsqueeze(2)
        pts_x = pts_x.unsqueeze(2)
        
        dist_sq = (y_grid - pts_y)**2 + (x_grid - pts_x)**2
        gaussian_kernels = torch.exp(-dist_sq / (2 * (sigma ** 2)))
        
        synthetic_volumes = torch.sum(gaussian_kernels, dim=0)
        max_vals = synthetic_volumes.view(self.K, -1).max(dim=1)[0].view(self.K, 1, 1)
        max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
        
        return synthetic_volumes / max_vals

if __name__ == "__main__":
    NUM_ACQUISITIONS = 100
    IMAGE_SIZE = (100, 180)
    KNOWN_ROOT = [10.0, 50.0]
    COL_DELAY = 0.5
    NUM_SEGMENTS = 10
    SEGMENT_LENGTH = 4.0
    
    device = torch.device("cpu")
    print(f"Initializing Multi-Acquisition Core. Total frames: {NUM_ACQUISITIONS}")

    # ==================================================
    # 1. GROUND TRUTH CREATION (With Random Initial Phases)
    # ==================================================
    gt_generator = MultiAcquisitionCiliaFitter(
        NUM_ACQUISITIONS, NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT, COL_DELAY
    ).to(device)
    
    with torch.no_grad():
        gt_generator.omega.copy_(torch.tensor(0.6))
        gt_generator.base_a.copy_(torch.tensor(0.5))
        gt_generator.base_b.copy_(torch.tensor(-0.1))
        gt_generator.relative_a.copy_(torch.linspace(0.05, 0.25, NUM_SEGMENTS-1))
        gt_generator.relative_b.copy_(torch.linspace(-0.15, 0.05, NUM_SEGMENTS-1))
        
        true_phases = (torch.rand(NUM_ACQUISITIONS) * 2 * math.pi) - math.pi
        gt_generator.phases.copy_(true_phases)
        clean_batch = gt_generator(sigma=1.6)

    noise = torch.randn_like(clean_batch) * 0.05
    ground_truth_dataset = torch.clamp(clean_batch + noise, 0.0, 1.0).detach()
    
    # 🟢 NEW: Save the entire ground truth volume stack directly to disk
    save_float32_tiff(ground_truth_dataset, "ground_truth_dataset.tif")

    # ==================================================
    # 2. RUN EMBEDDED OPTIMIZER ESTIMATION
    # ==================================================
    fit_model = MultiAcquisitionCiliaFitter(
        NUM_ACQUISITIONS, NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT, COL_DELAY
    ).to(device)
    
    optimizer = optim.Adam(fit_model.parameters(), lr=0.05)
    timeline_snapshots = []
    
    print("\nStarting optimization across shared features + unique phases...")
    for step in range(2000):
        optimizer.zero_grad()
        rendered_batch = fit_model(sigma=1.6)
        loss = F.mse_loss(rendered_batch, ground_truth_dataset)
        loss.backward()
        optimizer.step()
        
        # 🟢 NEW: Track intermediate progress every 20 steps
        if step % 20 == 0:
            with torch.no_grad():
                phase_diff = (fit_model.phases - true_phases)
                phase_mae = torch.mean(torch.abs(torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)))).item()
                
                # Clone Frame 0 at this exact moment in training
                frame_snapshot = rendered_batch[0].clone().detach()
                timeline_snapshots.append(frame_snapshot)
                
            print(f"Step {step:03d} | Batch MSE Loss: {loss.item():.6f} | Phase MAE (rad): {phase_mae:.4f}")
            
        if step < 200:
            del rendered_batch
        del loss

    # ==================================================
    # 3. SAVE METRIC RESULTS
    # ==================================================
    # 🟢 NEW: Save the finalized optimization results stack to disk
    # This matches the shape [100, 100, 180] of your ground truth data file.
    print("\nTraining complete. Exporting fit model results stack...")
    save_float32_tiff(rendered_batch, "fit_results_dataset.tif")
    
    # 🟢 NEW: Stack and save the intermediate training history to disk
    # This creates an [11, H, W] volume tracking steps 0, 20, 40 ... 200
    timeline_stack = torch.stack(timeline_snapshots, dim=0)
    save_float32_tiff(timeline_stack, "optimization_timeline.tif")
    #print(f"Saved training history snapshot stack to 'optimization_timeline.tif'")
    
    # Generate metric verification plots
    final_true_p = true_phases.numpy()
    final_est_p = fit_model.phases.detach().numpy()
    offset = np.median(final_est_p - final_true_p)
    aligned_est_p = final_est_p - offset
    
    plt.figure(figsize=(10, 4))
    plt.scatter(range(NUM_ACQUISITIONS), final_true_p, color='black', label='True Phase', alpha=0.7)
    plt.scatter(range(NUM_ACQUISITIONS), aligned_est_p, color='crimson', marker='x', label='Recovered Phase')
    plt.xlabel("Acquisition Index")
    plt.ylabel("Phase (radians)")
    plt.title("Phase Parameter Recovery Metrics")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig("phase_recovery_verification.png", bbox_inches='tight', dpi=120)
    plt.close()
    print("Performance verification metrics saved safely to disk.")