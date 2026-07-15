import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile
import math

class VectorizedRollingShutterCiliaModel(nn.Module):
    def __init__(self, num_segments, segment_length, img_size, known_root, dt_frame=1.0):
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.H, self.W = img_size
        self.dt = dt_frame
        
        self.register_buffer("root_pos", torch.tensor(known_root, dtype=torch.float32))
        
        # --- TRAINABLE KINEMATICS ---
        self.omega = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.base_amplitude = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.spatial_lag = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.row_phase_offsets = nn.Parameter(torch.zeros(self.H, dtype=torch.float32))

    def forward(self, N_frames, use_phase_shifts=True, sigma=1.5):
        """
        Generates an [N_frames, H, W] tensor using a fully differentiable, 
        vectorized coordinate-field. No loops over rows, no argmin.
        """
        device = self.root_pos.device
        
        # 1. Create a static meshgrid of the image pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=device),
            torch.arange(self.W, dtype=torch.float32, device=device),
            indexing='ij'
        ) # Shapes: [H, W]
        
        # 2. Sample points along the segment lengths [N_samples]
        samples_per_seg = 8
        total_samples = self.num_segments * samples_per_seg
        s_vals = torch.linspace(0, self.num_segments, total_samples, device=device)
        
        frames = []
        for t in range(N_frames):
            current_time = t * self.dt
            
            # --- VECTORIZED ROLLING SHUTTER TIME FIELD ---
            # Every row 'y' experiences a different point in the beating cycle
            # shape: [H, 1]
            if use_phase_shifts:
                row_times = current_time + (self.row_phase_offsets / self.omega)
            else:
                row_times = torch.full((self.H,), current_time, device=device)
            row_times = row_times.view(self.H, 1)
            
            # --- ANALYTICAL BACKBONE GENERATION ---
            # Compute angles for all structural sample points across all row timelines
            # Outputs shape: [H, total_samples]
            s_mesh = s_vals.view(1, -1) # [1, total_samples]
            
            # Physical envelope based on how far up the cilium the sample is
            norm_height = s_mesh / self.num_segments
            scaled_amp = self.base_amplitude * (norm_height ** 1.5)
            
            # Evaluate the beating wave equation directly across the spatial/temporal mesh
            phase = (self.omega * row_times) - (self.spatial_lag * s_mesh)
            angles = scaled_amp * torch.sin(phase)
            
            # Integrate angles to find the continuous (X, Y) positions of the filament
            # Cumulative sum maps the curvature along the chain
            dx = self.L * torch.sin(angles) / samples_per_seg
            dy = self.L * torch.cos(angles) / samples_per_seg
            
            # Cumulative positions along the skeleton: shape [H, total_samples]
            cilia_x = self.root_pos[0] + torch.cumsum(dx, dim=1)
            cilia_y = self.root_pos[1] - torch.cumsum(dy, dim=1)
            
            # --- DIFFERENTIABLE RASTERIZATION VIA DISTANCE FIELDS ---
            # Reshape tensors to calculate cross-distances between pixels and skeleton points
            # Pixel dimensions:   [H, W, 1]
            # Skeleton points:    [H, 1, total_samples]
            px = x_coords.unsqueeze(2)
            py = y_coords.unsqueeze(2)
            cx = cilia_x.unsqueeze(1)
            cy = cilia_y.unsqueeze(1)
            
            # Distance squared from every pixel to every point on the corresponding row's skeleton
            dist_sq = (px - cx)**2 + (py - cy)**2
            
            # Find the minimum distance from each pixel to the backbone
            # min() passes gradients perfectly to the closest active segment point!
            min_dist_sq, _ = torch.min(dist_sq, dim=2)
            
            # Render the continuous Gaussian intensity map
            frame_blurred = torch.exp(-min_dist_sq / (2 * (sigma ** 2)))
            
            # Mask out anything below the root anchor position
            root_mask = (y_coords <= self.root_pos[1]).float()
            frames.append(frame_blurred * root_mask)
            
        return torch.stack(frames, dim=0)

def save_tiff_stack(tensor, filepath):
    numpy_arr = tensor.detach().cpu().numpy().astype(np.float32)
    tifffile.imwrite(filepath, numpy_arr, imagej=True)
    print(f"Successfully saved: '{filepath}' (Shape: {numpy_arr.shape})")


if __name__ == "__main__":
    FRAMES = 200
    IMAGE_SIZE = (64, 64)
    KNOWN_ROOT = [32.0, 62.0]
    NUM_SEGMENTS = 12
    SEGMENT_LENGTH = 4.0
    
    device = torch.device("cpu")
    
    print("--- 1. Generating Rolling Shutter Ground Truth ---")
    gt_model = VectorizedRollingShutterCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        gt_model.omega.copy_(torch.tensor(0.16))
        gt_model.base_amplitude.copy_(torch.tensor(0.6))
        gt_model.spatial_lag.copy_(torch.tensor(0.4))
        
        np.random.seed(42)
        true_row_delays = (torch.tensor(np.random.rand(IMAGE_SIZE[0]), dtype=torch.float32) * 2 * math.pi) - math.pi
        gt_model.row_phase_offsets.copy_(true_row_delays)
        
        gt_no_shifts = gt_model(N_frames=FRAMES, use_phase_shifts=False)
        save_tiff_stack(gt_no_shifts, "vector_gt_NO_phase_shifts.tif")
        
        gt_with_shifts = gt_model(N_frames=FRAMES, use_phase_shifts=True)
        save_tiff_stack(gt_with_shifts, "vector_gt_with_phase_shifts.tif")
        
    print("\n--- 2. Convergence Loop via Vectorized Autograd ---")
    fit_model = VectorizedRollingShutterCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        fit_model.omega.copy_(torch.tensor(0.08))
        fit_model.base_amplitude.copy_(torch.tensor(0.2))
        fit_model.spatial_lag.copy_(torch.tensor(0.1))
        
    optimizer = optim.Adam(fit_model.parameters(), lr=0.04)
    
    for step in range(301):
        optimizer.zero_grad()
        
        estimated_stack = fit_model(N_frames=FRAMES, use_phase_shifts=True)
        loss = F.mse_loss(estimated_stack, gt_with_shifts)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            with torch.no_grad():
                phase_diff = fit_model.row_phase_offsets - true_row_delays
                mae = torch.mean(torch.abs(torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)))).item()
            print(f"Step {step:02d} | MSE Loss: {loss.item():.6f} | Omega: {fit_model.omega.item():.4f} | Phase MAE: {mae:.4f} rad")

    print("\n--- 3. Exporting Vectorized Fits ---")
    with torch.no_grad():
        fit_no_shifts = fit_model(N_frames=FRAMES, use_phase_shifts=False)
        save_tiff_stack(fit_no_shifts, "vector_fit_NO_phase_shifts.tif")
        
    print("\nDone! This architecture scales smoothly to modeling multiple concurrent cilia.")