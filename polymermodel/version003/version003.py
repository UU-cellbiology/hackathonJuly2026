import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile
import math

class StructuralAsynchronousCiliaModel(nn.Module):
    def __init__(self, num_segments, segment_length, img_size, known_root, dt_frame=1.0):
        super().__init__()
        self.num_segments = num_segments
        self.L = float(segment_length)
        self.H, self.W = img_size
        self.dt = dt_frame
        
        self.register_buffer("root_pos", torch.tensor(known_root, dtype=torch.float32))
        
        # --- PARAMETERS TO OPTIMIZE / ESTIMATE ---
        self.omega = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.base_amplitude = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.spatial_lag = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        
        # Row-by-row phase offsets
        self.row_phase_offsets = nn.Parameter(torch.zeros(self.H, dtype=torch.float32))

    def forward(self, N_frames, use_phase_shifts=True, sigma=1.5):
        """
        Generates/Fits an [N_frames, H, W] timelapse with corrected physics 
        and smooth tip-foreshortening falloff.
        """
        device = self.root_pos.device
        x_indices = torch.arange(self.W, dtype=torch.float32, device=device) # Shape: [W]
        root_x, root_y = self.root_pos[0], self.root_pos[1]
        
        # High resolution sampling along each segment for smooth curves
        samples_per_seg = 10
        t_samples = torch.linspace(0, 1, samples_per_seg, device=device).view(-1, 1)

        frames = []
        for t in range(N_frames):
            current_time = t * self.dt
            rows = []
            
            for y in range(self.H):
                # The cilium only exists above the root anchor
                if y > root_y:
                    rows.append(torch.zeros(self.W, device=device))
                    continue
                
                # --- CORRECTED FORWARD KINEMATICS ---
                # Build the complete physical skeleton coordinate tree
                jx = [root_x]
                jy = [root_y]
                accumulated_angle = torch.tensor(0.0, device=device)
                
                for seg in range(self.num_segments):
                    # Phase combines time sequence, spatial delay, and optional row delay jitter
                    phase = (self.omega * current_time) - (self.spatial_lag * seg)
                    if use_phase_shifts:
                        phase = phase + self.row_phase_offsets[y]
                    
                    # 1. FIX: Envelope scale depends on the SEGMENT index, not the row height
                    norm_seg_height = (seg + 1) / self.num_segments
                    scaled_amp = self.base_amplitude * (norm_seg_height ** 1.5)
                    
                    seg_angle = scaled_amp * torch.sin(phase)
                    accumulated_angle = accumulated_angle + seg_angle
                    
                    jx.append(jx[-1] + self.L * torch.sin(accumulated_angle))
                    jy.append(jy[-1] - self.L * torch.cos(accumulated_angle))
                
                # Dense interpolation of points along the skeleton
                dense_x = []
                dense_y = []
                for i in range(self.num_segments):
                    dense_x.append(jx[i] + t_samples * (jx[i+1] - jx[i]))
                    dense_y.append(jy[i] + t_samples * (jy[i+1] - jy[i]))
                
                pts_x = torch.cat(dense_x, dim=0)
                pts_y = torch.cat(dense_y, dim=0)
                
                # Find the point on our physical skeleton closest to this row height 'y'
                dist_to_row = torch.abs(pts_y - y)
                closest_idx = torch.argmin(dist_to_row)
                min_dist_y = dist_to_row[closest_idx]
                intersect_x = pts_x[closest_idx]
                
                # 2. FIX: Vertical Distance Falloff (Smooths out foreshortened tips)
                # If the physical skeleton is far from row y, decay the intensity to 0
                vertical_decay = torch.exp(- (min_dist_y ** 2) / (2 * (0.8 ** 2)))
                
                # Render clean horizontal line profile
                dist_sq = (x_indices - intersect_x) ** 2
                row_profile = torch.exp(-dist_sq / (2 * (sigma ** 2))) * vertical_decay
                rows.append(row_profile)
                
            frames.append(torch.stack(rows, dim=0))
            
        return torch.stack(frames, dim=0)

def save_tiff_stack(tensor, filepath):
    numpy_arr = tensor.detach().cpu().numpy().astype(np.float32)
    tifffile.imwrite(filepath, numpy_arr, imagej=True)
    print(f"Successfully saved: '{filepath}' (Shape: {numpy_arr.shape})")


if __name__ == "__main__":
    FRAMES = 100
    IMAGE_SIZE = (64, 64)
    KNOWN_ROOT = [32.0, 62.0]
    NUM_SEGMENTS = 7
    SEGMENT_LENGTH = 8.0
    
    device = torch.device("cpu")
    
    # ==========================================
    # 1. GENERATE GROUND TRUTH TIMELAPSES
    # ==========================================
    print("--- 1. Generating Ground Truth Timelapses ---")
    gt_generator = StructuralAsynchronousCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        gt_generator.omega.copy_(torch.tensor(0.16))
        gt_generator.base_amplitude.copy_(torch.tensor(0.55))
        gt_generator.spatial_lag.copy_(torch.tensor(0.35))
        
        np.random.seed(777)
        true_row_delays = (torch.tensor(np.random.rand(IMAGE_SIZE[0]), dtype=torch.float32) * 2 * math.pi) - math.pi
        gt_generator.row_phase_offsets.copy_(true_row_delays)
        
        # A. Clean Synchronized Ground Truth
        gt_no_shifts = gt_generator(N_frames=FRAMES, use_phase_shifts=False)
        save_tiff_stack(gt_no_shifts, "gt_NO_phase_shifts.tif")
        
        # B. Asynchronous Jittery Ground Truth
        gt_with_shifts = gt_generator(N_frames=FRAMES, use_phase_shifts=True)
        save_tiff_stack(gt_with_shifts, "gt_with_phase_shifts.tif")
    
    # ==========================================
    # 2. RUN GRADIENT DESCENT OPTIMIZATION
    # ==========================================
    print("\n--- 2. Optimization and Structural Recovery Loop ---")
    fit_model = StructuralAsynchronousCiliaModel(NUM_SEGMENTS, SEGMENT_LENGTH, IMAGE_SIZE, KNOWN_ROOT).to(device)
    
    with torch.no_grad():
        fit_model.omega.copy_(torch.tensor(0.06))
        fit_model.base_amplitude.copy_(torch.tensor(0.15))
        fit_model.spatial_lag.copy_(torch.tensor(0.10))
    
    optimizer = optim.Adam(fit_model.parameters(), lr=0.03)
    
    for step in range(61):
        optimizer.zero_grad()
        
        estimated_stack = fit_model(N_frames=FRAMES, use_phase_shifts=True, sigma=1.5)
        loss = F.mse_loss(estimated_stack, gt_with_shifts)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            with torch.no_grad():
                phase_diff = fit_model.row_phase_offsets - true_row_delays
                mae = torch.mean(torch.abs(torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)))).item()
            print(f"Step {step:02d} | Loss: {loss.item():.6f} | Est ω: {fit_model.omega.item():.4f} | Est Amp: {fit_model.base_amplitude.item():.4f} | Phase MAE: {mae:.4f} rad")

    # ==========================================
    # 3. EXPORT RECOVERED FITTED MODELS
    # ==========================================
    print("\n--- 3. Exporting Fitted Recovery Movies ---")
    with torch.no_grad():
        # A. Reconstructed Jittery Dataset
        fit_with_shifts = fit_model(N_frames=FRAMES, use_phase_shifts=True)
        save_tiff_stack(fit_with_shifts, "fit_with_phase_shifts.tif")
        
        # B. Recovered Clean Synchronous Dataset
        fit_no_shifts = fit_model(N_frames=FRAMES, use_phase_shifts=False)
        save_tiff_stack(fit_no_shifts, "fit_NO_phase_shifts.tif")
        
    print("\nCompleted! Drop 'gt_NO_phase_shifts.tif' and 'fit_NO_phase_shifts.tif' into Fiji to see the flawless recovery.")