import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import math

class DifferentiableChainFitter(nn.Module):
    def __init__(self, num_segments, segment_length, img_size, known_start_pos, sigma=3.0, max_bend_deg=45.0):
        super().__init__()
        self.num_segments = num_segments
        self.L = segment_length
        self.H, self.W = img_size
        self.sigma = sigma
        self.max_bend_rad = math.radians(max_bend_deg)
        
        # 1. FIXED BUFFER: Root coordinates do not change
        self.register_buffer("start_pos", torch.tensor(known_start_pos, dtype=torch.float32))
        
        # 2. OPTIMIZABLE PARAMETERS: Initial orientation angle + internal bending chain
        self.base_angle = nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) # Starts facing right (0 rad)
        self.relative_angles = nn.Parameter(torch.zeros(num_segments - 1, dtype=torch.float32))
        
    def get_constrained_angles(self):
        return torch.tanh(self.relative_angles) * self.max_bend_rad

    def get_joint_coordinates(self):
        constrained_angles = self.get_constrained_angles()
        cum_angles = torch.cumsum(constrained_angles, dim=0)
        
        # Both base_angle and relative_angles will flow gradients simultaneously
        abs_angles = torch.cat([self.base_angle.unsqueeze(0), self.base_angle + cum_angles])
        
        dx = self.L * torch.cos(abs_angles)
        dy = self.L * torch.sin(abs_angles)
        
        displacements = torch.stack([dx, dy], dim=1)
        return torch.cat([self.start_pos.unsqueeze(0), self.start_pos + torch.cumsum(displacements, dim=0)], dim=0)

    def forward(self):
        joints = self.get_joint_coordinates()
        samples_per_segment = max(5, int(self.L * 2)) 
        t = torch.linspace(0, 1, samples_per_segment, device=joints.device).view(-1, 1)
        
        dense_points = []
        for i in range(self.num_segments):
            dense_points.append(joints[i] + t * (joints[i+1] - joints[i]))
        points = torch.cat(dense_points, dim=0)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=points.device),
            torch.arange(self.W, dtype=torch.float32, device=points.device),
            indexing='ij'
        )
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(2) 
        dist_sq = torch.sum((grid - points.view(1, 1, -1, 2)) ** 2, dim=-1)
        
        synthetic_img = torch.sum(torch.exp(-dist_sq / (2 * (self.sigma ** 2))), dim=-1)
        if synthetic_img.max() > 0:
            synthetic_img = synthetic_img / synthetic_img.max()
        return synthetic_img

def save_float32_tiff(tensor_img, filename):
    img_np = tensor_img.detach().cpu().numpy().astype(np.float32)
    tifffile.imwrite(filename, img_np)

def visualize_and_save_curve(target_img_tensor, model, filename="curve_fit_visualization.png", show=True):
    bg_img = target_img_tensor.detach().cpu().numpy()
    with torch.no_grad():
        joints = model.get_joint_coordinates().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(bg_img, cmap='gray', origin='upper')
    plt.plot(joints[:, 0], joints[:, 1], color='red', linestyle='-', linewidth=2.5, label='Fixed Origin Chain')
    plt.scatter(joints[:, 0], joints[:, 1], color='cyan', zorder=5)
    plt.title(os.path.basename(filename))
    plt.legend(loc='upper right')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    if show: plt.show()
    else: plt.close(fig)

# --- Execution Entry Point ---
if __name__ == "__main__":
    INPUT_FILE_PATH = "C:\\Users\\ekatrukha\\Desktop\\fit_cilia\\MAX_GT_all-1.tif"   
    SNAPSHOT_DIR = "optimization_history"
    SNAPSHOT_INTERVAL = 25
    
    NUM_SEGMENTS = 40
    SEGMENT_LENGTH = 2.0
    SIGMA = 2.0             # Footprint width of the blur profile
    MAX_BEND_DEG = 15     # HARD BOUND: Max angle change allowed at any single joint
    RIGIDITY_WEIGHT = 0.01  # SOFT CONSTRAINT: Penalty weight for sharp kinks (stiffness)

    
    # 🔴 USER CONFIGURATION: ENTER YOUR EXACT ROOT POSITION HERE
    KNOWN_START_XY = [22.0, 50.0]  # [X, Y] Anchor coordinate
    
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"[Error] The file '{INPUT_FILE_PATH}' was not found.")

    # 1. Load data
    raw_img = tifffile.imread(INPUT_FILE_PATH)
    if raw_img.ndim == 3: raw_img = raw_img[:, :, 0]
    raw_img = raw_img.astype(np.float32)
    img_min, img_max = raw_img.min(), raw_img.max()
    if img_max > img_min: raw_img = (raw_img - img_min) / (img_max - img_min)
    target_image = torch.from_numpy(raw_img)
    IMAGE_SIZE = target_image.shape

    # 2. Initialize Model
    model = DifferentiableChainFitter(
        num_segments=NUM_SEGMENTS, 
        segment_length=SEGMENT_LENGTH, 
        img_size=IMAGE_SIZE, 
        known_start_pos=KNOWN_START_XY, 
        sigma=SIGMA, 
        max_bend_deg=MAX_BEND_DEG
    )
    
    # Optimization automatically groups base_angle + relative_angles together
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # 3. Fitting
    print(f"\nOptimizing base heading angle and internal bends from anchored point {KNOWN_START_XY}...")
    for step in range(1001):
        optimizer.zero_grad()
        rendered = model()
        
        image_loss = F.mse_loss(rendered, target_image)
        actual_relative_angles = model.get_constrained_angles()
        stiffness_penalty = torch.sum((actual_relative_angles[1:] - actual_relative_angles[:-1]) ** 2)
        
        total_loss = image_loss + (RIGIDITY_WEIGHT * stiffness_penalty)
        total_loss.backward()
        optimizer.step()
        
        if step % SNAPSHOT_INTERVAL == 0:
            current_base_deg = math.degrees(model.base_angle.item()) % 360
            print(f"Step {step:03d} | Image Loss: {image_loss.item():.5f} | Base Heading: {current_base_deg:.1f}°")
            visualize_and_save_curve(target_image, model, filename=os.path.join(SNAPSHOT_DIR, f"step_{step:03d}.png"), show=False)
            
    print("\nOptimization Complete!")
    
    # Final Output Processing
    final_base_angle = math.degrees(model.base_angle.item()) % 360
    final_relative_angles = np.degrees(model.get_constrained_angles().detach().cpu().numpy())
    
    print(f"\nExtracted Base Heading Orientation: {final_base_angle:.2f} degrees")
    print("Extracted Relative Chain Angles (Degrees):\n", final_relative_angles)
    
    visualize_and_save_curve(target_image, model, filename="final_fixed_xy_curve.png", show=True)