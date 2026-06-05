import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import os
from models.pinn_network import ThermalSurrogateModel

# 1. HARDWARE DETECTION: Strictly locked to CPU for local testing
device = torch.device("cpu")

def heat_equation_loss(model, spatial_time_tensor, alpha=0.01):
    spatial_time_tensor.requires_grad_(True)
    u_pred = model(spatial_time_tensor)
    
    du_dX = torch.autograd.grad(
        u_pred, spatial_time_tensor, 
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
    )[0]
    
    du_dt = du_dX[:, 3]
    du_dx = du_dX[:, 0]
    
    d2u_dx2 = torch.autograd.grad(
        du_dx, spatial_time_tensor, 
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0][:, 0]
    
    pde_residual = du_dt - (alpha * d2u_dx2)
    pde_loss = torch.mean(pde_residual ** 2)
    
    center_dist = ((spatial_time_tensor[:, 0] - 0.5)**2 + (spatial_time_tensor[:, 1] - 0.5)**2).unsqueeze(1)
    
    # 2. DYNAMIC DEVICE SYNC: Inherits the CPU state automatically
    current_device = spatial_time_tensor.device
    target_temp = torch.exp(-center_dist * 20).to(current_device)
    t_weight = torch.exp(-spatial_time_tensor[:, 3]).unsqueeze(1).to(current_device)
    
    boundary_loss = torch.mean(t_weight * (u_pred - target_temp)**2)
    
    return pde_loss + boundary_loss

def execute_training_pipeline():
    print("="*50)
    print(f"INITIALIZING HARDWARE ACCELERATION: {str(device).upper()}")
    print("="*50)
    
    model = ThermalSurrogateModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Loading massive dataset into memory...")
    try:
        with h5py.File('data/training_mesh.h5', 'r') as f:
            raw_data = f['space_time_coordinates'][:]
    except FileNotFoundError:
        print("ERROR: Run generate_data.py first to create the dataset.")
        return
        
    training_data = torch.tensor(raw_data, dtype=torch.float32).to(device)
    print(f"Training on {training_data.shape[0]} data points...")
    
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = heat_equation_loss(model, training_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total Loss: {loss.item():.8f}")
            
    os.makedirs('data', exist_ok=True)
    
    # Model is already on CPU, but calling it anyway is good practice
    model.to("cpu") 
    torch.save(model.state_dict(), 'data/trained_pinn_model.pth')
    print("SUCCESS: Elite mathematical model saved to data/trained_pinn_model.pth")

if __name__ == "__main__":
    execute_training_pipeline()