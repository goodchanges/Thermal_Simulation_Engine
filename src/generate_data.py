import numpy as np
import h5py
import os
from data_structures.spatial_index import prepare_mesh_data

def generate_massive_mesh(num_points=50000, time_steps=50):
    print("Initializing spatial coordinate generation...")
    
    # Generate random raw coordinates for (x, y, z) between 0.0 and 1.0
    raw_spatial_coords = np.random.rand(num_points, 3)
    
    print("Applying deterministic spatial sorting to mesh nodes...")
    # Utilize the strict Quick Sort algorithm to guarantee predictable data architecture
    sorted_spatial_coords = prepare_mesh_data(raw_spatial_coords)
    
    # Generate time steps from 0.0 to 10.0 seconds
    t_coords = np.linspace(0, 10, time_steps)
    
    print("Fusing spatial and temporal dimensions into master dataset...")
    # Create the final [x, y, z, t] dataset
    dataset = []
    for t in t_coords:
        t_array = np.full((num_points, 1), t)
        space_time_block = np.hstack((sorted_spatial_coords, t_array))
        dataset.append(space_time_block)
        
    final_tensor_data = np.vstack(dataset)
    return final_tensor_data

def save_to_hdf5(data, filename="data/training_mesh.h5"):
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Writing {data.shape[0]} data points to high-performance HDF5 format...")
    with h5py.File(filename, 'w') as f:
        f.create_dataset('space_time_coordinates', data=data, compression="gzip")
    print(f"Dataset successfully saved to {filename}")

if __name__ == "__main__":
    # Generating 40,000 data points for the initial test run
    massive_dataset = generate_massive_mesh(num_points=10000, time_steps=50) 
    save_to_hdf5(massive_dataset)