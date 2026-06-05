from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from src.models.pinn_network import ThermalSurrogateModel
# 1. Initialize the Web Server
app = FastAPI(title="Thermal Simulation Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website to talk to your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

# 2. Hardware: Force CPU for safe, lightweight API serving
device = torch.device("cpu")

# 3. Load the Cloud-Trained Brain
print("Loading trained model weights into memory...")
model = ThermalSurrogateModel().to(device)
try:
    # This must match the exact path where you dropped the Colab file
    model.load_state_dict(torch.load('data/trained_pinn_model.pth', map_location=device))
    model.eval() # CRITICAL: Lock the weights (No more learning!)
    print("Model successfully loaded and locked for inference.")
except FileNotFoundError:
    print("WARNING: Model file not found. Ensure 'trained_pinn_model.pth' is in the data/ folder.")

# 4. Define the Expected Web Traffic (The Request Body)
class Coordinates(BaseModel):
    x: float
    y: float
    z: float
    time: float

# 5. The Core Prediction Endpoint
@app.post("/predict_temperature")
async def predict(data: Coordinates):
    try:
        # Step A: Translate the JSON web traffic into a PyTorch Tensor
        input_array = np.array([[data.x, data.y, data.z, data.time]])
        input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        
        # Step B: Ask the AI for the prediction (No gradients calculated)
        with torch.no_grad():
            prediction = model(input_tensor)
            
        # Step C: Translate the Tensor back to JSON for the Frontend
        final_temp = float(prediction.item())
        
        return {
            "status": "success",
            "coordinates": {"x": data.x, "y": data.y, "z": data.z, "time": data.time},
            "predicted_temperature": final_temp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "Thermal Simulation Engine API is actively running."}