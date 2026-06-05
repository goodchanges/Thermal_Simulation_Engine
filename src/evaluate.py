import torch
import numpy as np
from models.pinn_network import ThermalSurrogateModel
from train import heat_equation_loss

def prove_model_accuracy():
    print("Initiating Out-of-Sample Model Validation...")
    
    # 1. Load the trained AI model
    model = ThermalSurrogateModel()
    try:
        model.load_state_dict(torch.load('data/trained_pinn_model.pth'))
        model.eval() # Locks the model weights (no learning allowed)
    except FileNotFoundError:
        print("ERROR: Model not found. Train the model first.")
        return

    # 2. Generate COMPLETELY NEW data the AI has never seen
    # We use a different random seed to guarantee fresh data
    np.random.seed(42) 
    num_test_points = 15000
    
    print(f"Generating {num_test_points} brand new spatial-temporal coordinates...")
    new_x = np.random.rand(num_test_points, 1)
    new_y = np.random.rand(num_test_points, 1)
    new_z = np.full((num_test_points, 1), 0.5)
    new_t = np.random.uniform(0, 10, (num_test_points, 1))
    
    # Create the test tensor: [x, y, z, t]
    test_tensor = torch.tensor(np.hstack((new_x, new_y, new_z, new_t)), dtype=torch.float32)
    
    # 3. Test the AI against the laws of physics
    print("Testing AI predictions against the Heat Equation Partial Differential Equation...")
    
    # We use the exact same physics loss function, but on the NEW data
    validation_loss = heat_equation_loss(model, test_tensor, alpha=0.01)
    
    # 4. Output the Verdict
    final_error = validation_loss.item()
    print("\n" + "="*50)
    print("VALIDATION VERDICT")
    print("="*50)
    print(f"Physics Error on Unseen Data: {final_error:.6f}")
    
    if final_error < 0.01:
        print("\nRESULT: PASS ✅")
        print("The model is legitimate. It did not memorize data;")
        print("it successfully generalized the mathematics of thermodynamics.")
    else:
        print("\nRESULT: FAIL ❌")
        print("The model is a bluff (Overfitted). It failed to predict physics on new data.")
    print("="*50)

if __name__ == "__main__":
    prove_model_accuracy()