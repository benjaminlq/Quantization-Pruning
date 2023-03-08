import torch
import numpy

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(model, path, device = "cuda"):
    model.load_state_dict(torch.load(path, map_location=device))
    # return model

def to_numpy(tensor: torch.tensor) -> numpy.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()