# import the necessary packages
from ..utils import sparse2tensor
import pickle
import torch


def load_mesh(path):
    print(f"[INFO] loading mesh {path}...")

    # read the mesh file
    data = pickle.load(open(path, "rb"))

    # extract the required matrices
    G = sparse2tensor(data["G"].tocoo())
    L = data["L"]
    NS = torch.tensor(data["NS"], dtype=torch.float32)
    EW = torch.tensor(data["EW"], dtype=torch.float32)
    F2V = data["F2V"]
    nv_prev = data["nv_prev"]
    nv = data["V"].shape[0]
    F = data["F"]

    try:
        VA = sparse2tensor(data["VA"])
    except KeyError:
        VA = None

    # return the matrices as a dictionary
    return {
        "G": G,
        "L": L,
        "NS": NS,
        "EW": EW,
        "F2V": F2V,
        "nv_prev": nv_prev,
        "nv": nv,
        "F": F,
        "VA": VA
    }


# load in all the mesh files
MESHES = {
    0: load_mesh("uscnn/meshes/v2/icosphere_0.pkl"),
    1: load_mesh("uscnn/meshes/v2/icosphere_1.pkl"),
    2: load_mesh("uscnn/meshes/v2/icosphere_2.pkl"),
    3: load_mesh("uscnn/meshes/v2/icosphere_3.pkl"),
    4: load_mesh("uscnn/meshes/v2/icosphere_4.pkl"),
    5: load_mesh("uscnn/meshes/v2/icosphere_5.pkl"),
    6: load_mesh("uscnn/meshes/v2/icosphere_6.pkl"),
    7: load_mesh("uscnn/meshes/v2/icosphere_7.pkl"),
}
