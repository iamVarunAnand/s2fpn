# import the necessary packages
from uscnn.utils import Icosphere, sparse2tensor, spmatmul
from scipy import sparse
import numpy as np
import argparse
import pickle
import torch


if __name__ == "__main__":
    # construct an argument parser to parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--level", required=True, type=int, help="mesh level")
    args = vars(ap.parse_args())

    # constants
    level = args["level"]

    # initialise the icospheres
    ico = Icosphere(level=level)
    ico_down = Icosphere(level=level - 1)

    # ico vertices influenced by type-0 ico_down vertices
    stack1 = np.stack([ico_down.faces[:, 0], ico.faces[3:][::4, 0], ico.faces[3:][::4, 2]], axis=-1)

    # ico vertices influenced by type-1 ico_down vertices
    stack2 = np.stack([ico_down.faces[:, 1], ico.faces[3:][::4, 0], ico.faces[3:][::4, 1]], axis=-1)

    # ico vertices influenced by type-2 ico_down vertices
    stack3 = np.stack([ico_down.faces[:, 2], ico.faces[3:][::4, 1], ico.faces[3:][::4, 2]], axis=-1)

    # stack all vertices together
    stack = np.vstack([stack1, stack2, stack3])

    # sort by ico_down vertices
    stack = stack[stack[:, 0].argsort()]

    # split by unique ico_down vertices
    u, idx = np.unique(stack[:, 0], return_index=True)
    splits = np.split(stack[:, 1:], idx[1:])

    # loop through the splits
    sd, si, sj = [], [], []
    for i, s in enumerate(splits):
        # obtain all the neighbours for the current vertex i
        neighbours = np.unique(s.flatten())

        # loop through the neighbours
        for vertex in neighbours:
            sd.append(1 / neighbours.shape[0])
            si.append(i)
            sj.append(vertex)

    # compute the neighbourhood vertex aggregation sparse matrix
    VA = sparse.coo_matrix((sd, (si, sj)), shape=(ico_down.nv, ico.nv))

    # load the appropriate mesh file
    mesh = pickle.load(open(f"uscnn/meshes/icosphere_{level - 1}.pkl", "rb"))

    # add in the VA matrix to the mesh
    mesh["VA"] = VA

    # save the mesh to a pickle file
    pickle.dump(mesh, open(f"uscnn/meshes/v2/icosphere_{level - 1}.pkl", "wb+"))
