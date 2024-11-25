import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import numpy as np
import itertools
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix, lil_matrix, csr_matrix
from joblib import Parallel, delayed
from dwave.system import LeapHybridSampler
import dimod
import os
import json
from collections import defaultdict
import hashlib
from event_model.event_model import event, track
from q_event_model import vp2q_event
import validator.validator_lite as vl
from copy import deepcopy
import restricted_event_model as vprem
from event_model import event_model as em

def visualize_segments(segments):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Constructed Segments")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for seg in segments:
        x = [seg.from_hit.x, seg.to_hit.x]
        y = [seg.from_hit.y, seg.to_hit.y]
        z = [seg.from_hit.z, seg.to_hit.z]
        ax.plot(x, y, z, color="purple", linestyle="-", label="Segment" if seg == segments[0] else "")

    ax.legend()
    plt.show()


def plot_qubo_sparsity(A_total, title="QUBO Matrix Sparsity"):
    plt.figure(figsize=(10, 10))
    plt.spy(A_total, markersize=1)
    plt.title(title)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.show()

def find_segments(s0, active):
    found_s = []
    for s1 in active:
        if s0.from_hit.id == s1.to_hit.id or \
        s1.from_hit.id == s0.to_hit.id:
            found_s.append(s1)
    return found_s


def get_qubo_solution(sol_sample, event, segments):
    #different active segments? 
    active_segments = [
        segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state == 1
    ]

    active = deepcopy(active_segments)
    tracks = []

    while active:
        s = active.pop()
        nextt = find_segments(s, active)
        track = set([s.from_hit.id, s.to_hit.id])


        while len(nextt):
            s = nextt.pop()
            try:
                active.remove(s)
            except ValueError:
                pass
            nextt += find_segments(s, active)
            track = track.union(set([s.from_hit.id, s.to_hit.id]))
        # List of tracks
        tracks.append(track)

    tracks_processed=[]
    for track in tracks:
        tracks_processed.append(em.track([list(filter(lambda b: b.id == a, event.hits))[0] for a in track]))
    print(f'\n tracks{tracks}')
    print(f'\n tracks_processed{tracks_processed}')

    return tracks_processed  # Added return statement

os.environ['DWAVE_API_TOKEN'] = 'DEV-b59f413d6a1407427e9f0079dd8e3cfb8106e58d'  

def qubosolverHr(A, b):
    A = csc_matrix(A)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})

    row, col = A.nonzero()
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])
        else:
            bqm.add_variable(i, A[i, j])

    #print the statistics just to be sure whats happenijng
    print(f"\n[QUBO Solver] BQM has {len(bqm.variables)} variables and {len(bqm.quadratic)} interactions.")

    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.array([best_sample[i] for i in range(len(b))], dtype=int)

    #solution statistics
    num_selected = np.sum(sol_sample)
    print(f"[QUBO Solver] Number of variables selected (value=1): {num_selected}")

    return sol_sample

class Segment:
    def __init__(self, from_hit, to_hit):

        if not from_hit or not to_hit:
            raise ValueError("Both from_hit and to_hit must be valid Hit objects.")
        self.from_hit = from_hit
        self.to_hit = to_hit

    def to_vect(self):

        return np.array([
            self.to_hit.x - self.from_hit.x,
            self.to_hit.y - self.from_hit.y,
            self.to_hit.z - self.from_hit.z
        ])

    def length(self):
        vect = self.to_vect()
        return np.linalg.norm(vect)

    def normalized_vect(self):
        vect = self.to_vect()
        norm = np.linalg.norm(vect)
        if norm == 0:
            raise ValueError("Zero-length segment cannot be normalized.")
        return vect / norm

    def __repr__(self):
        return (f"Segment(from_hit=Hit(x={self.from_hit.x}, y={self.from_hit.y}, z={self.from_hit.z}), "
                f"to_hit=Hit(x={self.to_hit.x}, y={self.to_hit.y}, z={self.to_hit.z}))")

    def __str__(self):
        return f"Segment from ({self.from_hit.x}, {self.from_hit.y}, {self.from_hit.z}) to ({self.to_hit.x}, {self.to_hit.y}, {self.to_hit.z})"


import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed

def generate_hamiltonian(event, params):
    print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    # Deep copy and sort modules by module_id
    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.module_id)  # Sorted by module_id instead of z coordinate

    print("[generate_hamiltonian] Modules deep-copied and sorted by module_id")
    print(f"[generate_hamiltonian] Number of modules after sorting: {len(modules)}")

    # Generate segments skipping over modules without hits
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_idx = idx + 1

        # Skip modules that have no hits
        while next_idx < len(modules) and not modules[next_idx].hits:
            next_idx += 1

        # No valid next module with hits, just break
        if next_idx >= len(modules):
            break

        hits_current = current_module.hits
        hits_next = modules[next_idx].hits

        # Skip if no hits on the current module
        if not hits_current:
            continue

        # Generate segments between all pairs of hits in the current and next modules
        new_segments = [
            Segment(from_hit, to_hit)
            for from_hit, to_hit in itertools.product(hits_current, hits_next)
        ]
        segments.extend(new_segments)

        print(f"Generated {len(new_segments)} segments.")

    N = len(segments)
    print(f"[generate_hamiltonian] Total number of segments generated: {N}")

    if N == 0:
        print("[generate_hamiltonian] No segments generated. Returning empty Hamiltonian.")
        return np.zeros((0, 0)), np.zeros(0), segments

    # Initialize Hamiltonian matrices and vector
    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    b = np.zeros(N)

    # Uncomment and correct if you intend to use A_inh
    # for i, seg in enumerate(segments):
        # if seg.from_hit.module_id == seg.to_hit.module_id:
            # A_inh[i, i] = beta
            # print(f"[Inh interaction] A_inh[{i}, {i}] = {A_inh[i, i]} (penalized for same module).")

    print("[generate_hamiltonian] Populating A_ang, A_bif")
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i != j:
                vect_i = seg_i.to_vect()
                vect_j = seg_j.to_vect()
                norm_i = np.linalg.norm(vect_i)
                norm_j = np.linalg.norm(vect_j)
                
                # Avoid division by zero
                if norm_i == 0 or norm_j == 0:
                    cosine = 0
                    print(f"[generate_hamiltonian] Warning: Zero-length vector detected for segments {i} or {j}.")
                else:
                    cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

                eps = 1e-5

                # Populate A_ang if vectors are parallel
                if np.abs(cosine - 1) < eps:
                    A_ang[i, j] = -50
                    print(f"[generate_hamiltonian] A_ang[{i}, {j}] set to -50 (angular consistency).")

                # Populate A_bif for bifurcations based on module_id and hit_id
                if (seg_i.from_hit.module_id == seg_j.from_hit.module_id) and (seg_i.to_hit.id != seg_j.to_hit.id):
                    A_bif[i, j] = alpha
                    print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to {alpha} (bifurcation: same from_hit).")
                if (seg_i.from_hit.module_id != seg_j.from_hit.module_id) and (seg_i.to_hit.module_id == seg_j.to_hit.module_id):
                    A_bif[i, j] = alpha
                    print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to {alpha} (bifurcation: same to_hit).")

    # Compute the final Hamiltonian matrix
    A = lambda_val * (A_ang + A_bif + A_inh)
    print("[generate_hamiltonian] Combined A_ang, A_bif, and A_inh into final Hamiltonian matrix A.")

    # Print Hamiltonian matrix statistics
    print(f"[generate_hamiltonian] Hamiltonian matrix A shape: {A.shape}")
    print(f"[generate_hamiltonian] Hamiltonian matrix A has {np.count_nonzero(A)} non-zero elements.")

    return A, b, segments




#function just to see whats up with the solutions
def calculate_residuals(reconstructed_tracks, ground_truth_particles):

    residuals = []
    for rec_track in reconstructed_tracks:
        best_match = None
        min_residual = float('inf')

        for true_particle in ground_truth_particles:
            rec_hit_ids = set(hit.id for hit in rec_track.hits)
            true_hit_ids = set(hit.id for hit in true_particle.velohits)
            common_hits = rec_hit_ids & true_hit_ids
            total_hits = rec_hit_ids | true_hit_ids
            residual = len(total_hits) - len(common_hits)

            if residual < min_residual:
                min_residual = residual
                best_match = true_particle

        residuals.append((rec_track, best_match, min_residual))

    return residuals




def visualize_tracks(reconstructed_tracks, ground_truth_particles):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle Track Reconstruction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #reconstructed track coppied
    for track in reconstructed_tracks:
        x = [hit.x for hit in track.hits]
        y = [hit.y for hit in track.hits]
        z = [hit.z for hit in track.hits]
        ax.plot(x, y, z, linestyle="--", color="red", label="Reconstructed Track" if track == reconstructed_tracks[0] else "")

    #true particle tracks plotting
    for particle in ground_truth_particles:
        x = [hit.x for hit in particle.velohits]
        y = [hit.y for hit in particle.velohits]
        z = [hit.z for hit in particle.velohits]
        ax.plot(x, y, z, linestyle="-", color="blue", label="True Track" if particle == ground_truth_particles[0] else "")

    ax.legend()
    plt.show()

def visualize_reconstructed_tracks(reconstructed_tracks):
    """
    Visualizes reconstructed particle tracks in a 3D plot.

    Parameters:
    - reconstructed_tracks (List[Set[Hit]]): A list where each element is a set of Hit objects representing a reconstructed track.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle Track Reconstruction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Iterate over each track
    for idx, track in enumerate(reconstructed_tracks):
        # Convert the set to a sorted list for coherent plotting
        # Sorting by z-coordinate; adjust the key if another order is preferred
        sorted_hits = sorted(track, key=lambda hit: hit.z)
        
        # Extract x, y, z coordinates
        x = [hit.x for hit in sorted_hits]
        y = [hit.y for hit in sorted_hits]
        z = [hit.z for hit in sorted_hits]
        
        # Plot the track
        # Label only the first track to avoid duplicate labels in the legend
        label = "Reconstructed Track" if idx == 0 else ""
        ax.plot(x, y, z, linestyle="--", color="red", label=label)
    
    # Add legend only if there are tracks
    if reconstructed_tracks:
        ax.legend()
    
    plt.show()
def visualize_truth(ground_truth_particles):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle Track Reconstruction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #true particle tracks plotting
    for particle in ground_truth_particles:
        x = [hit.x for hit in particle.velohits]
        y = [hit.y for hit in particle.velohits]
        z = [hit.z for hit in particle.velohits]
        ax.plot(x, y, z, linestyle="-", color="blue", label="True Track" if particle == ground_truth_particles[0] else "")

    ax.legend()
    plt.show()

#optimizing weights
def plot_qubo_histogram(A_total, title="QUBO Coefficient Distribution"):

    data = A_total.data
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


#plotting function to understand wtf is happening with coefficients
def analyze_qubo_coefficients(A_total, top_n=10):
 
    upper_tri = np.triu(A_total, k=1).tocoo()
    coefficients = upper_tri.data
    top_positive_indices = np.argsort(coefficients)[-top_n:]
    top_positive = coefficients[top_positive_indices]
    top_positive_vars = list(zip(upper_tri.row[top_positive_indices], upper_tri.col[top_positive_indices]))

    top_negative_indices = np.argsort(coefficients)[:top_n]
    top_negative = coefficients[top_negative_indices]
    top_negative_vars = list(zip(upper_tri.row[top_negative_indices], upper_tri.col[top_negative_indices]))

    print(f"Top {top_n} Positive Coefficients:")
    for coeff, vars in zip(top_positive, top_positive_vars):
        print(f"Variables {vars}: {coeff}")

    print(f"\nTop {top_n} Negative Coefficients:")
    for coeff, vars in zip(top_negative, top_negative_vars):
        print(f"Variables {vars}: {coeff}")


from types import SimpleNamespace


#combines everything, compares and plots correct and found solutions
def main():
    params = {
        'lambda': 1.0, #multiply at the end +
        'alpha': 50.0, #a_bif penelizes bifunctions -
        'beta': 100.0, #same module penalty A_inh -
        'track_length_penalty' : 30,
        'two_hit_penalty' : 10,
        'hit_overlap_penalty': 10 
    }

    solutions = {
        "qubo_track_reconstruction": []
    }
    validation_data = []

    for (dirpath, dirnames, filenames) in os.walk("events"):
        for i, filename in enumerate(filenames):
            if i != 2:
                continue

            restrict_consec_modules = False
            restrict_min_nb_hits = 3
            restricted_modules_even = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
            restricted_modules_odd = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
            restricted_modules = list(range(52))

            f = open(os.path.realpath(os.path.join(dirpath, filename)))
            event, json_data = vprem.restrict_event(json.loads(f.read()), restricted_modules, restrict_min_nb_hits, restrict_consec_modules)
            f.close()

            ev = em.event(json_data)
            q_event = vp2q_event(event)
            
            print(f"[Main] Reconstructing event {i}...")
            A, b, segments = generate_hamiltonian(q_event, params)
        
        
            sol_sample = qubosolverHr(A, b)
            reconstructed_tracks = get_qubo_solution(sol_sample, q_event, segments)
            print('\n', 'reconstructed tracks', reconstructed_tracks)

            solutions["qubo_track_reconstruction"].append(reconstructed_tracks)
            validation_data.append(json_data)
            print(validation_data)
            print(solutions)

            validator_event_instance = vl.parse_json_data(json_data)
            visualize_truth(validator_event_instance.particles)

    for k, v in sorted(solutions.items()):
        print(f"\n[Validation Summary] Validating tracks from {k}:")
        vl.validate_print(validation_data, v)
        print()

if __name__ == "__main__":
    main()