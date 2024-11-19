import numpy as np
import itertools
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed
from dwave.system import LeapHybridSampler
import dimod
import os
import json
from collections import defaultdict
import hashlib
from event_model.event_model import event, track
from validator import validator_lite as vl
from copy import deepcopy
import matplotlib.pyplot as plt
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

#define a segment class since its not in the event_model
class Segment:
    def __init__(self, from_hit, to_hit):
        """
        Initialize a Segment connecting two hits.

        Args:
            from_hit (Hit): The starting hit of the segment.
            to_hit (Hit): The ending hit of the segment.
        """
        if not from_hit or not to_hit:
            raise ValueError("Both from_hit and to_hit must be valid Hit objects.")
        self.from_hit = from_hit
        self.to_hit = to_hit

    def to_vect(self):
        """
        Returns the vector representation of the segment.

        Returns:
            np.ndarray: A 3D vector [dx, dy, dz] from from_hit to to_hit.
        """
        return np.array([
            self.to_hit.x - self.from_hit.x,
            self.to_hit.y - self.from_hit.y,
            self.to_hit.z - self.from_hit.z
        ])

    def length(self):
        """
        Compute the Euclidean length (norm) of the segment.

        Returns:
            float: The length of the segment.
        """
        vect = self.to_vect()
        return np.linalg.norm(vect)

    def normalized_vect(self):
        """
        Returns the normalized vector representation of the segment.

        Returns:
            np.ndarray: A unit vector [dx, dy, dz] from from_hit to to_hit.
        """
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

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    results_ang = []
    results_bif = []
    
    vect_i = vectors[i]
    norm_i = norms[i]

    for j in range(i + 1, N):  
        vect_j = vectors[j]
        norm_j = norms[j]
        cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

        # Angular consistency
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, -1))

        # Bifurcation consistency
        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, alpha))
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, alpha))

    return results_ang, results_bif

def generate_hamiltonian_real_data(event, params):
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)

# Sort modules by their z-coordinate
    modules = sorted(event.modules, key=lambda m: m.z)

# Generate segments, skipping over modules without hits
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        # Find the next module with hits
        next_idx = idx + 1
        while next_idx < len(modules) and not modules[next_idx].hits():
            next_idx += 1

        # If no valid next module with hits, break
        if next_idx >= len(modules):
            break

        # Current and next modules' hits
        hits_current = current_module.hits()
        hits_next = modules[next_idx].hits()

        # Skip if the current module has no hits (redundant safeguard)
        if not hits_current:
            continue

        # Generate segments between hits in the current and next module
        segments.extend(Segment(from_hit, to_hit) for from_hit, to_hit in itertools.product(hits_current, hits_next))

        print(f"Generated {len(segments)} segments.")


    # Check the total number of segments
    N = len(segments)
    print(f"[Hamiltonian Generation] Number of segments generated: {N}")

    # Generate vectors and norms for consistency checks
    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)
    eps = 1e-9

    # Perform angular and bifurcation consistency checks
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, alpha, eps)
        for i in range(N)
    )

    A_ang = dok_matrix((N, N), dtype=np.float64)
    A_bif = dok_matrix((N, N), dtype=np.float64)

    # Aggregate angular and bifurcation results
    for ang_results, bif_results in results:
        for i, j, value in ang_results:
            A_ang[i, j] = value
            A_ang[j, i] = value
        for i, j, value in bif_results:
            A_bif[i, j] = value
            A_bif[j, i] = value

    A_ang = A_ang.tocsc()
    A_bif = A_bif.tocsc()

    # Penalize overlapping segments
    A_inh = dok_matrix((N, N), dtype=np.float64)
    for i in range(N):
        seg_i = segments[i]
        for j in range(i + 1, N):
            seg_j = segments[j]
            if seg_i.from_hit == seg_j.from_hit or seg_i.to_hit == seg_j.to_hit:
                A_inh[i, j] = beta
                A_inh[j, i] = beta
    A_inh = A_inh.tocsc()

    # Combine Hamiltonian components
    A = lambda_val * (A_ang + A_bif + A_inh)

    # Debugging: Print Hamiltonian statistics
    print(f"[Hamiltonian Generation] Hamiltonian matrix A: shape {A.shape}, non-zero elements: {A.count_nonzero()}")
    return A, np.zeros(N), segments


def generate_hamiltonian(event, params):
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)

    modules = event.modules.copy()
    modules.sort(key=lambda a: a.z)

    # Generate segments between hits in adjacent modules
    segments = []
    for idx in range(len(modules) - 1):
        hits_from = modules[idx].hits()
        hits_to = modules[idx + 1].hits()
        for from_hit, to_hit in itertools.product(hits_from, hits_to):
            segments.append(Segment(from_hit, to_hit))


    N = len(segments)
    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    b = np.zeros(N)

    eps = 1e-3  # Adjusted threshold

    for i, seg_i in enumerate(segments):
        for j in range(i + 1, N):
            seg_j = segments[j]

            # Angular consistency
            vect_i = seg_i.to_vect()
            vect_j = seg_j.to_vect()
            cosine = np.dot(vect_i, vect_j) / (np.linalg.norm(vect_i) * np.linalg.norm(vect_j))

            if np.abs(cosine - 1) < eps:
                A_ang[i, j] += -1  # Negative weight to encourage selection
                A_ang[j, i] = A_ang[i, j]  # Symmetric

            # Bifurcation penalties
            if (seg_i.from_hit == seg_j.from_hit) and (seg_i.to_hit != seg_j.to_hit):
                A_bif[i, j] += alpha  # Positive penalty
                A_bif[j, i] = A_bif[i, j]  # Symmetric

            if (seg_i.from_hit != seg_j.from_hit) and (seg_i.to_hit == seg_j.to_hit):
                A_bif[i, j] += alpha  # Positive penalty
                A_bif[j, i] = A_bif[i, j]  # Symmetric

            # Inhibitory interactions
            if (seg_i.from_hit == seg_j.from_hit) or (seg_i.to_hit == seg_j.to_hit):
                A_inh[i, j] += beta  # Positive penalty
                A_inh[j, i] = A_inh[i, j]  # Symmetric

    A = A_ang + A_bif + A_inh


    return A, b, segments

def visualize_segments(segments):
    """
    Visualize the segments constructed during Hamiltonian generation.

    Args:
        segments (list of Segment): List of segments to visualize.
    """
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

def find_segments(s0, active):
    found_s = []
    for s1 in active:
        if s0.from_hit.id == s1.to_hit.id or \
           s1.from_hit.id == s0.to_hit.id:
            found_s.append(s1)
    print(f"find_segments: Found {len(found_s)} segments for segment {s0}")
    return found_s

def get_qubo_solution(sol_sample, event, segments):
    print(f"sol_sample: {sol_sample}")
    print(f"Number of segments: {len(segments)}")
    
    active_segments = [segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state > np.min(sol_sample)]
    print(f"Active segments count: {len(active_segments)}")
    active = deepcopy(active_segments)
    tracks = []


    while len(active):
        s = active.pop()
        nextt = find_segments(s, active)
        track_hits = set([s.from_hit.id, s.to_hit.id])
        while len(nextt):
            s = nextt.pop()
            try:
                active.remove(s)
            except ValueError:
                pass
            nextt += find_segments(s, active)
            track_hits = track_hits.union(set([s.from_hit.id, s.to_hit.id]))
        tracks.append(track_hits)


    print(f"Generated {len(tracks)} tracks.")
    
    #convert hit IDs to track objects
    tracks_processed = []
    for track_hit_ids in tracks:
        hits = [list(filter(lambda b: b.id == a, event.hits))[0] for a in track_hit_ids]
        tracks_processed.append(track(hits))  
        print(f"Track processed: {hits}")
    
    return tracks_processed


def main():
    params = {
        'lambda': 1.0,
        'alpha': 10.0,
        'beta': 10.0
    }

    solutions = {
        "qubo_track_reconstruction": []
    }
    validation_data = []

    # Iterate all events
    for (dirpath, dirnames, filenames) in os.walk("events"):
        for i, filename in enumerate(filenames):
            if i != 2:
               continue
            file_path = os.path.realpath(os.path.join(dirpath, filename))
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            event_instance = event(json_data)
            print(f"\n[Main] Processing event {filename}")
            print(f"[Main] Number of modules in event: {len(event_instance.modules)}")
            total_hits = sum(len(list(m)) for m in event_instance.modules)
            print(f"[Main] Total number of hits in event: {total_hits}")

            print(f"[Main] Reconstructing event {i}...")
            A, b, segments = generate_hamiltonian_real_data(event_instance, params)
            visualize_segments(segments)
            #use for debugging the optimized ham
            #if A.count_nonzero() == 0:
                #print("[Main] Hamiltonian matrix A is empty. Skipping event.")
                #continue

        
            sol_sample = qubosolverHr(A, b)
            tracks = get_qubo_solution(sol_sample, event_instance, segments)
            print(f"[Main] Number of tracks reconstructed: {len(tracks)}")
            solutions["qubo_track_reconstruction"].append(tracks)
            validation_data.append(json_data)

    for k, v in sorted(solutions.items()):
        print(f"\n[Validation] Validating tracks from {k}:")
        vl.validate_print(validation_data, v)
        print()

if __name__ == "__main__":
    main()
