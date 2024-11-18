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

# Set your D-Wave API token
os.environ['DWAVE_API_TOKEN'] = 'DEV-b59f413d6a1407427e9f0079dd8e3cfb8106e58d'  # Replace with your token

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

    # Debug: Print BQM statistics
    print(f"\n[QUBO Solver] BQM has {len(bqm.variables)} variables and {len(bqm.quadratic)} interactions.")

    # Use LeapHybridSampler to solve
    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.array([best_sample[i] for i in range(len(b))], dtype=int)

    # Debug: Print solution statistics
    num_selected = np.sum(sol_sample)
    print(f"[QUBO Solver] Number of variables selected (value=1): {num_selected}")

    return sol_sample

# Define the Segment class
class Segment:
    def __init__(self, from_hit, to_hit):
        self.from_hit = from_hit
        self.to_hit = to_hit

    def to_vect(self):
        """Returns the vector representation of the segment."""
        return np.array([
            self.to_hit.x - self.from_hit.x,
            self.to_hit.y - self.from_hit.y,
            self.to_hit.z - self.from_hit.z
        ])

import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed
import itertools

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    """Performs angular consistency and bifurcation checks for a given index `i`."""
    results_ang = []
    results_bif = []
    
    vect_i = vectors[i]
    norm_i = norms[i]

    for j in range(i + 1, N):  # Only upper triangle
        vect_j = vectors[j]
        norm_j = norms[j]
        cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

        # Angular consistency
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, 1))

        # Bifurcation consistency
        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, -alpha))
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, -alpha))

    return results_ang, results_bif

def generate_hamiltonian_real_data(event, params):
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    gamma = params.get('gamma', 1000.0)  # Penalty for segments within the same module, if needed

    # Filter out modules with no hits
    non_empty_modules = [m for m in event.modules if len(m.hits()) > 0]

    # Now sort the non-empty modules
    modules = sorted(non_empty_modules, key=lambda m: m.z)
    print(f"\n[Hamiltonian Generation] Number of non-empty modules: {len(modules)}")

    # Generate segments between hits in adjacent or nearby modules, skipping over empty modules
    segments = []
    num_modules = len(modules)
    for idx, module_from in enumerate(modules):
        hits_from = module_from.hits()
        # Find the next module with hits
        for next_idx in range(idx + 1, num_modules):
            module_to = modules[next_idx]
            hits_to = module_to.hits()
            if hits_to:
                # Create segments between hits_from and hits_to
                for from_hit, to_hit in itertools.product(hits_from, hits_to):
                    # Skip segments between hits on the same module
                    if from_hit.module_number == to_hit.module_number:
                        continue
                    segments.append(Segment(from_hit, to_hit))
                # Once we have connected to the next module with hits, we can break
                break

    N = len(segments)
    print(f"[Hamiltonian Generation] Number of segments generated: {N}")
    b = np.zeros(N)

    # Precompute vectors and norms
    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)
    eps = 1e-2  # Adjusted precision threshold

    # Perform angular and bifurcation checks in parallel
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, alpha, eps)
        for i in range(N)
    )

    # Aggregate results
    A_ang = dok_matrix((N, N), dtype=np.float64)
    A_bif = dok_matrix((N, N), dtype=np.float64)

    for ang_results, bif_results in results:
        for i, j, value in ang_results:
            A_ang[i, j] = value
            A_ang[j, i] = value  # Symmetric
        for i, j, value in bif_results:
            A_bif[i, j] = value
            A_bif[j, i] = value  # Symmetric

    # Convert angular and bifurcation matrices to sparse format
    A_ang = A_ang.tocsc()
    A_bif = A_bif.tocsc()

    # Inhibitory interactions: penalize segments sharing the same hit
    A_inh = dok_matrix((N, N), dtype=np.float64)
    for i in range(N):
        seg_i = segments[i]
        for j in range(i + 1, N):
            seg_j = segments[j]
            if seg_i.from_hit == seg_j.from_hit or seg_i.to_hit == seg_j.to_hit:
                A_inh[i, j] = beta
                A_inh[j, i] = beta  # Symmetric
    A_inh = A_inh.tocsc()

    num_ang_interactions = A_ang.count_nonzero()
    num_bif_interactions = A_bif.count_nonzero()
    num_inh_interactions = A_inh.count_nonzero()

    print(f"[Hamiltonian Generation] Number of angular interactions: {num_ang_interactions}")
    print(f"[Hamiltonian Generation] Number of bifurcation interactions: {num_bif_interactions}")
    print(f"[Hamiltonian Generation] Number of inhibitory interactions: {num_inh_interactions}")

    # Combine matrices into the Hamiltonian
    A = -1 * (A_ang + A_bif + A_inh)

    # Debug: Print Hamiltonian statistics
    total_non_zero = A.count_nonzero()
    print(f"[Hamiltonian Generation] Hamiltonian matrix A: shape {A.shape}, non-zero elements: {total_non_zero}")

    return A, b, segments


def reconstruct_tracks(sol_sample, segments):
    """Reconstructs tracks from the solution sample and segments."""
    # Build mappings
    outgoing_segments = defaultdict(list)  # hit_id -> list of (to_hit)
    incoming_segments = defaultdict(list)  # hit_id -> list of (from_hit)
    hits = {}  # hit_id -> hit object

    # Collect hits and build mappings based on selected segments
    num_selected_segments = 0
    for i, seg in enumerate(segments):
        if sol_sample[i] == 1:
            num_selected_segments += 1
            from_hit = seg.from_hit
            to_hit = seg.to_hit
            outgoing_segments[from_hit.id].append(to_hit)
            incoming_segments[to_hit.id].append(from_hit)
            hits[from_hit.id] = from_hit
            hits[to_hit.id] = to_hit

    print(f"[Track Reconstruction] Number of selected segments: {num_selected_segments}")
    print(f"[Track Reconstruction] Number of unique hits in selected segments: {len(hits)}")

    # Build tracks by traversing the graph of selected segments
    tracks = []
    visited_hits = set()

    for hit_id in hits:
        if hit_id in visited_hits:
            continue
        current_track_hits = []
        queue = [hits[hit_id]]
        while queue:
            current_hit = queue.pop(0)
            if current_hit.id in visited_hits:
                continue
            visited_hits.add(current_hit.id)
            current_track_hits.append(current_hit)
            for next_hit in outgoing_segments.get(current_hit.id, []):
                if next_hit.id not in visited_hits:
                    queue.append(next_hit)
        if current_track_hits:
            current_track_hits.sort(key=lambda h: h.module_number)
            tracks.append(track(current_track_hits))

    print(f"[Track Reconstruction] Number of tracks reconstructed: {len(tracks)}")
    # Optionally, print details of each track
    for idx, tr in enumerate(tracks):
        hit_ids = [hit.id for hit in tr.hits]
        print(f"[Track Reconstruction] Track {idx}: hit IDs {hit_ids}")

    return tracks

def main():
    # Set your parameters
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
            # Process all events or adjust as needed
            if i != 2:
               continue
            # Get an event
            file_path = os.path.realpath(os.path.join(dirpath, filename))
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            event_instance = event(json_data)
            print(f"\n[Main] Processing event {filename}")
            print(f"[Main] Number of modules in event: {len(event_instance.modules)}")
            total_hits = sum(len(list(m)) for m in event_instance.modules)
            print(f"[Main] Total number of hits in event: {total_hits}")

            # Do track reconstruction
            print(f"[Main] Reconstructing event {i}...")
            # Generate the Hamiltonian
            A, b, segments = generate_hamiltonian_real_data(event_instance, params)

            # Debug: Check if A is empty
            if A.count_nonzero() == 0:
                print("[Main] Hamiltonian matrix A is empty. Skipping event.")
                continue

            # Solve the QUBO problem
            sol_sample = qubosolverHr(A, b)
            # Reconstruct tracks from the solution
            tracks = reconstruct_tracks(sol_sample, segments)
            print(f"[Main] Number of tracks reconstructed: {len(tracks)}")

            # Append the solution and json_data
            solutions["qubo_track_reconstruction"].append(tracks)
            validation_data.append(json_data)

    # Validate the solutions
    for k, v in sorted(solutions.items()):
        print(f"\n[Validation] Validating tracks from {k}:")
        vl.validate_print(validation_data, v)
        print()

if __name__ == "__main__":
    main()
