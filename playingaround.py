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
import validator.validator_lite as vl
from copy import deepcopy


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
    """
    Plots the sparsity pattern of the QUBO matrix.
    """
    plt.figure(figsize=(10, 10))
    plt.spy(A_total, markersize=1)
    plt.title(title)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.show()



def find_segments(s0, active):
    found_s = []
    direction_s0 = s0.normalized_vect()
    for s1 in active:
        direction_s1 = s1.normalized_vect()
        cosine_similarity = np.dot(direction_s0, direction_s1)
        if s0.from_hit.id == s1.to_hit.id or s1.from_hit.id == s0.to_hit.id:
            if cosine_similarity > 0.98:  # Only connect if nearly aligned
                found_s.append(s1)
    return found_s

def get_qubo_solution(sol_sample, event, segments):
    print(f"sol_sample: {sol_sample}")
    print(f"Number of segments: {len(segments)}")
    
    # Filter segments based on the solution from QUBO
    active_segments = [
        segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state > 0
    ]
    print(f"Active segments count: {len(active_segments)}")
    
    # Initialize the active list and tracks
    active = deepcopy(active_segments)
    tracks = []
    used_hits = set()  # Keep track of used hit IDs to avoid overlaps

    while len(active):
        s = active.pop()

        # Skip if any of the hits in this segment have already been used
        if s.from_hit.id in used_hits or s.to_hit.id in used_hits:
            continue

        # Start a new track
        track_hits = {s.from_hit.id, s.to_hit.id}
        used_hits.update(track_hits)

        # Find connected segments
        next_segments = find_segments(s, active)
        while len(next_segments):
            s = next_segments.pop()

            # Skip segments that have already been used
            if s.from_hit.id in used_hits or s.to_hit.id in used_hits:
                continue

            try:
                active.remove(s)
            except ValueError:
                pass

            # Update the track with the new segment
            track_hits.update([s.from_hit.id, s.to_hit.id])
            used_hits.update([s.from_hit.id, s.to_hit.id])
            next_segments += find_segments(s, active)

        # Add the completed track to the list of tracks
        tracks.append(track_hits)

    print(f"Generated {len(tracks)} tracks.")

    # Convert hit IDs to track objects
    tracks_processed = []
    for track_hit_ids in tracks:
        hits = [list(filter(lambda b: b.id == a, event.hits))[0] for a in track_hit_ids]
        tracks_processed.append(track(hits))
        print(f"Track processed: {hits}")

    return tracks_processed


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

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    results_ang = []
    results_bif = []

    # Normalize all the vectors and take care of 0devision case
    vect_i = vectors[i]
    norm_i = norms[i]
    if norm_i == 0:
        raise ValueError(f"Zero-length vector encountered at index {i}")
    vect_i_normalized = vect_i / norm_i  

    for j in range(i + 1, N):
        vect_j = vectors[j]
        norm_j = norms[j]
        if norm_j == 0:
            raise ValueError(f"Zero-length vector encountered at index {j}")
        vect_j_normalized = vect_j / norm_j

        cosine = np.dot(vect_i_normalized, vect_j_normalized)
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, -1))  # Add angular consistency interaction

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

#Sort modules by their z-coordinate
    modules = sorted(event.modules, key=lambda m: m.z)

#generate segments skipping over modules without hits
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_idx = idx + 1
        while next_idx < len(modules) and not modules[next_idx].hits():
            next_idx += 1

        #no valid next module with hits, just break
        if next_idx >= len(modules):
            break

        hits_current = current_module.hits()
        hits_next = modules[next_idx].hits()

        #skip if no hits on a modules
        if not hits_current:
            continue

        #segments between hits in the current and next module
        segments.extend(Segment(from_hit, to_hit) for from_hit, to_hit in itertools.product(hits_current, hits_next))

        print(f"Generated {len(segments)} segments.")


    N = len(segments)
    print(f"[Hamiltonian Generation] Number of segments generated: {N}")

    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)
    eps = 1e-9
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, alpha, eps)
        for i in range(N)
    )

    A_ang = dok_matrix((N, N), dtype=np.float64)
    A_bif = dok_matrix((N, N), dtype=np.float64)

    for ang_results, bif_results in results:
        for i, j, value in ang_results:
            A_ang[i, j] = value
            A_ang[j, i] = value
        for i, j, value in bif_results:
            A_bif[i, j] = value
            A_bif[j, i] = value

    A_ang = A_ang.tocsc()
    A_bif = A_bif.tocsc()

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

    #Hamiltonian statistics
    print(f"[Hamiltonian Generation] Hamiltonian matrix A: shape {A.shape}, non-zero elements: {A.count_nonzero()}")
    return A, np.zeros(N), segments

def generate_hamiltonianC(event, params, verbose=True):
    """
    Generates the Hamiltonian matrix for the QUBO formulation based on the provided event and parameters.

    Parameters:
    - event: An object containing detector modules and hits.
    - params: A dictionary containing the parameters 'alpha', 'beta', and 'epsilon'.
    - verbose: Boolean flag to control the verbosity of the output.

    Returns:
    - A_total: The final Hamiltonian matrix (sparse CSR matrix).
    - b: The linear term vector (zeros in this formulation).
    - segments: The list of generated segments.
    """
    if verbose:
        print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    
    # Retrieve parameters with default values
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    epsilon = 1e-2

    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.z)
    if verbose:
        print(f"[generate_hamiltonian] Modules sorted by z-coordinate. Number of modules: {len(modules)}")

    # Generate all possible segments between consecutive modules
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_module = modules[idx + 1]
        hits_current = current_module.hits()
        hits_next = next_module.hits()
        for from_hit, to_hit in itertools.product(hits_current, hits_next):
            try:
                segment = Segment(from_hit, to_hit)
                if segment.length() == 0:
                    raise ValueError("Zero-length segment.")
                segments.append(segment)
            except ValueError as e:
                if verbose:
                    print(f"[generate_hamiltonian] Skipping invalid segment: {e}")

    N = len(segments)
    if verbose:
        print(f"[generate_hamiltonian] Total number of segments generated: {N}")

    if N == 0:
        if verbose:
            print("[generate_hamiltonian] No segments generated. Returning empty Hamiltonian.")
        return csr_matrix((0, 0)), np.zeros(0), segments

    # Initialize Hamiltonian components as sparse
    A_ang = lil_matrix((N, N))
    A_bif = lil_matrix((N, N))
    A_inh = lil_matrix((N, N))

    #module assignments for A_inh
    from_modules = np.array([seg.from_hit.module_number for seg in segments])
    to_modules = np.array([seg.to_hit.module_number for seg in segments])

    for i in range(N):
        for j in range(N):
            if i != j:
                shared = (to_modules[i] == from_modules[j]) or (from_modules[i] == to_modules[j])
                if shared: 
                    A_inh[i, j] = beta

    # Populate A_ang and A_bif
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i == j:
                continue

            vect_i = seg_i.to_vect()
            vect_j = seg_j.to_vect()
            norm_i = np.linalg.norm(vect_i)
            norm_j = np.linalg.norm(vect_j)

            cosine = 0
            if norm_i > 0 and norm_j > 0:
                cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

            if np.isclose(cosine, 1, atol=epsilon): 
                A_ang[i, j] = 1

            if seg_i.from_hit.id == seg_j.from_hit.id and seg_i.to_hit.id != seg_j.to_hit.id:
                A_bif[i, j] = -alpha

            if seg_i.from_hit.id != seg_j.from_hit.id and seg_i.to_hit.id == seg_j.to_hit.id:
                A_bif[i, j] = -alpha

    A_ang = A_ang.tocsr()
    A_bif = A_bif.tocsr()
    A_inh = A_inh.tocsr()

    A_total = A_ang + A_bif + A_inh

    if verbose:
        print(f"[generate_hamiltonian] Hamiltonian matrix shape: {A_total.shape}")
        print(f"[generate_hamiltonian] Non-zero elements: {A_total.nnz}")

    b = np.zeros(N)

    return A_total, b, segments

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


#changed approach, more selective formation of tracks
def get_qubo_solutionEEE(sol_sample, event, segments):
    print("\n[get_qubo_solution] Processing QUBO solution...")
    print(f"[get_qubo_solution] sol_sample: {sol_sample}")
    print(f"[get_qubo_solution] Number of segments: {len(segments)}")
    
    active_segments = [segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state == 1]
    print(f"[get_qubo_solution] Active segments count: {len(active_segments)}")
    
    active = deepcopy(active_segments)
    tracks = []

    while active:
        # Start a new track with the last segment in the active list
        current_seg = active.pop()
        print(f"[get_qubo_solution] Starting new track with segment: {current_seg}")
        track_hits = set([current_seg.from_hit.id, current_seg.to_hit.id])
        extending = True

        while extending:
            extending = False
            next_segments = []
            for seg in active:
                # Check if this segment can be connected to the current track
                if seg.from_hit.id in track_hits:
                    next_hits = seg.to_hit.id
                    track_hits.add(next_hits)
                    tracks.append(set([seg.from_hit.id, seg.to_hit.id]))
                    next_segments.append(seg)
                    extending = True
                elif seg.to_hit.id in track_hits:
                    next_hits = seg.from_hit.id
                    track_hits.add(next_hits)
                    tracks.append(set([seg.from_hit.id, seg.to_hit.id]))
                    next_segments.append(seg)
                    extending = True
            #remove the connected segments from active
            for seg in next_segments:
                active.remove(seg)

        tracks.append(track_hits)
        print(f"[get_qubo_solution] Completed track with hits: {track_hits}")

    print(f"[get_qubo_solution] Generated {len(tracks)} tracks.")
    
    tracks_processed = []
    for idx, track_hit_ids in enumerate(tracks):
        try:
            hits = [next(filter(lambda b: b.id == a, event.hits)) for a in track_hit_ids]
            tracks_processed.append(track(hits))  
            print(f"[get_qubo_solution] Track {idx+1} processed: {hits}")
        except StopIteration as e:
            print(f"[get_qubo_solution] Error: Hit ID not found in event.hits for track {idx+1}: {e}")

    return tracks_processed


def visualize_tracks(reconstructed_tracks, ground_truth_particles, event, residuals):

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

    for rec_track, true_particle, residual in residuals:
        if true_particle is None or residual > 0:
            x = [hit.x for hit in rec_track.hits]
            y = [hit.y for hit in rec_track.hits]
            z = [hit.z for hit in rec_track.hits]
            ax.scatter(x, y, z, color="green", label="Mismatch Residual" if rec_track == residuals[0][0] else "")

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

import itertools
from copy import deepcopy
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import dimod
from dimod import BinaryQuadraticModel
from dwave.system import LeapHybridSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Assuming the Segment class and load_event function are defined elsewhere

def build_tracksTRYOUT(sol_sample, segments, min_segments=3, threshold=0.98):
    """
    Builds tracks from selected segments ensuring a minimum number of segments per track.
    """
    selected_segments = [seg for seg, active in zip(segments, sol_sample) if active]
    tracks = []
    used_segments = set()

    for seg in selected_segments:
        if seg in used_segments:
            continue

        current_track = [seg]
        used_segments.add(seg)

        # Extend the track forward
        next_segments = find_segmentsTRYOUT(seg, selected_segments, threshold=threshold)
        while next_segments:
            next_seg = next_segments[0]  # Choose the first compatible segment
            if next_seg in used_segments:
                break
            current_track.append(next_seg)
            used_segments.add(next_seg)
            seg = next_seg
            next_segments = find_segmentsTRYOUT(seg, selected_segments, threshold=threshold)

        # Extend the track backward
        prev_segments = find_segments_backwardsTRYOUT(current_track[0], selected_segments, threshold=threshold)
        while prev_segments:
            prev_seg = prev_segments[0]
            if prev_seg in used_segments:
                break
            current_track.insert(0, prev_seg)
            used_segments.add(prev_seg)
            current_track[0] = prev_seg
            prev_segments = find_segments_backwardsTRYOUT(current_track[0], selected_segments, threshold=threshold)

        if len(current_track) >= min_segments:
            tracks.append(current_track)

    return tracks
def find_segmentsTRYOUT(s0, active, threshold=0.98):
    """
    Finds segments that can be connected forward to the current segment based on cosine similarity.
    """
    found_s = []
    direction_s0 = s0.normalized_vect()
    for s1 in active:
        direction_s1 = s1.normalized_vect()
        cosine_similarity = np.dot(direction_s0, direction_s1)
        # Ensure that s1 starts where s0 ends and is highly aligned
        if (s0.to_hit.id == s1.from_hit.id) and (cosine_similarity > threshold):
            found_s.append(s1)
    return found_s

def find_segments_backwardsTRYOUT(s0, active, threshold=0.98):
    """
    Finds segments that can be connected backward to the current segment based on cosine similarity.
    """
    found_s = []
    direction_s0 = s0.normalized_vect()
    for s1 in active:
        direction_s1 = s1.normalized_vect()
        cosine_similarity = np.dot(direction_s0, direction_s1)
        # Ensure that s1 ends where s0 starts and is highly aligned
        if (s0.from_hit.id == s1.to_hit.id) and (cosine_similarity > threshold):
            found_s.append(s1)
    return found_s

def generate_hamiltonianCCC(event, params, verbose=True):
    """
    Generates the Hamiltonian matrix for the QUBO formulation based on the provided event and parameters.
    Integrates penalties to discourage ghost tracks and encourage longer, coherent tracks.
    """
    if verbose:
        print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    
    # Retrieve parameters with default values
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 20.0)  # Increased from 10.0
    beta = params.get('beta', 20.0)    # Increased from 10.0
    track_length_penalty = params.get('track_length_penalty', 5.0)
    two_hit_penalty = params.get('two_hit_penalty', 10.0)
    hit_overlap_penalty = params.get('hit_overlap_penalty', 15.0)
    epsilon = 1e-2  # Tolerance for cosine similarity
    
    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.z)
    if verbose:
        print(f"[generate_hamiltonian] Modules sorted by z-coordinate. Number of modules: {len(modules)}")
    
    # Generate all possible segments between consecutive modules
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_module = modules[idx + 1]
        hits_current = current_module.hits()
        hits_next = next_module.hits()
        for from_hit, to_hit in itertools.product(hits_current, hits_next):
            try:
                segment = Segment(from_hit, to_hit)
                if segment.length() == 0:
                    raise ValueError("Zero-length segment.")
                segments.append(segment)
            except ValueError as e:
                if verbose:
                    print(f"[generate_hamiltonian] Skipping invalid segment: {e}")
    
    N = len(segments)
    if verbose:
        print(f"[generate_hamiltonian] Total number of segments generated: {N}")
    
    if N == 0:
        if verbose:
            print("[generate_hamiltonian] No segments generated. Returning empty Hamiltonian.")
        return csr_matrix((0, 0)), np.zeros(0), segments
    
    # Initialize Hamiltonian components as sparse matrices
    A_ang = lil_matrix((N, N))
    A_bif = lil_matrix((N, N))
    A_inh = lil_matrix((N, N))
    
    # Module assignments for A_inh (inhibit overlapping segments)
    from_modules = np.array([seg.from_hit.module_number for seg in segments])
    to_modules = np.array([seg.to_hit.module_number for seg in segments])
    
    for i in range(N):
        for j in range(N):
            if i != j:
                shared = (to_modules[i] == from_modules[j]) or (from_modules[i] == to_modules[j])
                if shared: 
                    A_inh[i, j] = beta
    
    # Populate A_ang and A_bif
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i == j:
                continue

            vect_i = seg_i.to_vect()
            vect_j = seg_j.to_vect()
            norm_i = np.linalg.norm(vect_i)
            norm_j = np.linalg.norm(vect_j)

            cosine = 0
            if norm_i > 0 and norm_j > 0:
                cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

            if np.isclose(cosine, 1, atol=epsilon): 
                A_ang[i, j] = 1

            if (seg_i.from_hit.id == seg_j.from_hit.id and seg_i.to_hit.id != seg_j.to_hit.id) or \
               (seg_i.from_hit.id != seg_j.from_hit.id and seg_i.to_hit.id == seg_j.to_hit.id):
                A_bif[i, j] = -alpha
    
    # Convert to CSR format for efficient arithmetic operations
    A_ang = A_ang.tocsr()
    A_bif = A_bif.tocsr()
    A_inh = A_inh.tocsr()
    
    # Build the total Hamiltonian
    A_total = A_ang + A_bif + A_inh
    
    # Add Track Length Penalty: Encourage longer tracks
    for i in range(N):
        A_total[i, i] += track_length_penalty
    
    # Identify and Penalize Two-Hit Tracks
    two_hit_pairs = []
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i != j and seg_i.to_hit.id == seg_j.from_hit.id:
                two_hit_pairs.append((i, j))
    
    for i, j in two_hit_pairs:
        A_total[i, j] += two_hit_penalty
    
    # Add Hit Overlap Penalty: Prevent multiple segments from sharing the same hit
    hit_segment_map = {}
    for idx, seg in enumerate(segments):
        hit_segment_map.setdefault(seg.from_hit.id, []).append(idx)
        hit_segment_map.setdefault(seg.to_hit.id, []).append(idx)
    
    for hit_id, seg_indices in hit_segment_map.items():
        if len(seg_indices) > 1:
            for i in range(len(seg_indices)):
                for j in range(i + 1, len(seg_indices)):
                    A_total[seg_indices[i], seg_indices[j]] += hit_overlap_penalty
    
    if verbose:
        print(f"[generate_hamiltonian] Hamiltonian matrix shape: {A_total.shape}")
        print(f"[generate_hamiltonian] Non-zero elements: {A_total.nnz}")
    
    b = np.zeros(N)
    
    return A_total.tocsr(), b, segments



#combines everything, compares and plots correct and found solutions
def main():
    params = {
        'lambda': 5.0,
        'alpha': 50.0,
        'beta': 20.0
    }

    solutions = {
        "qubo_track_reconstruction": []
    }
    validation_data = []

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
            A, b, segments = generate_hamiltonianCCC(event_instance, params)
            visualize_segments(segments)
            plot_qubo_sparsity(A, title="QUBO Matrix Sparsity after Adjustments")
            plot_qubo_histogram(A, title="QUBO Coefficient Distribution after Adjustments")

        
            sol_sample = qubosolverHr(A, b)
            reconstructed_tracks = build_tracksTRYOUT(sol_sample, segments)
            print(f"[Main] Number of tracks reconstructed: {len(reconstructed_tracks)}")

            solutions["qubo_track_reconstruction"].append(reconstructed_tracks)
            validation_data.append(json_data)

            validator_event_instance = vl.parse_json_data(json_data)
            weights = vl.comp_weights(reconstructed_tracks, validator_event_instance)
            t2p, p2t = vl.hit_purity(reconstructed_tracks, validator_event_instance.particles, weights)


            residuals = calculate_residuals(reconstructed_tracks, validator_event_instance.particles)


            print("\n[Residual Analysis]")
            for rec_track, true_particle, residual in residuals:
                print(f"Reconstructed Track: {rec_track}")
                print(f"Associated Particle: {true_particle}")
                print(f"Residual: {residual}")

            visualize_tracks(reconstructed_tracks, validator_event_instance.particles, event_instance, residuals)

            #ghost_rate_value = vl.validate_ghost_fraction([json_data], [reconstructed_tracks])
            #clone_fraction = vl.validate_clone_fraction([json_data], [reconstructed_tracks])
            #reconstruction_efficiency = vl.validate_efficiency([json_data], [reconstructed_tracks])

            print("\n[Validation Metrics]")
            #print(f"Ghost Rate: {ghost_rate_value:.2%}")
            #print(f"Clone Fraction: {clone_fraction:.2%}")
            #print(f"Reconstruction Efficiency: {reconstruction_efficiency:.2%}")

    for k, v in sorted(solutions.items()):
        print(f"\n[Validation Summary] Validating tracks from {k}:")
        vl.validate_print(validation_data, v)
        print()

if __name__ == "__main__":
    main()