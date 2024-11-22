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
    direction_s0 = s0.normalized_vect()
    for s1 in active:
        direction_s1 = s1.normalized_vect()
        cosine_similarity = np.dot(direction_s0, direction_s1)
        if s0.from_hit.id == s1.to_hit.id or s1.from_hit.id == s0.to_hit.id:
            if cosine_similarity > 1e-2:  #added condition: only connect if nearly aligned
                found_s.append(s1)
    return found_s

def get_qubo_solution(sol_sample, event, segments):
    print(f"sol_sample: {sol_sample}")
    print(f"Number of segments: {len(segments)}")
    
    #based on quboi 
    active_segments = [
        segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state > 0
    ]
    print(f"Active segments count: {len(active_segments)}")
    active = deepcopy(active_segments)
    tracks = []
    used_hits = set() #kepp track of hit ids

    while len(active):
        s = active.pop()
        track_hits = {s.from_hit.id, s.to_hit.id}
        used_hits.update(track_hits)

        #connected segments
        next_segments = find_segments(s, active)
        while len(next_segments):
            s = next_segments.pop()
            try:
                active.remove(s)
            except ValueError:
                pass

            #update the new segment
            track_hits.update([s.from_hit.id, s.to_hit.id])
            used_hits.update([s.from_hit.id, s.to_hit.id])
            next_segments += find_segments(s, active)

        #list of tracks
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

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, params):
    results_ang = []
    results_bif = []
    alpha = params.get('alpha')
    eps = 1e-2

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
            results_ang.append((i, j,-1))  # Add angular consistency interaction

        # Bifurcation consistency
        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, alpha))
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, alpha))

    return results_ang, results_bif


def generate_hamiltonian_real_data(event, params):
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

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
    eps = 1e-2
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, params)
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
        for j in range(N):  # Includes i == j for consistency with matrix dimension
            seg_j = segments[j]
            try:
                #Segment i originates from the same module segment j terminates
                if seg_i.from_hit == seg_j.from_hit or seg_i.to_hit == seg_j.to_hit:
                    A_inh[i, j] = beta
            except AttributeError as e:
                print(f"[generate_hamiltonian_real_data] AttributeError for segments {i}, {j}: {e}")
                print(f"Segment {i}: {seg_i}")
                print(f"Segment {j}: {seg_j}")

    A_inh = A_inh.tocsc()

    A = lambda_val * (A_ang + A_bif + A_inh)

    #Hamiltonian statistics
    print(f"[Hamiltonian Generation] Hamiltonian matrix A: shape {A.shape}, non-zero elements: {A.count_nonzero()}")

    return A, np.zeros(N), segments

def generate_hamiltonian(event, params):
    print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.module_number) #sorted by modules id instead of z coordinate

    print("[generate_hamiltonian] Modules deep-copied and sorted by module numbere")
    print(f"[generate_hamiltonian] Number of modules after sorting: {len(modules)}")

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

        print(f"Generated {len(segments)} segments.") #

    N = len(segments)
    print(f"[generate_hamiltonian] Total number of segments generated: {N}")

    if N == 0:
        print("[generate_hamiltonian] No segments generated. Returning empty Hamiltonian.")
        return np.zeros((0,0)), np.zeros(0), segments


    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    b = np.zeros(N)

    #is this correctly implemented?
    #for i, seg in enumerate(segments):
        #if seg.from_hit.module_number == seg.to_hit.module_number:
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

                eps = 1e-6

                # Populate A_ang if vectors are parallel
                if np.abs(cosine - 1) < eps:
                    A_ang[i, j] = -50
                    print(f"[generate_hamiltonian] A_ang[{i}, {j}] set to something negative (angular consistency).")

                # Populate A_bif for bifurcations
                if (seg_i.from_hit == seg_j.from_hit) and (seg_i.to_hit != seg_j.to_hit):
                    A_bif[i, j] = alpha
                    print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to {alpha} (bifurcation: same from_hit).")
                if (seg_i.from_hit != seg_j.from_hit) and (seg_i.to_hit == seg_j.to_hit):
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
                #check if this segment can be connected to the current track
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


def generate_hamiltonianCCC(event, params, verbose=True):
    if verbose:
        print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 20.0)  
    beta = params.get('beta', 20.0)    
    track_length_penalty = params.get('track_length_penalty', 5.0)
    two_hit_penalty = params.get('two_hit_penalty', 10.0)
    hit_overlap_penalty = params.get('hit_overlap_penalty', 15.0)
    epsilon = 1e-2  # Tolerance for cosine similarity
    
    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.z)
    if verbose:
        print(f"[generate_hamiltonian] Modules sorted by z-coordinate. Number of modules: {len(modules)}")
    
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
    
    #module assignments for A_inh (inhibit overlapping segments)
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
                A_ang[i, j] = -1

            if (seg_i.from_hit.id == seg_j.from_hit.id and seg_i.to_hit.id != seg_j.to_hit.id) or \
               (seg_i.from_hit.id != seg_j.from_hit.id and seg_i.to_hit.id == seg_j.to_hit.id):
                A_bif[i, j] = alpha
    
    # Convert to CSR format for efficient arithmetic operations
    A_ang = A_ang.tocsr()
    A_bif = A_bif.tocsr()
    A_inh = A_inh.tocsr()
    A_total = A_ang + A_bif + A_inh
    
    #encourage longer tracks
    for i in range(N):
        A_total[i, i] += track_length_penalty
    
    #penalize Two-Hit Tracks
    two_hit_pairs = []
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i != j and seg_i.to_hit.id == seg_j.from_hit.id:
                two_hit_pairs.append((i, j))
    
    for i, j in two_hit_pairs:
        A_total[i, j] += two_hit_penalty
    
    #prevent multiple segments from sharing the same hit
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

            file_path = os.path.realpath(os.path.join(dirpath, filename))
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            event_instance = event(json_data)
            print(f"\n[Main] Processing event {filename}")
            print(f"[Main] Number of modules in event: {len(event_instance.modules)}")
            total_hits = sum(len(list(m)) for m in event_instance.modules)
            print(f"[Main] Total number of hits in event: {total_hits}")

            q_event_instance = vp2q_event(event_instance)
            
            print(f"[Main] Reconstructing event {i}...")
            A, b, segments = generate_hamiltonian(q_event_instance, params)
            visualize_segments(segments)
            plot_qubo_sparsity(A, title="QUBO Matrix Sparsity after Adjustments")
            #plot_qubo_histogram(A, title="QUBO Coefficient Distribution after Adjustments")

        
            sol_sample = qubosolverHr(A, b)
            reconstructed_tracks = get_qubo_solution(sol_sample, q_event_instance, segments)
            print(f"[Main] Number of tracks reconstructed: {len(reconstructed_tracks)}")

            solutions["qubo_track_reconstruction"].append(reconstructed_tracks)
            validation_data.append(json_data)

            validator_event_instance = vl.parse_json_data(json_data)
            weights = vl.comp_weights(reconstructed_tracks, validator_event_instance)
            t2p, p2t = vl.hit_purity(reconstructed_tracks, validator_event_instance.particles, weights)
            visualize_truth(validator_event_instance.particles)
            visualize_reconstructed_tracks(reconstructed_tracks)
            visualize_tracks(reconstructed_tracks, validator_event_instance.particles)

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