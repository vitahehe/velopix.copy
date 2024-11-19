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
from copy import deepcopy


os.environ['DWAVE_API_TOKEN'] = 'DEV-b59f413d6a1407427e9f0079dd8e3cfb8106e58d'  

def qubosolverHr(A, b):
    print("\n[QUBO Solver] Starting QUBO Solver...")
    A = csc_matrix(A)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    linear_terms = {i: b[i] for i in range(len(b))}
    bqm.add_variables_from(linear_terms)
    print(f"[QUBO Solver] Added {len(linear_terms)} linear terms.")

    row, col = A.nonzero()
    print(f"[QUBO Solver] Non-zero interactions found: {len(row)}")
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])
        else:
            bqm.add_variable(i, A[i, j])
    
    print(f"[QUBO Solver] BQM has {len(bqm.variables)} variables and {len(bqm.quadratic)} interactions.")

    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.array([best_sample.get(i, 0) for i in range(len(b))], dtype=int)

    # Solution statistics
    num_selected = np.sum(sol_sample)
    print(f"[QUBO Solver] Number of variables selected (value=1): {num_selected}")

    return sol_sample

#not in the event_model
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

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    results_ang = []
    results_bif = []
    
    vect_i = vectors[i]
    norm_i = norms[i]

    for j in range(i + 1, N):  
        vect_j = vectors[j]
        norm_j = norms[j]
        cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j) if (norm_i != 0 and norm_j != 0) else 0

        # Angular consistency
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, -1))
            print(f"[angular_and_bifurcation_checks] Angular consistency between segments {i} and {j}.")

        # Bifurcation consistency
        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, alpha))
            print(f"[angular_and_bifurcation_checks] Bifurcation detected between segments {i} and {j} (same from_hit).")
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, alpha))
            print(f"[angular_and_bifurcation_checks] Bifurcation detected between segments {i} and {j} (same to_hit).")

    return results_ang, results_bif

def generate_hamiltonian_real_data(event, params):
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)

    # Sort modules by their z-coordinate
    modules = sorted(event.modules, key=lambda m: m.z)
    print("[generate_hamiltonian_real_data] Modules sorted by z-coordinate.")
    print(f"[generate_hamiltonian_real_data] Number of modules: {len(modules)}")

    # Generate segments skipping over modules without hits
    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_idx = idx + 1
        while next_idx < len(modules) and not modules[next_idx].hits():
            print(f"[generate_hamiltonian_real_data] Skipping module {next_idx} due to no hits.")
            next_idx += 1

        # No valid next module with hits, just break
        if next_idx >= len(modules):
            print(f"[generate_hamiltonian_real_data] No more modules with hits after module {idx}.")
            break

        hits_current = current_module.hits()
        hits_next = modules[next_idx].hits()

        # Skip if no hits on a module
        if not hits_current:
            print(f"[generate_hamiltonian_real_data] No hits in current module {idx}. Skipping.")
            continue

        # Segments between hits in the current and next module
        new_segments = list(itertools.product(hits_current, hits_next))
        segments.extend(Segment(from_hit, to_hit) for from_hit, to_hit in new_segments)
        print(f"[generate_hamiltonian_real_data] Generated {len(new_segments)} segments between module {idx} and {next_idx}.")

    N = len(segments)
    print(f"[generate_hamiltonian_real_data] Total number of segments generated: {N}")

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
                print(f"[generate_hamiltonian_real_data] Inherent interaction between segments {i} and {j}.")

    A_inh = A_inh.tocsc()

    A = lambda_val * (A_ang + A_bif + A_inh)
    print(f"[generate_hamiltonian_real_data] Hamiltonian matrix A: shape {A.shape}, non-zero elements: {np.count_nonzero(A)}")

    return A, np.zeros(N), segments

import numpy as np
import itertools
from copy import deepcopy


def generate_hamiltonian(event, params):
    print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = deepcopy(event.modules)
    modules.sort(key=lambda module: module.z)
    print("[generate_hamiltonian] Modules deep-copied and sorted by z-coordinate.")
    print(f"[generate_hamiltonian] Number of modules after sorting: {len(modules)}")


    segments = []
    for idx in range(len(modules) - 1):
        current_module = modules[idx]
        next_module = modules[idx + 1]
        hits_current = current_module.hits()
        hits_next = next_module.hits()
        segments_generated = 0
        for from_hit, to_hit in itertools.product(hits_current, hits_next):
            try:
                segment = Segment(from_hit, to_hit)
                segments.append(segment)
                segments_generated += 1
            except ValueError as e:
                print(f"[generate_hamiltonian] Skipping invalid segment between Hit {from_hit.id} and Hit {to_hit.id}: {e}")
        print(f"[generate_hamiltonian] Generated {segments_generated} segments between module {idx} and module {idx + 1}.")

    N = len(segments)
    print(f"[generate_hamiltonian] Total number of segments generated: {N}")

    if N == 0:
        print("[generate_hamiltonian] No segments generated. Returning empty Hamiltonian.")
        return np.zeros((0,0)), np.zeros(0), segments


    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    
    b = np.zeros(N)
    s_ab = np.zeros((N, N), dtype=int)
    print("[generate_hamiltonian] Initializing s_ab matrix based on module numbers.")
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            try:
                s_ab[i, j] = int(seg_i.from_hit.module_number == 1 and seg_j.to_hit.module_number == 1)
            except AttributeError as e:
                print(f"[generate_hamiltonian] AttributeError for segments {i}, {j}: {e}")
                print(f"Segment {i}: {seg_i}")
                print(f"Segment {j}: {seg_j}")

    #debugg
    print("[generate_hamiltonian] Sample s_ab matrix entries:")
    for i in range(min(N, 3)):
        for j in range(min(N, 3)):
            print(f"s_ab[{i}, {j}] = {s_ab[i, j]}")

    #apply conditions
    print("[generate_hamiltonian] Populating A_ang, A_bif, and A_inh matrices.")
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

                eps = 1e-9

                # Populate A_ang if vectors are parallel
                if np.abs(cosine - 1) < eps:
                    A_ang[i, j] = 1
                    print(f"[generate_hamiltonian] A_ang[{i}, {j}] set to 1 (angular consistency).")

                # Populate A_bif for bifurcations
                if (seg_i.from_hit == seg_j.from_hit) and (seg_i.to_hit != seg_j.to_hit):
                    A_bif[i, j] = -alpha
                    print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to -{alpha} (bifurcation: same from_hit).")
                if (seg_i.from_hit != seg_j.from_hit) and (seg_i.to_hit == seg_j.to_hit):
                    A_bif[i, j] = -alpha
                    print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to -{alpha} (bifurcation: same to_hit).")

                # Populate A_inh based on s_ab
                A_inh[i, j] = s_ab[i, j] * s_ab[j, i] * beta
                if A_inh[i, j] != 0:
                    print(f"[generate_hamiltonian] A_inh[{i}, {j}] set to {A_inh[i, j]} (inherent interaction).")

    # Compute the final Hamiltonian matrix
    A = -1 * (A_ang + A_bif + A_inh)
    print("[generate_hamiltonian] Combined A_ang, A_bif, and A_inh into final Hamiltonian matrix A.")

    # Store components for reference
    components = {'A_ang': -A_ang, 'A_bif': -A_bif, 'A_inh': -A_inh}

    # Print Hamiltonian matrix statistics
    print(f"[generate_hamiltonian] Hamiltonian matrix A shape: {A.shape}")
    print(f"[generate_hamiltonian] Hamiltonian matrix A has {np.count_nonzero(A)} non-zero elements.")

    return A, b, segments

def visualize_segments(segments):
    print("[visualize_segments] Visualizing segments...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Constructed Segments")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for idx, seg in enumerate(segments):
        x = [seg.from_hit.x, seg.to_hit.x]
        y = [seg.from_hit.y, seg.to_hit.y]
        z = [seg.from_hit.z, seg.to_hit.z]
        ax.plot(x, y, z, color="purple", linestyle="-", label="Segment" if idx == 0 else "")

    ax.legend()
    plt.show()
    print("[visualize_segments] Visualization complete.")

def find_segments(s0, active):
    found_s = []
    for s1 in active:
        if s0.from_hit.id == s1.to_hit.id or \
           s1.from_hit.id == s0.to_hit.id:
            found_s.append(s1)
    print(f"find_segments: Found {len(found_s)} segments for segment {s0}")
    return found_s

def get_qubo_solution(sol_sample, event, segments):
    print("\n[get_qubo_solution] Processing QUBO solution...")
    print(f"[get_qubo_solution] sol_sample: {sol_sample}")
    print(f"[get_qubo_solution] Number of segments: {len(segments)}")
    
    # Selecting segments where sol_sample is 1
    active_segments = [segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state == 1]
    print(f"[get_qubo_solution] Active segments count: {len(active_segments)}")
    
    active = deepcopy(active_segments)
    tracks = []

    while len(active):
        s = active.pop()
        print(f"[get_qubo_solution] Starting new track with segment: {s}")
        nextt = find_segments(s, active)
        track_hits = set([s.from_hit.id, s.to_hit.id])
        print(f"[get_qubo_solution] Initial track hits: {track_hits}")

        while len(nextt):
            s = nextt.pop()
            try:
                active.remove(s)
                print(f"[get_qubo_solution] Added segment to track: {s}")
            except ValueError:
                print(f"[get_qubo_solution] Segment {s} not found in active list.")
                pass
            nextt += find_segments(s, active)
            track_hits = track_hits.union(set([s.from_hit.id, s.to_hit.id]))
            print(f"[get_qubo_solution] Updated track hits: {track_hits}")

        tracks.append(track_hits)
        print(f"[get_qubo_solution] Completed track with hits: {track_hits}")

    print(f"[get_qubo_solution] Generated {len(tracks)} tracks.")
    
    # Convert hit IDs to track objects
    tracks_processed = []
    for idx, track_hit_ids in enumerate(tracks):
        try:
            hits = [next(filter(lambda b: b.id == a, event.hits)) for a in track_hit_ids]
            tracks_processed.append(track(hits))  
            print(f"[get_qubo_solution] Track {idx+1} processed: {hits}")
        except StopIteration as e:
            print(f"[get_qubo_solution] Error: Hit ID not found in event.hits for track {idx+1}: {e}")

    return tracks_processed

def main():
    print("[Main] Starting main function...")
    params = {
        'lambda': 1.0,
        'alpha': 1.0,
        'beta': 1.0
    }

    solutions = {
        "qubo_track_reconstruction": []
    }
    validation_data = []

    event_files_processed = 0

    for (dirpath, dirnames, filenames) in os.walk("events"):
        for i, filename in enumerate(filenames):
            if i != 2:
               print(f"[Main] Skipping file {i}: {filename}")
               continue  # Processes only the third file (0-based indexing)
            file_path = os.path.realpath(os.path.join(dirpath, filename))
            print(f"\n[Main] Loading event from file: {file_path}")
            with open(file_path, 'r') as f:
                try:
                    json_data = json.load(f)
                    print(f"[Main] Successfully loaded JSON data for {filename}.")
                except json.JSONDecodeError as e:
                    print(f"[Main] Error decoding JSON from {filename}: {e}")
                    continue

            try:
                event_instance = event(json_data)
                print(f"[Main] Event instance created successfully.")
            except Exception as e:
                print(f"[Main] Error initializing Event for {filename}: {e}")
                continue

            print(f"\n[Main] Processing event {filename}")
            print(f"[Main] Number of modules in event: {len(event_instance.modules)}")
            total_hits = sum(len(list(m)) for m in event_instance.modules)
            print(f"[Main] Total number of hits in event: {total_hits}")

            print(f"[Main] Reconstructing event {i}...")
            try:
                A, b, segments = generate_hamiltonian(event_instance, params)
                print(f"[Main] Hamiltonian generated with shape {A.shape} and {np.count_nonzero(A)} non-zero elements.")
                if len(segments) > 0:
                    print(f"[Main] Number of segments: {len(segments)}")
                    print("[Main] Sample segments:")
                    for seg in segments[:5]:
                        print(seg)
                else:
                    print("[Main] No segments generated.")
            except Exception as e:
                print(f"[Main] Error generating Hamiltonian for {filename}: {e}")
                continue

            print("[Main] Visualizing segments...")
            try:
                visualize_segments(segments)
            except Exception as e:
                print(f"[Main] Error visualizing segments for {filename}: {e}")

            # Use for debugging the optimized Hamiltonian
            # if A.count_nonzero() == 0:
                # print("[Main] Hamiltonian matrix A is empty. Skipping event.")
                # continue

            try:
                sol_sample = qubosolverHr(A, b)
                print("[Main] QUBO Solver completed.")
            except Exception as e:
                print(f"[Main] Error during QUBO solving for {filename}: {e}")
                continue

            try:
                tracks = get_qubo_solution(sol_sample, event_instance, segments)
                print(f"[Main] Number of tracks reconstructed: {len(tracks)}")
                solutions["qubo_track_reconstruction"].append(tracks)
                validation_data.append(json_data)
                event_files_processed += 1
            except Exception as e:
                print(f"[Main] Error processing QUBO solution for {filename}: {e}")
                continue

    if event_files_processed == 0:
        print("[Main] Warning: No events were successfully processed.")
    else:
        for k, v in sorted(solutions.items()):
            print(f"\n[Validation] Validating tracks from {k}:")
            vl.validate_print(validation_data, v)
            print()

    print(f"\n[Main] Total events processed: {event_files_processed}")

if __name__ == "__main__":
    main()
