import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
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
#combines everything, compares and plots correct and found solutions
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
            sol_sample = qubosolverHr(A, b)
            reconstructed_tracks = get_qubo_solution(sol_sample, event_instance, segments)
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