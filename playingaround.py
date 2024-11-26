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
import dataclasses
from event_model import event_model as em
import third_event_model as tem


def find_segments(s0, active):
    found_s = []
    for s1 in active:
        if s0.hit_to.id == s1.hit_from.id or s0.hit_from.id == s1.hit_to.id:
            found_s.append(s1)
    return found_s


def get_qubo_solution(sol_sample, event, segments):
    #different active segments? 
    active_segments = [
        segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state == 1
    ]
    print(segments,"\n")
    print(active_segments)
    active = deepcopy(active_segments)
    tracks = []

    while active:
        s = active.pop()
        nextt = find_segments(s, active)
        track = set([s.hit_from.id, s.hit_to.id])


        while len(nextt):
            s = nextt.pop()
            try:
                active.remove(s)
            except ValueError:
                pass
            nextt += find_segments(s, active)
            track = track.union(set([s.hit_from.id, s.hit_to.id]))
        # List of tracks
        tracks.append(track)

    tracks_processed=[]
    for track in tracks:
        tracks_processed.append(em.track([list(filter(lambda b: b.id == a, event.hits))[0] for a in track]))
    print(f'\n tracks{tracks}')
    print(f'\n tracks_processed{tracks_processed}')

    return tracks_processed
#markoss tokan
os.environ['DWAVE_API_TOKEN'] = 'DEV-21eed68bc845cad41711b2246f5765393f209d1f'  

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

    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.array([best_sample[i] for i in range(len(b))], dtype=int)

    num_selected = np.sum(sol_sample)
    print(f"[QUBO Solver] Number of variables selected (value=1): {num_selected}")

    return sol_sample

import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed


def generate_hamiltonian(event, params):
    print("\n[generate_hamiltonian] Starting Hamiltonian generation...")
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = sorted(event.modules, key=lambda module: module.z)

    n_segments = 0
    segments_grouped = []
    segments = []
    segment_id = itertools.count()
    for idx in range(len(event.modules) - 1):
        from_hits = event.modules[idx].hits
        to_hits = event.modules[idx + 1].hits
        if not from_hits or not to_hits:
            continue 
        print(from_hits)
        print(to_hits)
        #assert False
        segments_group = []
        for hit_from, hit_to in itertools.product(from_hits, to_hits):
            seg = tem.Segment(next(segment_id), hit_from, hit_to)
            segments_group.append(seg)
            segments.append(seg)
            n_segments = n_segments + 1
        segments_grouped.append(segments_group)
    print(segments_grouped)
    #assert False
    N = len(segments)

    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))

    b = np.zeros(N)

    # Uncomment and correct if you intend to use A_inh
    # for i, seg in enumerate(segments):
        # if seg.from_hit.module_id == seg.to_hit.module_id:
            # A_inh[i, i] = beta
            # print(f"[Inh interaction] A_inh[{i}, {i}] = {A_inh[i, i]} (penalized for same module).")
            
    for i, seg_i in enumerate(segments):
        vect_i = seg_i.to_vect()
        norm_i = np.linalg.norm(vect_i)
        for j, seg_j in enumerate(segments):
            if i != j:
                vect_j = seg_j.to_vect()
                norm_j = np.linalg.norm(vect_j)

                #add back the norm, cos formula
                if norm_i == 0 or norm_j == 0:
                    cosine_similarity = 0
                else:
                    cosine_similarity = np.dot(vect_i, vect_j) / (norm_i * norm_j)
                eps = 1e-4

    
                if np.abs(cosine_similarity - 1) < eps:
                    A_ang[i, j] += -beta
                if (seg_i.hit_from.module_id == seg_j.hit_from.module_id) and (seg_i.hit_to.id != seg_j.hit_to.id):
                    A_bif[i, j] += alpha
                    #print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to {alpha} (bifurcation: same hit_from).")
                if (seg_i.hit_from.module_id != seg_j.hit_from.module_id) and (seg_i.hit_to.module_id == seg_j.hit_to.module_id):
                    A_bif[i, j] += alpha
                    #print(f"[generate_hamiltonian] A_bif[{i}, {j}] set to {alpha} (bifurcation: same to_hit).")

    A = lambda_val * (A_ang + A_bif)


    # Print Hamiltonian matrix statistics
    print(f"[generate_hamiltonian] Hamiltonian matrix A shape: {A.shape}")
    print(f"[generate_hamiltonian] Hamiltonian matrix A has {np.count_nonzero(A)} non-zero elements.")

    return A, b, segments

def plot_reconstructed_tracks(reconstructed_tracks):
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for track_idx, track in enumerate(reconstructed_tracks):
        x = [hit.x for hit in track.hits]
        y = [hit.y for hit in track.hits]
        z = [hit.z for hit in track.hits]
        
        ax.plot(x, y, z, label=f'Track {track_idx+1}')

    ax.set_title("Reconstructed Tracks")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


#combines everything, compares and plots correct and found solutions
def main():
    params = {
        'lambda': 1.0, #multiply at the end +
        'alpha': 1.0, #a_bif penelizes bifunctions -
        'beta': 1.0, #same module penalty A_inh -
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
            #restricted_modules_odd = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
            #restricted_modules = list(range(52))

            f = open(os.path.realpath(os.path.join(dirpath, filename)))
            event, json_data = vprem.restrict_event(json.loads(f.read()), restricted_modules_even, restrict_min_nb_hits, restrict_consec_modules)
            f.close()

            #ev = em.event(json_data)
            q_event = vp2q_event(event)
            
            print(f"[Main] Reconstructing event {i}...")
            A, b, segments = generate_hamiltonian(q_event, params)
        
        
            sol_sample = qubosolverHr(A, b)
            reconstructed_tracks = get_qubo_solution(sol_sample, q_event, segments)
            #print('\n', 'reconstructed tracks', reconstructed_tracks)

            solutions["qubo_track_reconstruction"].append(reconstructed_tracks)
            validation_data.append(json_data)
            #print(validation_data)
            #print(solutions)
            plot_reconstructed_tracks(reconstructed_tracks)

            validator_event_instance = vl.parse_json_data(json_data)

    for k, v in sorted(solutions.items()):
        print(f"\n[Validation Summary] Validating tracks from {k}:")
        vl.validate_print(validation_data, v)
        print()

if __name__ == "__main__":
    main()