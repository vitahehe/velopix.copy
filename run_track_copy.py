import os
import json
import numpy as np
from scipy.sparse import csc_matrix
from my_event_model import my_event_model as em  
from my_event_model import Segment
from validator import validator_lite as vl
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
import itertools
import dimod
from scipy.sparse import lil_matrix, csc_matrix, block_diag
import tracemalloc
from dwave.system import LeapHybridSampler
from copy import deepcopy 
from random import random, randint
from dimod.reference.samplers import ExactSolver
import dimod
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix
from joblib import Parallel, delayed

os.environ['DWAVE_API_TOKEN'] = 'DEV-21eed68bc845cad41711b2246f5765393f209d1f'

def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    results_ang = []
    results_bif = []
    
    vect_i = vectors[i]
    norm_i = norms[i]

    for j in range(i + 1, N):
        vect_j = vectors[j]
        norm_j = norms[j]
        cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, 1))

        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, -alpha))
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, -alpha))

    return results_ang, results_bif

def generate_hamiltonian_optimizedPAR(event, params):
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = sorted(event.modules, key=lambda a: a.z)

    # Generate segments
    segments = [
        Segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits(), modules[idx + 1].hits())
    ]
    
    N = len(segments)
    b = np.zeros(N)

    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)

    eps = 1e-9  

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

    A_ang = A_ang.tocsc()
    A_bif = A_bif.tocsc()

    # Inhibitory interactions
    module_ids_from = np.array([seg.from_hit.module_number for seg in segments])
    module_ids_to = np.array([seg.to_hit.module_number for seg in segments])
    A_inh = sp.csc_matrix((module_ids_from == module_ids_to[:, None]), dtype=int) * beta
    A = -1 * (A_ang + A_bif + A_inh)

    # Debug prints
    print(f"Hamiltonian matrix (A) shape: {A.shape}, non-zero elements: {A.nnz}")
    print(f"b vector: {b}")
    print(f"Segments count: {len(segments)}")

    return A, b, segments

def qubosolverHr(A, b):
    A = csc_matrix(A)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    
    row, col = A.nonzero()  # Get non-zero entries in the matrix A
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])

    # Debug prints
    print("Constructed Binary Quadratic Model:")
    print(bqm)

    # Use LeapHybridSampler to solve
    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.fromiter(best_sample.values(), dtype=int)
    
    # Debug prints
    print(f"Solver response: {response}")
    print(f"Best sample: {best_sample}")
    print(f"Solution Hybrid: {sol_sample}")

    return sol_sample

def find_segments(s0, active):
    found_s = []
    for s1 in active:
        if s0.from_hit.id == s1.to_hit.id or \
           s1.from_hit.id == s0.to_hit.id:
            found_s.append(s1)
    # Debug print
    print(f"find_segments: Found {len(found_s)} segments for segment {s0}")
    return found_s

def get_qubo_solution(sol_sample, event, segments):
    # Debug print
    print(f"sol_sample: {sol_sample}")
    print(f"Number of segments: {len(segments)}")

    active_segments = [segment for segment, pseudo_state in zip(segments, sol_sample) if pseudo_state > np.min(sol_sample)]
    print(f"Active segments count: {len(active_segments)}")
    active = deepcopy(active_segments)
    tracks = []

    while len(active):
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
        tracks.append(track)

    # Debug print
    print(f"Generated {len(tracks)} tracks.")

    tracks_processed = []
    for track in tracks:
        hits = [list(filter(lambda b: b.id == a, event.hits))[0] for a in track]
        tracks_processed.append(em.track(hits))
        print(f"Track processed: {hits}")

    return tracks_processed

params = {
    'lambda': 100.0,
    'alpha': 1.0,
    'beta': 1.0
}

solutions = {
    "quantum_annealing": []
}
validation_data = []


for dirpath, dirnames, filenames in os.walk("events"):
    for i, filename in enumerate(filenames):
        if i != 2:
            continue  


        with open(os.path.realpath(os.path.join(dirpath, filename)), 'r') as f:
            json_data = json.load(f)
            event = em.Event(json_data)



        # Generate Hamiltonian and solve it
        print(f"Processing event {i}: {filename}")
        A, b, segments = event.compute_hamiltonian(generate_hamiltonian_optimizedPAR, params)
        print(A.toarray())

        sol_sample = qubosolverHr(A, b)
        #qubosolverTABU(A, b)
        tracks = get_qubo_solution(sol_sample.tolist(), event, segments)

        solutions["quantum_annealing"].append(tracks)
        validation_data.append(json_data)

    

for k, v in sorted(solutions.items()):
    print(f"\nValidating tracks from {k}:")
    vl.validate_print(validation_data, v)
    print()

