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

        # Angular consistency: encourage these segments to be selected together
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, 1))  # Positive value in A_ang

        # Bifurcation consistency: penalize these segments being selected together
        seg_i, seg_j = segments[i], segments[j]
        if ((seg_i.from_hit.module_number == seg_j.from_hit.module_number and
             seg_i.to_hit.module_number != seg_j.to_hit.module_number) or
            (seg_i.from_hit.module_number != seg_j.from_hit.module_number and
             seg_i.to_hit.module_number == seg_j.to_hit.module_number)):
            results_bif.append((i, j, alpha))  # Positive value in A_bif

    return results_ang, results_bif

def generate_hamiltonian_optimizedPAR(event, params):
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)

    total_modules = event.number_of_modules

    module_dict = {module.module_number: module for module in event.modules}

    segments = []

    print("Constructing segments between consecutive modules with hits...")
    for m_num in range(total_modules - 1):
        from_module = module_dict.get(m_num)
        to_module = module_dict.get(m_num + 1)

        # Debug: Print module numbers and check if they have hits
        print(f"Processing modules {m_num} and {m_num + 1}")
        if not from_module or not to_module:
            print(f"  One of the modules {m_num} or {m_num + 1} does not exist.")
            continue

        from_hits = from_module.hits()
        to_hits = to_module.hits()

        # Debug: Print hits in modules
        print(f"  Module {m_num} hits: {[hit.id for hit in from_hits]}")
        print(f"  Module {m_num + 1} hits: {[hit.id for hit in to_hits]}")

        if not from_hits or not to_hits:
            print(f"  One of the modules {m_num} or {m_num + 1} has no hits.")
            continue

        for from_hit, to_hit in itertools.product(from_hits, to_hits):
            segments.append(Segment(from_hit, to_hit))

    print(f"Total Segments Constructed: {len(segments)}")

    N = len(segments)
    b = np.zeros(N)

    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)

    print("Calculating angular and bifurcation checks...")
    eps = 1e-2
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, alpha, eps)
        for i in range(N)
    )

    # ... [rest of your code] ...

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

    # Inhibition term (positive penalties for shared hits)
    hit_ids_from = np.array([seg.from_hit.id for seg in segments])
    hit_ids_to = np.array([seg.to_hit.id for seg in segments])
    shared_hits = (hit_ids_from[:, None] == hit_ids_from[None, :]) | (hit_ids_to[:, None] == hit_ids_to[None, :])
    A_inh = sp.csc_matrix(shared_hits, dtype=int) * beta

    # Construct Hamiltonian: Penalize bifurcations and shared hits, encourage angular consistency
    A = A_inh + A_bif - A_ang

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

def generate_hamiltonian_corrected(event, params):
    lambda_val = params.get('lambda', 1.0)
    alpha = params.get('alpha', 1.0)  # For rewards
    beta = params.get('beta', 1.0)    # For penalties
    gamma = params.get('gamma', 1.0)
    delta = params.get('delta', 1.0)

    modules = sorted(event.modules, key=lambda a: a.z)

    # Generate segments between consecutive modules
    segments = [
        Segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits(), modules[idx + 1].hits())
    ]

    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)

    # Initialize matrices
    N = len(segments)
    A_ang = dok_matrix((N, N), dtype=np.float64)
    A_bif = dok_matrix((N, N), dtype=np.float64)
    A_inh = dok_matrix((N, N), dtype=np.float64)

    # Angular consistency rewards
    for i in range(N):
        vect_i = vectors[i]
        norm_i = norms[i]
        seg_i = segments[i]
        for j in range(i + 1, N):
            vect_j = vectors[j]
            norm_j = norms[j]
            seg_j = segments[j]
            cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

            if np.abs(cosine - 1) < eps:
                A_ang[i, j] = -alpha  # Negative reward
                A_ang[j, i] = -alpha

            # Bifurcation penalties
            if (seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit) or \
               (seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit):
                A_bif[i, j] = beta
                A_bif[j, i] = beta

            # Inhibition penalties
            if (seg_i.from_hit == seg_j.from_hit) or (seg_i.to_hit == seg_j.to_hit):
                A_inh[i, j] = beta
                A_inh[j, i] = beta

    # Combine matrices into the Hamiltonian
    A = A_ang + A_bif + A_inh

    # Set diagonal entries
    A.setdiag(gamma * np.ones(N))

    # Linear term b
    b = delta * np.ones(N)

    return A, b, segments
eps= 1e-2

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
        A, b, segments = event.compute_hamiltonian(generate_hamiltonian_corrected, params)
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

