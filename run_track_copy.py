import os
import json
import numpy as np
from scipy.sparse import csc_matrix
from my_event_model import my_event_model as em  
from my_event_model import Segment
from validator import validator_lite as vl
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
import psutil
import time
import itertools
import dimod
import psutil
from scipy.sparse import lil_matrix, csc_matrix, block_diag
import tracemalloc
from dwave.system import LeapHybridSampler
from copy import deepcopy 
from random import random, randint
from dimod.reference.samplers import ExactSolver

os.environ['DWAVE_API_TOKEN'] = 'DEV-031741f1792495e220d4d55aeb72ab7961cc16cd'

def qubosolverCl(A, b):
    """
    Classical QUBO solver using dimod's ExactSolver for deterministic results.
    """
    A = csc_matrix(A)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    
    row, col = A.nonzero()  # Get non-zero entries in the matrix A
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])

    # Use ExactSolver to solve
    sampler = ExactSolver()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.fromiter(best_sample.values(), dtype=int)
    
    print(f"Solution Classical: {sol_sample}")
    return sol_sample

def find_segments(s0, active):
       found_s = []
       for s1 in active:
           if s0.from_hit.id == s1.to_hit.id or \
           s1.from_hit.id == s0.to_hit.id:
               found_s.append(s1)
       return found_s

def get_qubo_solution(sol_sample, event, segments):
   active_segments = [segment for segment,pseudo_state in zip(segments,sol_sample) if pseudo_state > np.min(sol_sample)]
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
           except:
               pass
           nextt += find_segments(s, active)
           track = track.union(set([s.from_hit.id, s.to_hit.id]))
       tracks.append(track)
   tracks_processed = []
   for track in tracks:
       tracks_processed.append(em.track([list(filter(lambda b: b.id == a, event.hits))[0] for a in track]))
   return tracks_processed

# Hamiltonian generator and solver function
def generate_hamiltonian_optimized(event, params):
    lambda_val = params.get('lambda', 100.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)

    modules = sorted(event.modules, key=lambda module: module.z)
    print("Modules sorted by z-coordinate.")

    segments = []
    for idx in range(len(modules) - 1):
        module_from = modules[idx]
        module_to = modules[idx + 1]
        # Optionally, print module indices and z-values
        print(f"Creating segments between Module {module_from.module_number} (z={module_from.z}) "
              f"and Module {module_to.module_number} (z={module_to.z}).")
        for from_hit, to_hit in itertools.product(module_from.hits(), module_to.hits()):
            segments.append(Segment(from_hit, to_hit))

    N = len(segments)
    print("Number of segments created:", N)

    A_ang_blocks = []
    A_bif_blocks = []
    A_inh_blocks = []
    b = np.zeros(N)

    block_size = 500
    num_blocks = (N + block_size - 1) // block_size
    print("Number of blocks:", num_blocks)

    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, N)
        print(f"Processing block {block_idx + 1}/{num_blocks} "
              f"(segments {start_idx} to {end_idx - 1}).")

        # lil_matrix for each block
        size = end_idx - start_idx
        A_ang_block = lil_matrix((size, size), dtype=np.float32)
        A_bif_block = lil_matrix((size, size), dtype=np.float32)
        A_inh_block = lil_matrix((size, size), dtype=np.float32)

        # Filling
        for i in range(start_idx, end_idx):
            seg_i = segments[i]
            vect_i = seg_i.to_vect()
            norm_i = np.linalg.norm(vect_i)
            if norm_i == 0:
                print(f"Zero-length vector encountered at segment {i}. Skipping.")
                continue  # Skip zero-length vectors

            for j in range(i + 1, end_idx):
                seg_j = segments[j]
                vect_j = seg_j.to_vect()
                norm_j = np.linalg.norm(vect_j)
                if norm_j == 0:
                    continue 

                # Avoid division by zero
                denominator = norm_i * norm_j
                if denominator == 0:
                    continue

                cosine = np.dot(vect_i, vect_j) / denominator


                if np.abs(cosine - 1) < 1e-9:
                    A_ang_block[i - start_idx, j - start_idx] = 1
                    A_ang_block[j - start_idx, i - start_idx] = 1  # Symmetry with positive sign

                if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
                    A_bif_block[i - start_idx, j - start_idx] = -alpha
                    A_bif_block[j - start_idx, i - start_idx] = -alpha 

                if seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
                    A_bif_block[i - start_idx, j - start_idx] = -alpha
                    A_bif_block[j - start_idx, i - start_idx] = -alpha  # Symmetry with negative sign

                s_ab = int(seg_i.from_hit.module_id == 1 and seg_j.to_hit.module_id == 1)
                if s_ab > 0:
                    A_inh_block[i - start_idx, j - start_idx] = beta * s_ab * s_ab
                    A_inh_block[j - start_idx, i - start_idx] = beta * s_ab * s_ab  # Symmetry with positive sign

        A_ang_blocks.append(A_ang_block)
        A_bif_blocks.append(A_bif_block)
        A_inh_blocks.append(A_inh_block)

    A_ang = block_diag(A_ang_blocks, format='csc')
    A_bif = block_diag(A_bif_blocks, format='csc')
    A_inh = block_diag(A_inh_blocks, format='csc')

    A = -1 * (A_ang + A_bif + A_inh)

    print("Hamiltonian matrices constructed.")
    print("Shape of A_ang:", A_ang.shape)
    print("Shape of A_bif:", A_bif.shape)
    print("Shape of A_inh:", A_inh.shape)
    print("Shape of A:", A.shape)

    return A, b, segments

import dimod
import psutil
import time
from scipy.sparse import csc_matrix

def qubosolver(A, b):

    #performance measurement
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    start_time = time.time()

    #Keep A sparse
    A = csc_matrix(A)
    print('sucessfully matrix in csc')

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    #vectors for efficiency 
    bqm.add_variables_from({i: b[i] for i in range(len(b))})

    row, col = A.nonzero() 

    print('test2') 
    for i, j in zip(row, col):
        if i != j:  
            bqm.add_interaction(i, j, A[i, j])
    print(np.shape(A.toarray()))
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=100)
    print('test4')

    best_sample = response.first.sample

    print('test5')
    sol_sample = np.fromiter(best_sample.values(), dtype=int)  

    end_memory = process.memory_info().rss / (1024 ** 2) 
    end_time = time.time()

   
    memory_used = end_memory - start_memory
    time_taken = end_time - start_time

    print(f"Solution Simulated Annealing:{sol_sample}")
    print(f"Memory {memory_used:.2f} MB")
    print(f"Time {time_taken:.6f} seconds")

    return sol_sample

def qubosolverHr(A, b):
    A = csc_matrix(A)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    
    row, col = A.nonzero()  # Get non-zero entries in the matrix A
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])

    # Use LeapHybridSampler to solve
    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol_sample = np.fromiter(best_sample.values(), dtype=int)
    
    print(f"Solution Hybrid: {sol_sample}")
    return sol_sample

# Parameters for Hamiltonian generation
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
            continue  # Skip events other than i=4

        # Load event 4
        with open(os.path.realpath(os.path.join(dirpath, filename)), 'r') as f:
            json_data = json.load(f)
            event = em.Event(json_data)

        # Generate Hamiltonian and solve it
        print(f"Processing event {i}: {filename}")
        A, b, segments = event.compute_hamiltonian(generate_hamiltonian_optimized, params)
        print(A.toarray())

        sol_sample = qubosolverHr(A, b)
        sol
        assert False 
        tracks = get_qubo_solution(sol_sample.tolist(), event, segments)
        print(tracks)
        print(tracks.pop())
        print(type(tracks.pop()))

        solutions["quantum_annealing"].append(tracks)
        validation_data.append(json_data)

    

for k, v in sorted(solutions.items()):
    print(f"\nValidating tracks from {k}:")
    vl.validate_print(validation_data, v)
    print()
