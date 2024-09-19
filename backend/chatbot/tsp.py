import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.applications import Tsp
from qiskit_optimization import QiskitOptimizationError
from qiskit_algorithms import NumPyMinimumEigensolver, SamplingVQE
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_optimization.converters import QuadraticProgramToQubo


# algorithm_globals.random_seed = 123

class TSPSolver:
    # distances should be a list of tuples such that (i,j,weight) where (i,j) is the edge
    def __init__(self, num_nodes, distances=None, seed=None):
        self.num_nodes = num_nodes
        self.distances = distances
        self.adj_matrix = None
        self.seed = seed
        self.tsp_instance = self.create_tsp_instance()
        self.quadratic_problem = self.tsp_to_quadratic_program()
        self.qubo = None
        self.pauli_decomposition = None
        self.offset = None
        self.ansatz = None
        self.optimizer = SPSA(maxiter=300)

    def create_tsp_instance(self):
        if self.distances:
            tsp_graph = nx.Graph()
            tsp_graph.add_nodes_from(np.arange(0, self.num_nodes, 1))
            tsp_graph.add_weighted_edges_from(self.distances)
            self.adj_matrix = nx.to_numpy_array(tsp_graph)
            return Tsp(tsp_graph)
        else:
            return Tsp.create_random_instance(self.num_nodes, seed=self.seed)

    def tsp_to_quadratic_program(self):
        return self.tsp_instance.to_quadratic_program()

    def tsp_to_ising(self):
        qp2qubo = QuadraticProgramToQubo()
        self.qubo = qp2qubo.convert(self.quadratic_problem)
        self.pauli_decomposition, self.offset = self.qubo.to_ising()
        self.ansatz = TwoLocal(self.pauli_decomposition.num_qubits, "ry", "cz", reps=5, entanglement="linear")

    def solve_classically(self):
        cl_solver = NumPyMinimumEigensolver()
        classical_optimizer = MinimumEigenOptimizer(cl_solver)
        try:
            classical_result = classical_optimizer.solve(self.quadratic_problem)
            return classical_result
        except QiskitOptimizationError:
            print(f"Problem cannot be solved with the optimizer implementing this method.")
            return None

    def solve_quantum_with_MinimumEigenOptimizer(self):
        self.tsp_to_ising()
        vqe = SamplingVQE(sampler=Sampler(), ansatz=self.ansatz, optimizer=self.optimizer)

        vqe_optimizer = MinimumEigenOptimizer(vqe)

        quantum_result = vqe_optimizer.solve(self.quadratic_problem)

        solution = {'q_state': quantum_result,
                    'list_nodes': quantum_result.variables_dict,
                    'optimal_value': quantum_result.fval}

        return solution

    def solve_quantum(self):
        print("Looking for a quantum solution..")
        self.tsp_to_ising()
        vqe = SamplingVQE(sampler=Sampler(), ansatz=self.ansatz, optimizer=self.optimizer)

        # Starts the part which is computed by the MinimumEigenOptimizer
        q_state = vqe.compute_minimum_eigenvalue(self.pauli_decomposition)

        cl_most_likely_result = self.tsp_instance.sample_most_likely(q_state.eigenstate)
        is_feasible = self.qubo.is_feasible(cl_most_likely_result)

        if is_feasible:
            list_nodes = self.tsp_instance.interpret(cl_most_likely_result)
            solution = {'q_state': q_state,
                        'is_feasible': is_feasible,
                        'list_nodes': list_nodes,
                        'optimal_value': self.tsp_instance.tsp_value(list_nodes, self.adj_matrix)}
        else:
            solution = {'q_state': q_state,
                        'is_feasible': is_feasible,
                        'list_nodes': None,
                        'optimal_value': None}

        return solution


if __name__ == "__main__":
    num_nodes = 3
    nodes_labels = ("A", "B", "C")
    distances = [(0, 1, 5), (1, 2, 8), (2, 0, 2)]
    # distances = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    seed = 123

    tsp = TSPSolver(num_nodes, distances, seed=seed)
    tsp.create_tsp_instance()
    tsp.tsp_instance.draw()
    qp = tsp.tsp_instance.to_quadratic_program()
    print(qp.prettyprint())
    cl_result = tsp.solve_classically()
    print(cl_result.prettyprint())
    plt.show()

    quantum_solution = tsp.solve_quantum()

    print("energy:", quantum_solution['q_state'].eigenvalue.real)
    print("time:", quantum_solution['q_state'].optimizer_time)
    print("feasible:", quantum_solution['is_feasible'])
    print("solution:", quantum_solution['list_nodes'])
    print("solution objective:", quantum_solution['optimal_value'])

    quantum_solution_with_MinimumEigenOptimizer = tsp.solve_quantum_with_MinimumEigenOptimizer()

    print(quantum_solution_with_MinimumEigenOptimizer['q_state'].prettyprint())

    print("solution:", quantum_solution_with_MinimumEigenOptimizer['list_nodes'])
    print("solution objective:", quantum_solution_with_MinimumEigenOptimizer['optimal_value'])
