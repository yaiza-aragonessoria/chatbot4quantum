# Code to show in C4Q

# import qiskit as qiskit
# from matplotlib.backends.backend_agg import FigureCanvasAgg
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
# import numpy as np
# import math
# import io

# Initialise the state
state = Statevector([1,0])

# Create a quantum circuit with one quibit
qc = QuantumCircuit(1)

# Apply the Pauli X on the qubit
qc.x(0)
state = state.evolve(qc)

# You can print the state as a txt
print(state.draw(output='text'))
