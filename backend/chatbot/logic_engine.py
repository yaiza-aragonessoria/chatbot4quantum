import os

import qiskit as qiskit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from qiskit.quantum_info import Statevector
import numpy as np
import math
import io



class Gate:
    def __init__(self, name, n_qubits, definition, qiskit_name, alternative_names=None):
        self.name = name
        self.alternative_names = alternative_names
        self.n_qubits = n_qubits
        self.definition = definition
        self.qiskit_name = qiskit_name

    def explain(self):
        return self.definition

    def apply(self, qubit, circ=None):
        if not circ:
            circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(qubit)

        return circ

    def draw(self):
        if self.n_qubits == 1:
            circ = self.apply(qubit=0)
        else:
            circ = self.apply()


        draw = circ.draw(output='mpl')
        canvas = FigureCanvasAgg(draw)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        image_data = buffer.getvalue()

        return image_data

    def compute_state(self, initial_state=None):
        if self.n_qubits == 1:
            if not initial_state:
                state = Statevector([1, 0])
            else:
                state = Statevector(initial_state)

            circ = self.apply(0)

        else:
            if not initial_state:
                state=Statevector([1, 0, 0, 0])
            else:
                state = Statevector(initial_state)

            circ = self.apply()

            # Set the initial state of the simulator to the ground state using from_int
            state = Statevector(state)

        # Evolve the state by the quantum circuit
        state = state.evolve(circ)

        # draw using text
        return state.draw(output='text')

class PhaseGate(Gate):
    def __init__(self, phase_shift,
                 name='phase gate',
                 alternative_names=('phase', 'phase shift'),
                 n_qubits=1,
                 definition='The Phase gate is a family of single-qubit operations that map the basis states |0> --> |0> and |1> --> e^(i\u03B8)|1>, where \u03B8 takes any value in [0, 2\u03C0).',
                 qiskit_name='p'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.phase_shift = phase_shift

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.phase_shift, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ

class RX(Gate):
    def __init__(self, angle,
                 name='rotation around x',
                 alternative_names=('RX',),
                 n_qubits=1,
                 definition='The quantum gate Rx(\u03B8) is a single-qubit operation that performs a rotation of \u03B8 radians around the x-axis.',
                 qiskit_name='rx'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.angle = angle

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.angle, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ

class RY(Gate):
    def __init__(self, angle,
                 name='rotation around y',
                 alternative_names=('RY',),
                 n_qubits=1,
                 definition='The quantum gate Ry(\u03B8) is a single-qubit operation that performs a rotation of \u03B8 radians around the y-axis.',
                 qiskit_name='ry'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.angle = angle

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.angle, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ

class RZ(Gate):
    def __init__(self, angle,
                 name='rotation around z',
                 alternative_names=('RZ',),
                 n_qubits=1,
                 definition='The quantum gate Rz(\u03B8) is a single-qubit operation that performs a rotation of \u03B8 radians around the z-axis.',
                 qiskit_name='rz'):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.angle = angle

    def apply(self, qubit=0):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.angle, qubit)  # maybe later we need a parameter for the qubit where the gate is applied

        return circ


class CNOT(Gate):
    def __init__(self, control_qubit,
                 target_qubit,
                 name='CNOT',
                 n_qubits=2,
                 qiskit_name='cx',
                 definition='The controlled NOT (CNOT) gate is a two-qubit gate that flips the target qubit state '
                            'from |0〉to |1〉or vice versa if and only if the control qubit |1>. Otherwise, the target '
                            'qubit is unchanged.',
                 alternative_names=('control not', 'CNOT', 'C-NOT', 'CX', 'control x'),
                 ):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.control_qubit, self.target_qubit)

        return circ


class CZ(Gate):
    def __init__(self, control_qubit, target_qubit, name='control z',
                 alternative_names=('CZ',),
                 n_qubits=2, qiskit_name='cz',
                 definition='The controlled phase, (CZ) gate is a two-qubit gate that applies a Pauli Z on the target '
                            'qubit state if and only if the control qubit |1>. Otherwise, the target qubit is '
                            'unchanged.',
                 ):
        super().__init__(name, n_qubits, definition, qiskit_name, alternative_names)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.control_qubit, self.target_qubit)

        return circ


class Swap(Gate):
    def __init__(self, qb1, qb2, name='swap', n_qubits=2, qiskit_name='swap',
                 definition='The Swap gate is a two-qubit operation that swaps the state of the two qubits involved '
                            'in the operation.'):
        super().__init__(name, n_qubits, definition, qiskit_name)
        self.qb1 = qb1
        self.qb2 = qb2

    def apply(self):
        circ = qiskit.QuantumCircuit(self.n_qubits)
        qiskit_function = getattr(circ, self.qiskit_name)
        qiskit_function(self.qb1, self.qb2)

        return circ


id = Gate(name='identity',
          n_qubits=1,
          definition='The Identity gate is a single-qubit operation that leaves any unchanged.',
          qiskit_name='id'
          )

id2 = Gate(name='identity',
          n_qubits=1,
          definition='The Identity gate is a single-qubit operation that leaves any unchanged.',
          qiskit_name='id'
          )

pauli_x = Gate(name='Pauli x',
               n_qubits=1,
               definition='The Pauli X gate is a single-qubit operation that rotates the qubit around the x axis by '
                          '\u03C0 radians.',
               qiskit_name='x',
               alternative_names=('pauliX', 'bit flip', 'bit-flip',),
               )

pauli_y = Gate(name='Pauli y',
               n_qubits=1,
               definition='The Pauli Y gate is a single-qubit operation that rotates the qubit around the y axis by '
                          '\u03C0 radians.',
               qiskit_name='y',
               alternative_names=('pauliY',),
               )

pauli_z = Gate(name='Pauli z',
               n_qubits=1,
               definition='The Pauli Z gate is a single-qubit operation that rotates the qubit around the z axis by '
                          '\u03C0 radians.',
               qiskit_name='z',
               alternative_names=('pauliZ', 'phase flip', 'phase-flip',),
               )

hadamard = Gate(name='Hadamard',
                n_qubits=1,
                definition='The Hadamard gate is a single-qubit operation that maps the basis states |0> --> |+> = ('
                           '|0>+|1>)/\u221A2 and |1> --> |-> = (|0>-|1>)/\u221A2.',
                qiskit_name='h'
                )

s = Gate(name='S gate',
               n_qubits=1,
               definition='The S gate is a single-qubit operation that performs a \u03C0/2-rotation around the z axis.',
               qiskit_name='s',
               alternative_names=('s daga', 'S\u2020', 'inverse of S', ),
               )

sdg = Gate(name='s daga',
               n_qubits=1,
               definition='The S\u2020 is the inverse of the S gate, i.e., SS\u2020 = S\u2020S = 1. Therefore, '
                          'S\u2020 is also a one-qubit gate that rotates the qubit \u03C0/2 radians around the z '
                          'axis, but on the other direction.',
               qiskit_name='s',
               alternative_names=('square root of z', '√Z', ),
               )

phasePI2 = PhaseGate(math.pi / 2)
RXPI = RX(math.pi)
cnot = CNOT(0, 1)
cz = CZ(1, 0)
swap = Swap(0, 1)

official_gates = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, hadamard.name: hadamard,
         s.name: s, sdg.name: sdg, phasePI2.name: PhaseGate, cnot.name: cnot, cz.name: cz, swap.name: swap,
         'rotation': {'RX': RX, 'RY': RY, 'RZ': RZ,}, 'phase': PhaseGate}

gate_for_names = {id.name: id, pauli_x.name: pauli_x, pauli_y.name: pauli_y, pauli_z.name: pauli_z, s.name: s,
                   sdg.name: sdg, hadamard.name: hadamard, phasePI2.name: phasePI2,
                   cnot.name: cnot, cz.name: cz, swap.name: swap, 'rotation': RXPI}

gate_names = []
gates = {}
for gate_key in gate_for_names.keys():
    gate = gate_for_names[gate_key]
    official_gate = official_gates[gate_key]
    gate_names.append(gate_key)
    gates[gate_key] = gate
    if gate.alternative_names:
        gate_names.extend(gate.alternative_names)
        for alternative_name in gate.alternative_names:
            gates[alternative_name] = official_gate
gates['rotation'] = {'RX': RX, 'RY': RY, 'RZ': RZ}

initial_states = {'|0>': Statevector([1,0]),
                  '|1>': Statevector([0,1]),
                  '|+>': Statevector([1/np.sqrt(2), 1/np.sqrt(2)]),
                  '|->': Statevector([1/np.sqrt(2), -1/np.sqrt(2)]),
                  '|r>': Statevector([1/np.sqrt(2), complex(1/np.sqrt(2) * 1j)]),
                  '|l>': Statevector([1/np.sqrt(2), complex(-1/np.sqrt(2) * 1j)]),
                  '|00>': Statevector([1, 0, 0, 0]),
                  '|01>': Statevector([0, 1, 0, 0]),
                  '|10>': Statevector([0, 0, 1, 0]),
                  '|11>': Statevector([0, 0, 0, 1]),
                  '|ϕ+>': Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
                  '|phi+>': Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
                  '|ψ+>': Statevector([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),
                  '|psi+>': Statevector([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),
                  '|ψ->': Statevector([0, 1/np.sqrt(2), -1/np.sqrt(2), 0]),
                  '|psi->': Statevector([0, 1/np.sqrt(2), -1/np.sqrt(2), 0]),
                  }

if __name__ == '__main__':
    # r_class = gates.get('rotation').get("RX")
    # print(r_class)
    # r_object = r_class(math.pi)
    # r_object.draw()

    print(len(gate_names))
    print(gate_names)
    print(len(gates))
    print(gates)
    print(type(gates.get('phase')))
    print(len(official_gates))