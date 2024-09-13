import json
import math
import re
import datetime

from django.core.files.base import ContentFile

pi = math.pi
PI = math.pi

from django.db import models
from user.models import User
from .apps import MessageConfig
from chatbot.interface import AnswerHandler, FileManager
import chatbot.logic_engine as le
from chatbot.tsp import TSPSolver
import torch

# Defining message types
MESSAGE_TYPES = [('text', 'text'),
                 ('code', 'code'),
                 ('parameters', 'parameters')
                 ]

# Defining button types
BUTTON_TYPES = [('False', 'false'),
                ('ok', 'ok'),
                ('compute', 'compute')
                ]


# Function to transform string input to numerical value
def transform_string_to_value(input_string):
    try:
        result = eval(input_string)
    except Exception:
        # Define regular expression patterns to match the input string
        patterns = [re.compile(r'([a-zA-Zα-ωΑ-Ω]+)\/(\d*\.?\d*)'),
                    re.compile(r'(\d*\.?\d*)([a-zA-Zα-ωΑ-Ω]+)\/(\d*\.?\d*)'),
                    re.compile(r'(\d*\.?\d*)([a-zA-Zα-ωΑ-Ω]+)'),
                    re.compile(r'([a-zA-Zα-ωΑ-Ω]+)')]

        # Match the pattern in the input string
        for i, pattern in enumerate(patterns):
            match = pattern.match(input_string)
            if match:
                break

        if i == 0:
            unit = match.group(1)
            denominator = float(match.group(2))

            # Check if the unit is a recognized mathematical constant
            if unit.lower() == 'pi' or unit.lower() == 'π':
                result = math.pi / denominator
            else:
                print(f"Unsupported unit: {unit}")
        elif i == 1:
            numeric_value = float(match.group(1))
            unit = match.group(2)
            denominator = float(match.group(3))

            # Check if the unit is a recognized mathematical constant
            if unit.lower() == 'pi' or unit.lower() == 'π':
                result = numeric_value * math.pi / denominator
            else:
                print(f"Unsupported unit: {unit}")

        elif i == 2:
            if match:
                if input_string.lower() == 'π':
                    unit = match.group(2)

                    if unit.lower() == 'π':
                        result = math.pi
                    else:
                        print(f"Unsupported unit: {unit}")
                else:
                    numeric_value = float(match.group(1))
                    unit = match.group(2)

                    if unit.lower() == 'pi' or unit.lower() == 'π':
                        result = numeric_value * math.pi
                    else:
                        print(f"Unsupported unit: {unit}")
            else:
                print("Invalid input format")

    return result


# Function to create the message when C4Q hasn't understood the user question
def create_more_details_message(previous_message, user):
    more_details_message = Message(content="Sorry, I didn't quite catch that. " +
                                           "Could you provide more details or ask in a different way?",
                                   previous_message=previous_message,
                                   user=user)
    more_details_message.save()

    what_I_do_message = Message(content="Remember, you can ask me about:\n" +
                                        "1. Defining a quantum gate.\n\n" +
                                        "2. Drawing a quantum gate.\n\n" +
                                        "3. Applying a quantum gate.\n\n" +
                                        "4. Solving a travelling-salesperson problem.\n\n" +
                                        "Gates include: Identity, Pauli X, Pauli Y, Pauli Z, S, Hadamard, Phase, "
                                        "Rotations, CNOT, CZ, SWAP.\n\n" +
                                        "States include: |0>, |1>, |+>, |->, |r>, |l>, |00>, |01>, |10>, |11>, "
                                        "|phi+>, |phi->, |psi+>, |phi->. \n\n" +
                                        "Let's try again! 🚀✨",
                                previous_message=more_details_message,
                                user=user)

    what_I_do_message.save()

    previous_message = what_I_do_message

    return previous_message


# Function to create TSP code content
def create_tsp_code_content(tsp_parameters):
    # Imports required for TSP code
    imports_text = "import networkx as nx  # Importing the networkx library for graph operations\n" \
                   "import numpy as np  # Importing numpy for numerical operations\n" \
                   "from qiskit_algorithms.optimizers import SPSA  # Importing the SPSA optimizer from Qiskit algorithms\n" \
                   "from qiskit.circuit.library import TwoLocal  # Importing TwoLocal circuit from Qiskit circuit library\n" \
                   "from qiskit.primitives import Sampler  # Importing Sampler from Qiskit primitives\n" \
                   "from qiskit_optimization.applications import Tsp  # Importing the Traveling Salesman Problem (TSP) module from Qiskit optimization applications\n" \
                   "from qiskit_algorithms import SamplingVQE  # Importing SamplingVQE from Qiskit algorithms\n" \
                   "from qiskit_optimization.algorithms import MinimumEigenOptimizer  # Importing MinimumEigenOptimizer from Qiskit optimization algorithms\n" \
                   "from qiskit_optimization.converters import QuadraticProgramToQubo  # Importing QuadraticProgramToQubo from Qiskit optimization converters\n\n"

    # Retrieving parameters for TSP
    distances = tsp_parameters.get('distances')
    num_cities = tsp_parameters.get('num_cities')
    city_list = tsp_parameters.get('city_list')

    # Converting city list to dictionary for TSP code
    city_dic = {}
    for i, city in enumerate(city_list):
        city_dic[str(i)] = city

    parameters_text = f"# Parameters for the TSP instance\n" \
                      f"distances = {distances} # Distances between cities\n" \
                      f"num_cities = {num_cities}  # Number of cities\n" \
                      f"city_dic = {city_dic}  # Dictionary mapping city indices to city names\n\n"

    tsp_instance_text = "# Create the TSP graph\n" \
                        "tsp_graph = nx.Graph()  # Initialize an empty graph for the TSP\n" \
                        "tsp_graph.add_nodes_from(np.arange(0, num_cities, 1))  # Add nodes to the graph\n" \
                        "tsp_graph.add_weighted_edges_from(distances)  # Add weighted edges to the graph based on distances\n" \
                        "adj_matrix = nx.to_numpy_array(tsp_graph)  # Convert the graph to an adjacency matrix\n" \
                        "tsp_instance = Tsp(tsp_graph)  # Create a TSP instance using the graph\n\n"

    tsp_to_qubo_text = "# Convert TSP instance to quadratic problem (QP)\n" \
                       "quadratic_problem = tsp_instance.to_quadratic_program()\n\n" \
                       "# Convert QP to QUBO\n" \
                       "qp2qubo = QuadraticProgramToQubo()  # Create a converter object\n" \
                       "qubo = qp2qubo.convert(quadratic_problem)  # Convert the quadratic problem to QUBO form\n" \
                       "pauli_decomposition, offset = qubo.to_ising()  # Convert QUBO to Ising form\n" \
                       "ansatz = TwoLocal(pauli_decomposition.num_qubits, 'ry', 'cz', reps=5, entanglement='linear')  # Define an ansatz for the VQE algorithm\n\n"

    vqe_optimizer_text = "# Define the optimizer for the VQE algorithm\n" \
                         "optimizer = SPSA(maxiter=300)  # Stochastic Perturbation Simulated Annealing (SPSA) optimizer with maximum iterations\n\n" \
                         "# Initialize SamplingVQE with defined parameters\n" \
                         "vqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)\n\n" \
                         "# Initialize MinimumEigenOptimizer with VQE\n" \
                         "vqe_optimizer = MinimumEigenOptimizer(vqe)\n\n"

    result_text = "# Compute the minimum eigenvalue of the Pauli decomposition using VQE\n" \
                  "q_result = vqe.compute_minimum_eigenvalue(pauli_decomposition)\n\n" \
                  "# Sample the most likely result from the computed quantum result\n" \
                  "cl_most_likely_result = tsp_instance.sample_most_likely(q_result.eigenstate)\n\n" \
                  "# Check if the sampled result is feasible\n" \
                  "is_feasible = qubo.is_feasible(cl_most_likely_result)\n\n" \
                  "# Interpret the sampled result to get the list of nodes representing the optimal city order\n" \
                  "list_nodes = tsp_instance.interpret(cl_most_likely_result)\n\n"

    solution_text = "# Construct the solution dictionary containing relevant information\n" \
                    "solution = {'q_state': q_result,\n" \
                    "            'is_feasible': is_feasible,\n" \
                    "            'list_nodes': list_nodes,\n" \
                    "            'city_order': [city_dic[str(node)] for node in list_nodes],  # Map node indices to city names\n" \
                    "            'optimal_value': tsp_instance.tsp_value(list_nodes, adj_matrix)}  # Compute the total distance of the optimal route\n\n" \
                    "# Print the solution\n" \
                    "print(\"time:\", solution['q_state'].optimizer_time)\n" \
                    "print(\"feasible:\", solution['is_feasible'])\n" \
                    "print(\"solution:\", solution['list_nodes'])\n" \
                    "print(\"city order:\", solution['city_order'])\n" \
                    "print(\"solution objective:\", solution['optimal_value'])\n"

    return imports_text + parameters_text + tsp_instance_text + tsp_to_qubo_text + vqe_optimizer_text + result_text + solution_text


class Message(models.Model):
    content = models.TextField(blank=False, null=False)
    role = models.TextField(blank=False, null=False, default="ai")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')
    previous_message = models.OneToOneField('self', on_delete=models.SET_NULL, null=True, blank=True)
    parameters = models.TextField(blank=True, null=True)
    draw = models.ImageField(verbose_name='draw', max_length=255, blank=True, null=True)
    type = models.CharField(
        blank=False,
        null=False,
        max_length=20,
        choices=MESSAGE_TYPES,
        default='text'
    )
    is_solved = models.BooleanField(blank=False, null=False, default=False)
    # needs_button = models.BooleanField(blank=False, null=False, default=False)
    needs_button = models.CharField(
        blank=False,
        null=False,
        max_length=20,
        choices=BUTTON_TYPES,
        default='false'
    )

    def __str__(self):
        return f'{self.content} with id = {self.id}'

    def save(self, *args, **kwargs):
        if self.role == 'user':
            super().save(*args, **kwargs)  # Save the original message
            previous_message = self

            if self.content.lower() == 'yes':
                if self.previous_message.parameters:
                    # Chatbot answers that it will proceed with the question
                    ai_ok = Message(content="Ok, let's do it!",
                                    previous_message=previous_message,
                                    user=self.user,
                                    parameters=self.parameters)
                    ai_ok.save()
                    previous_message = ai_ok

                    # Extract parameters from the user question
                    parameters = json.loads(self.previous_message.parameters)
                    category = parameters.get('category')
                    user_question = self.previous_message.previous_message.content

                    # Save user question and category for retraining
                    file_manager = MessageConfig.classifbert.file_manager
                    if isinstance(category, int) and user_question:
                        with open(file_manager.folder_path + file_manager.file_name, 'a') as file:
                            file.write(f'{category}\t{user_question}' + '\n')

                    gate_name = parameters.get('gate_name')
                    gate = le.gates.get(gate_name)

                    initial_state_name = parameters.get('initial_state_name')
                    initial_state = le.initial_states.get(initial_state_name)

                    if gate_name == 'phase':
                        bert_qa = MessageConfig.bert_qa
                        bert_answer = bert_qa.ask_questions(user_question, ["What is the phase shift?"])

                        if bert_answer:
                            try:
                                phase_shift = transform_string_to_value(
                                    bert_answer.get('What is the phase shift?').get('answer').get('answer')[0])
                                parameters['phase_shift'] = phase_shift
                            except Exception:
                                phase_shift = 0
                                parameters['phase_shift'] = phase_shift
                                parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                                error_message = Message(
                                    content='The reading for phase shift was unavailable, leading to '
                                            'the assumption of a phase shift of zero radians.',
                                    previous_message=previous_message,
                                    user=self.user,
                                    parameters=parameters_json)
                                error_message.save()
                                previous_message = error_message
                                print(
                                    'The reading for phase shift was unavailable, leading to the assumption of a phase '
                                    'shift of zero radians.')
                        else:
                            phase_shift = 0

                        gate = le.PhaseGate(phase_shift)
                    elif gate_name == 'rotation':
                        bert_qa = MessageConfig.bert_qa
                        questions = ["What is the angle of the rotation?", "What is the axis of the rotation?"]
                        bert_answer = bert_qa.ask_questions(context=user_question, questions=questions)

                        if bert_answer:
                            try:
                                angle = transform_string_to_value(
                                    bert_answer.get(questions[0]).get('answer').get('answer')[0])
                                parameters['angle'] = angle

                            except Exception:
                                print('Rotation angle could not be read, and so an angle of zero radians was assumed.')
                                angle = 0
                                parameters['angle'] = angle
                                parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                                error_message = Message(
                                    content='The rotation angle could not be determined, resulting in the assumption of '
                                            'an angle of zero radians.',
                                    previous_message=previous_message,
                                    user=self.user,
                                    parameters=parameters_json)
                                error_message.save()
                                previous_message = error_message

                            axis = bert_answer.get(questions[1]).get('answer').get('answer')[0].lower()
                            prob = bert_answer.get(questions[1]).get('probability').get('probability')[0]

                            if axis == 'empty' or prob < 0.9:
                                print(
                                    'The rotation axis reading was unavailable, prompting the assumption of the x-axis '
                                    'as the rotation axis.')
                                axis = 'x'
                                error_message = Message(
                                    content='The rotation axis reading was unavailable, prompting the assumption of the '
                                            'x-axis as the rotation axis.',
                                    previous_message=previous_message,
                                    user=self.user,
                                    parameters=parameters)
                                error_message.save()
                                previous_message = error_message

                            parameters['axis'] = axis

                        else:
                            print('Bert did not give any answer...')
                            angle = 0
                            axis = 'x'
                            parameters['angle'] = angle
                            parameters['axis'] = axis
                            error_message = Message(
                                content='The rotation axis and angle readings were unavailable. I assumed the x-axis for '
                                        'the rotation axis and zero radians for the angle.',
                                previous_message=previous_message,
                                user=self.user,
                                parameters=parameters)
                            error_message.save()

                        axis_rotation_map = {'x': 'RX', 'y': 'RY', 'z': 'RZ'}
                        gate_name = axis_rotation_map.get(axis)
                        gate_object = gate.get(gate_name)

                        if gate_object:
                            gate = gate_object(angle)
                        else:
                            gate = le.gates.get('rotation').get('RX')(0)

                    answer_handler = AnswerHandler(category, gate, [initial_state])
                    le_answer = answer_handler.apply_gate_method()

                    if category == 1:
                        parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                        le_answer_message = Message(content='Here is the circuit:',
                                                    previous_message=previous_message,
                                                    user=self.user,
                                                    parameters=parameters_json)

                        current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                        le_answer_message.draw.save(f'qiskit_draws/{current_date}.png', ContentFile(le_answer),
                                                    save=True)

                    else:
                        parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                        le_answer_message = Message(content=le_answer,
                                                    previous_message=previous_message,
                                                    user=self.user,
                                                    parameters=parameters_json)

                    le_answer_message.save()
                    previous_message = le_answer_message

                    another_question_message = Message(content="What is your next question?",
                                                       user=self.user,
                                                       previous_message=previous_message)
                    another_question_message.save()
                    previous_message = another_question_message
                else:
                    previous_message = create_more_details_message(previous_message, self.user)

            elif self.content.lower() == 'ok':
                tsp_parameters = json.loads(self.previous_message.parameters)

                parameter_message_id = self.previous_message.previous_message.id
                parameter_message = Message.objects.get(pk=parameter_message_id)
                parameter_message.is_solved = True
                parameter_message.save()

                print("tsp_parameters", tsp_parameters)
                tsp_parameters_json = json.dumps(tsp_parameters, sort_keys=True, indent=4)

                intro_text = "Using Qiskit, we can solve this TSP problem as follows:\n"
                tsp_intro_code_message = Message(content=intro_text,
                                                 previous_message=previous_message,
                                                 user=self.user,
                                                 parameters=tsp_parameters_json)
                tsp_intro_code_message.save()
                previous_message = tsp_intro_code_message

                tsp_code_message_content = create_tsp_code_content(tsp_parameters)

                tsp_code_message = Message(content=tsp_code_message_content,
                                           previous_message=previous_message,
                                           user=self.user,
                                           parameters=previous_message.parameters,
                                           type='code')
                tsp_code_message.save()
                previous_message = tsp_code_message

                ask_computing_message = Message(content="If you want me to compute the solution, press the "
                                                        "compute button or send 'compute'. Take into account that "
                                                        "the computation can take a while.\n Otherwise, "
                                                        "you can ask your next question.",
                                                previous_message=previous_message,
                                                user=self.user,
                                                parameters=previous_message.parameters,
                                                needs_button='compute')
                ask_computing_message.save()
                previous_message = ask_computing_message

            elif self.content.lower() == 'compute':
                tsp_parameters = json.loads(self.previous_message.parameters)
                num_cities = tsp_parameters.get('num_cities')
                distances = tsp_parameters.get('distances')
                print('tsp_parameters after compute', tsp_parameters)

                tsp_solver = TSPSolver(num_cities, distances, seed=123)
                quantum_solution = tsp_solver.solve_quantum()

                # We delete the q_state of the solution because it is not json serializable and we do not need it (atm)
                del quantum_solution['q_state']

                tsp_parameters['quantum_solution'] = quantum_solution

                # Converting city list to dictionary for TSP code
                city_dic = {}
                for i, city in enumerate(tsp_parameters.get('city_list')):
                    city_dic[str(i)] = city

                list_nodes = tsp_parameters.get('quantum_solution').get('list_nodes')
                city_order = [city_dic[str(node)] for node in list_nodes]
                optimal_value = tsp_parameters.get('quantum_solution').get('optimal_value')
                print(city_order)
                print(list_nodes)

                solution_content = f"In your concrete case, we obtain that the order of the cities should be {city_order} and the optimal distance is {round(optimal_value, 2)}."

                solution_message = Message(content=solution_content,
                                           previous_message=previous_message,
                                           user=self.user,
                                           parameters=previous_message.parameters)
                solution_message.save()
                previous_message = solution_message

            elif self.content.lower() == 'no':
                previous_message = create_more_details_message(previous_message, self.user)

            else:
                category, logits, top_indices = MessageConfig.classifbert.classify_user_input(self.content)

                print("logits", logits)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                print(probs)

                if probs[0][category] < 0.9:
                    previous_message = create_more_details_message(previous_message, self.user)
                else:
                    if category == 3:
                        context = self.content
                        city_question = "Which cities wants to visit the person?"
                        city_answer_dic = MessageConfig.tspQAmodel.answer_question(question=city_question, context=context)

                        if ',' in city_answer_dic['answer']:
                            city_list = city_answer_dic['answer'].split(',')
                        else:
                            city_list = city_answer_dic['answer'].split(' ')

                        # Remove extra spaces and create a list
                        city_list = [city.strip() for city in city_list]
                        city_list[-1] = city_list[-1].replace('and', '')

                        num_cities = len(city_list)

                        distances = []
                        distance_text = ""

                        for i in range(num_cities - 1):
                            for j in range(i + 1, num_cities):
                                distance_question = f"What is the distance between {city_list[i]} and {city_list[j]}?"
                                distance_dic = MessageConfig.tspQAmodel.answer_question(question=distance_question,
                                                                                        context=context)
                                distance = float(distance_dic['answer'])
                                distances.append((i, j, distance))
                                distance_text = distance_text + f"\n\t{city_list[i]} - {city_list[j]} {distance}"

                        tsp_parameters = {'num_cities': num_cities, 'city_list': city_list, 'distances': distances}
                        tsp_parameters_json = json.dumps(tsp_parameters, sort_keys=True, indent=4)

                        content = "I understand that you want to solve a travelling-salesperson problem (TSP). The " \
                                  "parameters I am considering for the TSP are the following:\n"

                        tsp_understand_message = Message(content=content,
                                                         previous_message=previous_message,
                                                         user=self.user,
                                                         parameters=tsp_parameters_json
                                                         )
                        tsp_understand_message.save()
                        previous_message = tsp_understand_message

                        print("tsp_parameters =", tsp_parameters)

                        parameters_message = Message(content="Show parameters",
                                                     previous_message=previous_message,
                                                     user=self.user,
                                                     parameters=tsp_parameters_json,
                                                     type='parameters'
                                                     )
                        parameters_message.save()
                        previous_message = parameters_message

                        edit_message_content = "If you want to change any parameter, click on the edit button. When the parameters are correct, press the 'ok' button or send 'ok' to continue."
                        edit_message = Message(content=edit_message_content,
                                               previous_message=previous_message,
                                               user=self.user,
                                               parameters=previous_message.parameters,
                                               needs_button='ok'
                                               )
                        edit_message.save()
                        previous_message = edit_message

                    else:
                        gate_name, initial_state_name, understood_question = MessageConfig.classifbert.process_user_question(
                            self.content, category)

                        print("gate_name =", gate_name)

                        if gate_name is None:
                            previous_message = create_more_details_message(previous_message, self.user)

                        else:
                            parameters = {"initial_state_name": initial_state_name, 'gate_name': gate_name,
                                          'category': category}
                            parameters_json = json.dumps(parameters, sort_keys=True, indent=4)
                            ai_question_check = Message(content=understood_question,
                                                        previous_message=self,
                                                        user=self.user,
                                                        parameters=parameters_json
                                                        )
                            ai_question_check.save()
                            previous_message = ai_question_check
        else:
            super().save(*args, **kwargs)  # Save the new message
