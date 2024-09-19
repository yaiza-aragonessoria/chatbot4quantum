import datetime
import os
import sys

import torch
import math

pi = math.pi
PI = math.pi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chatbot.logic_engine as le
from chatbot.LLM import BertTextClassifier
from chatbot.LLM_QA import BertQA


class FileManager:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_name = None

    def get_latest_file(self):
        # Get the latest data file in the specified folder
        files = os.listdir(self.folder_path)

        # Filter the list to include only files (not directories)
        files = [file for file in files if os.path.isfile(os.path.join(self.folder_path, file))]

        if files:
            # Sort the files by modification time (most recent first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.folder_path, x)))

            # Set the data_file_name attribute to the name of the latest file
            latest_file = files[-1]

            if latest_file[0] == 'g':
                current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.file_name = f"questions{current_date}.txt"
                with open(self.folder_path + self.file_name, 'w') as file:
                    pass
            else:
                self.file_name = latest_file


# Define a class for calculating Levenshtein distance
class LevenshteinDistance:
    @staticmethod
    def calculate_distance(s1, s2):
        # Calculate the Levenshtein distance between two strings
        N, M = len(s1), len(s2)
        dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

        # Populate the DP matrix
        for j in range(M + 1):
            dp[0][j] = j
        for i in range(N + 1):
            dp[i][0] = i
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # Insertion
                        dp[i][j - 1],  # Deletion
                        dp[i - 1][j - 1]  # Replacement
                    )

        return dp[N][M]


# Define a class for finding parameters
class ParameterFinder:
    def __init__(self, gates, initial_states, max_distance=3):
        self.gates = gates
        self.max_distance = max_distance
        self.initial_states = initial_states

    def find_closest_substrings(self, s):
        # Find the closest substrings in the provided text
        closest_gates = []
        closest_initial_states = []

        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]

                for item_list, item_type in [(self.gates, 'Gate'), (self.initial_states, 'Initial State')]:
                    closest_item = None
                    closest_distance = self.max_distance

                    for item in item_list:
                        # Calculate the Levenshtein distance between the substring and item
                        distance = LevenshteinDistance.calculate_distance(substring.lower(), item.lower())

                        # If the distance is within the maximum allowed distance, consider it
                        if distance <= closest_distance:
                            closest_item = item
                            closest_distance = distance

                    if item_type == 'Gate' and closest_item is not None:
                        closest_gates.append((substring, closest_item, closest_distance))
                    elif item_type == 'Initial State' and closest_item is not None:
                        closest_initial_states.append((substring, closest_item, closest_distance))

        return closest_gates, closest_initial_states

    def find_parameters(self, s):
        # Find parameters based on closest substrings
        closest_gates, closest_initial_states = self.find_closest_substrings(s)

        if closest_gates:
            # Find the minimum distance among gate matches
            min_distance = min(item[2] for item in closest_gates)

            # Filter gate matches to keep only those with the minimum distance
            closest_gates = [item for item in closest_gates if item[2] == min_distance]

        if closest_initial_states:
            # Find the minimum distance among initial state matches
            min_distance = min(pair[2] for pair in closest_initial_states)

            # Filter initial state matches to keep only those with the minimum distance
            closest_initial_states = [pair for pair in closest_initial_states if pair[2] == min_distance]

        # Return the closest gate and initial state matches
        return {'closest_gates': closest_gates, 'closest_initial_states': closest_initial_states}


# Define a class for checking and formatting user questions
class QuestionChecker:
    # Define a dictionary of question templates with placeholders
    question_templates = {
        0: "I understand that you want to know the definition of the gate {gate_name}. Is this correct?",
        1: "I understand that you want me to draw the circuit representation of the gate {gate_name}. Is this correct?",
        2: "I understand that you want me to compute the resulting state after applying the gate {gate_name} on {initial_state}. Is this correct?"
    }

    def __init__(self, type_of_question, gate_name, initial_state=None):
        self.type_of_question = type_of_question
        self.gate_name = gate_name
        self.initial_state = initial_state

    def check_question(self):
        # Throw an error when the question type is 2 (computation) and an initial state is not provided
        if self.type_of_question == 2 and self.initial_state is None:
            return "I understand that you want me to compute the resulting state after applying the gate {gate_name}. Is this correct?".format(
                gate_name=self.gate_name) + ' (yes/no): '

        # Get the appropriate question template based on the question type
        template = self.question_templates.get(self.type_of_question)

        # If a valid template is found, format it with the gate name and initial state
        if template is not None:
            return template.format(gate_name=self.gate_name, initial_state=self.initial_state) + ' (yes/no): '
        else:
            return "Invalid question type."


# Define a class for handling answers
class AnswerHandler:
    def __init__(self, category, gate, parameters):
        self.category = category
        self.gate = gate
        self.parameters = parameters

    def apply_gate_method(self):
        method_map = {
            # Map question category to appropriate gate methods
            0: self.gate.explain,  # Explanation of the gate
            1: self.gate.draw,  # Drawing the gate
            2: self.gate.compute_state,  # Computing state after applying the gate
        }

        method = method_map.get(self.category)
        if self.category == 2 and self.parameters[0] is None:
            print('No initial state was given.\nAssuming that the initial state is |0>.')

        if method:
            if self.parameters[0]:
                return method(*self.parameters)
            else:
                return method()
        else:
            print("Invalid category")


# Define the main Chatbot class
class Chatbot:
    def __init__(self, data_folder_path, le, save=True):
        self.bert_model = BertTextClassifier(val_ratio=0.2, batch_size=16, epochs=1, num_labels=4)
        self.file_manager = FileManager(data_folder_path)
        self.gates = le.gates
        self.gate_names = le.gate_names
        self.initial_states = le.initial_states
        self.save = save

    def initialize(self, checkpoint_path, map_location, retraining_bound):
        # Initialize the chatbot with a pretrained BERT model
        self.bert_model.load_checkpoint(train_from_scratch=False, path_to_model=checkpoint_path)
        self.file_manager.get_latest_file()

        print("*** Chatbot initialized ***")

        if self.file_manager.file_name:
            data_path = self.file_manager.folder_path + self.file_manager.file_name
            with open(data_path, 'r') as file:
                num_lines = sum(1 for line in file)

            # check if retraining BERT model is required
            if num_lines >= retraining_bound:
                print("Improving chatbot...")

                # self.bert_model.train(train_from_scratch=False, path_to_model=checkpoint_path, data_path=data_path, save=self.save)

                current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.file_manager.file_name = f"questions{current_date}.txt"

    def classify_user_input(self, user_question):
        """
        Classify the user's input using the BERT model.
        :param user_question: The user's input question.
        :return: The predicted category and logits.
        """
        inputs = self.bert_model.tokenize_data(user_question)
        print(user_question)

        with torch.no_grad():
            outputs = self.bert_model.model(**inputs)

        logits = outputs.logits
        top_indices = torch.topk(logits, k=3).indices
        category = top_indices[0][0].item()  # Assuming you want the top predicted category

        return category, logits, top_indices

    def process_user_question(self, user_question, category):
        # Find parameters related to the user's question
        finder = ParameterFinder(self.gate_names, self.initial_states, max_distance=0)
        parameters = finder.find_parameters(user_question)
        closest_gates = parameters['closest_gates']
        closest_initial_states = parameters['closest_initial_states']

        gate_name = closest_gates[0][1] if closest_gates else None
        initial_state_name = closest_initial_states[0][1] if closest_initial_states else None

        question_checker = QuestionChecker(category, gate_name=gate_name, initial_state=initial_state_name)
        understood_question = question_checker.check_question()

        return gate_name, initial_state_name, understood_question

    def ask_user_for_confirmation(self, understood_question):
        while True:
            user_response = input(understood_question)

            if user_response.lower() == "yes" or user_response == '':
                print("Ok, let's do it!")
                return user_response
            elif user_response.lower() == 'no':
                print("Ups... Let me try again.")
                return 'no'
            else:
                print("Please enter 'yes' or 'no' to confirm or deny the question.")

    def handle_user_response(self, user_response, category, user_question, i):
        exit_loop = False

        if user_response == 'yes' or user_response == '':
            # Append the user's question to a data file
            if not self.file_manager.file_name:
                self.file_manager.get_latest_file()
            with open(self.file_manager.folder_path + self.file_manager.file_name, 'a') as file:
                file.write(f'{category}\t{user_question}' + '\n')

            exit_loop = True
        elif user_response == 'no':
            i += 1
        else:
            print("Please enter 'yes' or 'no' to confirm or deny the question.")

        return exit_loop, i

    def handle_unclassified_question(self, logits, top_indices, i):
        if logits[0][top_indices[0][i]].item() <= 0:
            print("Unfortunately I cannot understand your question. Please, try to ask it again giving more details.")

    def handle_phase_or_rotation_question(self, gate_name, user_question):
        if gate_name == 'phase':
            bert_qa = BertQA(train_from_scratch=False)
            bert_answer = bert_qa.ask_questions(user_question, ["What is the phase shift?"])

            if bert_answer:
                try:
                    phase_shift = eval(bert_answer.get('What is the phase shift?').get('answer').get('answer')[0])
                except Exception:
                    print('Phase shift could not be read, and so a phase shift of zero radians was assumed.')
                    phase_shift = 0
            else:
                phase_shift = 0

            gate = self.gates.get(gate_name)(phase_shift)
            return phase_shift, None, None, gate_name, gate

        elif gate_name == 'rotation':
            bert_qa = BertQA(train_from_scratch=False)
            questions = ["What is the angle of the rotation?", "What is the axis of the rotation?"]
            bert_answer = bert_qa.ask_questions(context=user_question, questions=questions)

            if bert_answer:
                try:
                    angle = eval(bert_answer.get(questions[0]).get('answer').get('answer')[0])
                except Exception:
                    print('Rotation angle could not be read, and so an angle of zero radians was assumed.')
                    angle = 0

                try:
                    axis = bert_answer.get(questions[1]).get('answer').get('answer')[0].lower()
                except Exception:
                    print('Rotation axis could not be read, and so the x-axis was assumed.')
                    axis = 'x'
            else:
                print('Bert did not give any answer...')
                angle = 0

            axis_rotation_map = {'x': 'RX', 'y': 'RY', 'z': 'RZ'}
            gate_name = axis_rotation_map.get(axis)
            gate_object = self.gates.get('rotation').get(gate_name)

            if gate_object:
                gate = gate_object(angle)
            else:
                gate = self.gates.get('rotation').get('RX')(0)

            return None, angle, axis, gate_name, gate

    def apply_gate_method(self, category, gate, parameters):
        answer_handler = AnswerHandler(category, gate, parameters)
        return answer_handler.apply_gate_method()

    def start(self):
        while True:
            user_question = input("Enter your question (type 'exit' to quit): ")

            if user_question.lower() == 'exit':
                print("Exiting the chatbot.")
                break

            category, logits, top_indices = self.classify_user_input(user_question)

            i = 0
            while logits[0][top_indices[0][i]].item() >= 0:
                category = top_indices[0][i].item()

                gate_name, initial_state_name, understood_question = self.process_user_question(user_question,
                                                                                                category)
                gate = self.gates.get(gate_name)
                initial_state = self.initial_states.get(initial_state_name)

                user_response = self.ask_user_for_confirmation(understood_question)

                exit_loop , i = self.handle_user_response(user_response, category, user_question, i)

                if exit_loop:
                    break

            self.handle_unclassified_question(logits, top_indices, i)

            if user_response.lower() == 'yes' or user_response == '':
                parameters = [initial_state]

                if gate_name == 'phase' or gate_name == 'rotation':
                    phase_shift, angle, axis, gate_name, gate = self.handle_phase_or_rotation_question(gate_name,
                                                                                                       user_question)
                    # parameters = [phase_shift] if gate_name == 'phase' else [angle, axis]


if __name__ == "__main__":
    # Create an instance of the Chatbot class, passing the data folder path
    data_folder_path = './data/gate_questions/'
    chatbot = Chatbot(data_folder_path, le, save=False)

    # Initialize the chatbot with the specified checkpoint file
    checkpoint_folder_path = './model/'
    file_manager = FileManager(checkpoint_folder_path)
    file_manager.get_latest_file()
    checkpoint_path = checkpoint_folder_path + file_manager.file_name

    chatbot.initialize(checkpoint_path, map_location='cpu', retraining_bound=10**5)

    tsp_question = "I need to plan a road trip between three cities – New York City, Tokyo, and London. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (25.3), Tokyo to London (61.5), and London to New York City (46.01). I want to optimize for the shortest overall distance. Can you assist with that?"
    tsp_question = 'compute pauli x'
    tsp_question = 'I want to solve a tsp'

    print(chatbot.classify_user_input(tsp_question))

    # Start the chatbot interaction
    # chatbot.start()
