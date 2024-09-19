import json
import os
from simpletransformers.question_answering import QuestionAnsweringModel

class BertQA:
    def __init__(self, epochs=4, train_files=["./data/QA/phase_shifts_data_train.json", "./data/QA/rotation_data_train.json"],
                 test_files=["./data/QA/phase_shifts_data_test.json", "./data/QA/rotation_data_test.json"],
                 model_type="bert", model_name="bert-base-uncased",
                 output_dir='LLM_QA',
                 train_from_scratch=True):
        self.train_files = train_files
        self.test_files = test_files
        self.model_type = model_type
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_from_scratch = train_from_scratch

        directory = f"{os.path.dirname(os.path.abspath(__file__))}/{self.output_dir}"

        self.train_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "use_cached_eval_features": True,
            "output_dir": f"{directory}/outputs/{self.model_type}",
            "best_model_dir": f"{directory}/{self.model_type}/best_model",
            "evaluate_during_training": True,
            "max_seq_length": 128,
            "num_train_epochs": epochs,
            "evaluate_during_training_steps": 1000,
            "save_model_every_epoch": False,
            "save_eval_checkpoints": False,
            "n_best_size": 8,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "learning_rate": 2e-5,
            "verbose": True,
            "evaluation_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_steps": False
        }

        self.model = QuestionAnsweringModel(self.model_type, self.model_name, args=self.train_args, use_cuda=False)

        if self.train_from_scratch:
            self.train_model()


    def load_data(self, file_paths):
        # Initialize an empty dictionary to store the data
        data = []

        # Iterate through the file paths and read data from each file
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    # Load JSON data from the file
                    file_data = json.load(file)

                    # Merge the loaded data into the main data dictionary
                    data.extend(file_data)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_path}")

        return data

    def train_model(self):
        train_data = self.load_data(self.train_files)
        test_data = self.load_data(self.test_files)
        self.model.train_model(train_data, eval_data=test_data)

    def load_saved_model(self, model_path="/app/backend/chatbot/LLM_QA/bert/best_model"):
        model_path = os.path.dirname(os.path.abspath(__file__)) + "/LLM_QA/bert/best_model"

        self.model = QuestionAnsweringModel("bert", model_path, use_cuda=False)

    def ask_questions(self, context, questions):
        self.load_saved_model()
        # Prepare the input data for multiple questions
        to_predict = [
            {
                "context": context,
                "qas": [
                    {
                        "question": q,
                        "id": str(i),
                    } for i, q in enumerate(questions)
                ],
            }
        ]

        # Make predictions with the model
        answers, probabilities = self.model.predict(to_predict, n_best_size=1)

        # Extract answers and probabilities for each question
        question_answers = {}
        for i, q in enumerate(questions):
            question_answers[q] = {
                "answer": answers[i],
                "probability": probabilities[i]
            }

        return question_answers

if __name__ == "__main__":
    train_file_paths = ["./data/QA/rotation_data_train.json", "./data/QA/phase_shifts_data_train.json"]
    test_file_paths = ["./data/QA/rotation_data_test.json", "./data/QA/phase_shifts_data_test.json"]

    bert_qa = BertQA(epochs=1, train_files=train_file_paths,
                         test_files=test_file_paths,
                         train_from_scratch=True)
    # bert_qa = BertQA(train_from_scratch=False)

    answer = bert_qa.ask_questions("After the application of a phase gate with a phase shift of 5π/6 to the |0> state, what will the new state be?", ["What is the phase shift?"] )
    print(answer['What is the phase shift?']['answer']['answer'][0])

    answer = bert_qa.ask_questions("Could you perform a smooth rotation of 5pi/8 around the y?", ["What is the angle of the rotation?"] )
    print(answer['What is the angle of the rotation?']['answer']['answer'][0])
