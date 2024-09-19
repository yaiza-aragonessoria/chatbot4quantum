import json
import os
import random

from datasets import Dataset
from transformers import AutoTokenizer, default_data_collator, get_scheduler, pipeline
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
metric = evaluate.load("squad")
import numpy as np
import collections
import torch
from transformers import AutoModelForQuestionAnswering
from accelerate import Accelerator
from sklearn.model_selection import train_test_split

class TSPQAModel:
    def __init__(
        self,
        model_checkpoint,
        max_length=384,
        stride=128,
        n_best=20,
        max_answer_length=30,
        num_train_epochs=3,
        output_dir="bert-finetuned-squad-accelerate",
        train_size=0.8
    ):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.max_length = max_length
        self.stride = stride
        self.n_best = n_best
        self.max_answer_length = max_answer_length
        self.num_train_epochs = num_train_epochs
        self.output_dir = output_dir
        self.train_size = train_size

    def load_data(self, file_paths, max_num_rows=None):
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

                    if max_num_rows:
                        random.shuffle(data) # Shuffle the data randomly

                        # Select the specified number of entries
                        data = data[:max_num_rows]
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_path}")

        return data

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx  # the index in squence_ids where the context starts
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1  # the index in squence_ids where the context ends

            # Find the tokens that correspond to the answer
            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                # start = idx - 1  # only to check if the labeled answer is the same as theoretical answer (see below)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

                # end = idx + 1
                # labeled_answer = self.tokenizer.decode(inputs["input_ids"][sample_idx][start: end + 1])

                # print(f"Theoretical answer: {answer['text'][0]}")
                # print(f"labels give: {labeled_answer}")
                # print(answer['text'][0]==labeled_answer)
                # for index in range(len(answer['text'][0])):
                #     print(answer['text'][0][index])
                #     print(labeled_answer[index])

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def compute_metrics(self, start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -self.n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -self.n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > self.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    def save_metrics(self, metrics):
        with open(os.path.join(self.output_dir, "evaluation_metrics.txt"), "w") as file:
            file.write(json.dumps(metrics))
            file.write("\n")
            # file.write(f"epoch {epoch}: {json.dumps(metrics)}\n")

    def preprocess_data(self, file_data):

        train_set, validation_set = train_test_split(
            file_data, train_size=self.train_size, shuffle=True
        )

        raw_train_dataset = Dataset.from_list(train_set)
        raw_validation_dataset = Dataset.from_list(validation_set)

        train_dataset = raw_train_dataset.map(
            self.preprocess_training_examples,
            batched=True,
            remove_columns=raw_train_dataset.column_names,
        )

        validation_dataset = raw_validation_dataset.map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=raw_validation_dataset.column_names,
        )

        train_dataset.set_format("torch")
        validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
        validation_set.set_format("torch")

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=8,
        )

        eval_dataloader = DataLoader(
            validation_set, collate_fn=default_data_collator, batch_size=8
        )

        # Save data for testing the method fine_tune
        # torch.save(train_dataloader, 'tests_tsp/data_for_testing/train_dataloader.pth')
        # torch.save(eval_dataloader, 'tests_tsp/data_for_testing/eval_dataloader.pth')
        # validation_dataset.save_to_disk('tests_tsp/data_for_testing/validation_dataset_fine_tune')
        # raw_validation_dataset.save_to_disk('tests_tsp/data_for_testing/raw_validation_dataset')

        return train_dataloader, raw_validation_dataset, validation_dataset, eval_dataloader

    def fine_tune(self, file_paths, max_num_train_rows=None, save=True):

        file_data = self.load_data(file_paths, max_num_train_rows)

        if not file_data:
            return

        train_dataloader, raw_validation_dataset, validation_dataset, eval_dataloader = self.preprocess_data(
            file_data)

        model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = self.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        full_metrics = {}

        for epoch in range(self.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            start_logits = []
            end_logits = []
            accelerator.print("\nEvaluation!")
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
                end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

            start_logits = np.concatenate(start_logits)
            end_logits = np.concatenate(end_logits)
            start_logits = start_logits[: len(validation_dataset)]
            end_logits = end_logits[: len(validation_dataset)]

            # Save data for testing the method compute_metrics
            np.savetxt(' tests_tsp/data_for_testing/compute_metrics/start_logits.csv', start_logits, delimiter=',')
            np.savetxt('tests_tsp/data_for_testing/compute_metrics/end_logits.csv', end_logits, delimiter=',')
            validation_dataset.save_to_disk('tests_tsp/data_for_testing/compute_metrics/validation_dataset')
            raw_validation_dataset.save_to_disk('tests_tsp/data_for_testing/compute_metrics/raw_validation_dataset')

            # print("start_logits =", start_logits)
            # print("end_logits =", end_logits)
            # print("validation_dataset =", validation_dataset)
            # print("small_validation_dataset =", small_validation_dataset)

            metrics = self.compute_metrics(
                start_logits, end_logits, validation_dataset, raw_validation_dataset
            )
            print(f"epoch {epoch}:", metrics)
            full_metrics[str(epoch)] = metrics

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            if save:
                unwrapped_model.save_pretrained(self.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.output_dir)

        if save:
            self.save_metrics(full_metrics)


    def answer_question(self, question, context):
        directory_path = os.path.dirname(os.path.abspath(__file__))
        model_checkpoint = directory_path + "/bert-finetuned-squad-accelerate"
        question_answerer = pipeline("question-answering", model=model_checkpoint)
        answer_dic = question_answerer(question=question, context=context)
        return answer_dic


if __name__ == "__main__":
    model = TSPQAModel("bert-base-cased")
    directory_path = '/Users/yaizaaragonessoria/Documents/Constructor/Chatbot for QC/chatbot4quantum/create_data/data/SE_challenges/'
    # directory_path = '/Users/yaizaaragonessoria/Documents/Constructor/chatbot4quantum/Chatbot for QC/tsp/tests_tsp/data_for_testing/'
    file_paths = [directory_path + 'dataset_tsp_distances_very_small.txt',
                  directory_path + 'dataset_tsp_cities_very_small.txt']

    examples = [
        {
            "id": "b5beacf726bf4a5284069dd67a50f5b3",
            "context": "I need to plan a road trip between three cities – New York City, Tokyo, and London. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (2.67), Tokyo to London (78.32), and London to New York City (2.5). I want to optimize for the shortest overall distance. Can you assist with that?",
            "question": "What is the distance between New York City and Tokyo?",
            "answers": {
                "text": [
                    "2.67"
                ],
                "answer_start": [
                    223
                ]
            }
        },
        {
            "id": "605b87c572c44fe9b31e1080a474534b",
            "context": "I need to plan a road trip between three cities – New York City, Tokyo, and London. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (2.67), Tokyo to London (78.32), and London to New York City (2.5). I want to optimize for the shortest overall distance. Can you assist with that?",
            "question": "What is the distance between Tokyo and London?",
            "answers": {
                "text": [
                    "78.32"
                ],
                "answer_start": [
                    247
                ]
            }
        },
        {
            "id": "69dfe775fefa4701bcf1edac10e97571",
            "context": "I need to plan a road trip between three cities – New York City, Tokyo, and London. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (2.67), Tokyo to London (78.32), and London to New York City (2.5). I want to optimize for the shortest overall distance. Can you assist with that?",
            "question": "What is the distance between London and New York City?",
            "answers": {
                "text": [
                    "2.5"
                ],
                "answer_start": [
                    284
                ]
            }
        },
        {
            "id": "59c9aef45f8b45abb6562dba925a7e0b",
            "context": "I need to plan a road trip between three cities – New York City, Tokyo, and Paris. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (26.08), Tokyo to Paris (87.73), and Paris to New York City (44.76). I want to optimize for the shortest overall distance. Can you assist with that?",
            "question": "What is the distance between New York City and Tokyo?",
            "answers": {
                "text": [
                    "26.08"
                ],
                "answer_start": [
                    222
                ]
            }
        },
        {
            "id": "146b0c5db9e54214a7643a40a511fccd",
            "context": "I need to plan a road trip between three cities – New York City, Tokyo, and Beijing. Can you help me find the most efficient route based on the distances between these cities? Here are the distances: New York City to Tokyo (17.02), Tokyo to Beijing (25.45), and Beijing to New York City (29.22). I want to optimize for the shortest overall distance. Can you assist with that?",
            "question": "What is the distance between Beijing and New York City?",
            "answers": {
                "text": [
                    "29.22"
                ],
                "answer_start": [
                    288
                ]
            }
        },
    ]

    # model.preprocess_data(examples)
    # model.fine_tune(file_paths=file_paths, max_num_train_rows=None, save=False)

    context = """Can you help me figure out the best route for a road trip I'm planning? I want to hit up Sydney, Rio de Janeiro, and Bangkok, and I need to minimize the total distance traveled. The distances between the cities are Sydney to Rio de Janeiro (82.0), Rio de Janeiro to Bangkok (53.91), and Bangkok to Sydney (70.92). Any ideas on the most efficient route?"""
    question = "Which cities wants to visit the person?"

    answer_dic = model.answer_question(question, context)

    print('predictions', answer_dic)




