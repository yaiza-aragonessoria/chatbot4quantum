# copy from file multi-class.py in test-bert project on 21.08.2023

# Import required libraries
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup, logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
import numpy as np
import pandas as pd
import os
import datetime


class BertTextClassifier:
    def __init__(self, val_ratio, batch_size, epochs, num_labels):
        self.text = None
        self.labels = None
        self.df = None
        self.token_id = None
        self.attention_masks = None
        self.labels_tensor = None
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed_val = 42
        self.num_labels = num_labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.pretrained_model = None
        self.starting_epoch = None

        # initialize the model
        self.create_model()
        self.create_optimizer()

    def create_model(self):
        # Create the BERT-based model for sequence classification
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

    def create_optimizer(self):
        # Create an AdamW optimizer for training the model
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def set_seed(self, seed=42):
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_device(self):
        # Set the device to CUDA (GPU) if available; otherwise, use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the selected device (GPU or CPU)
        self.model.to(self.device)

    def load_data(self, file_paths, max_num_rows=None):
        data = []
        # Iterate through the file paths and read data from each file
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    for line in file.readlines():
                        split = line.split('\t')
                        data.append({'label': int(split[0]),
                                     'text': split[1].rstrip()})
            except FileNotFoundError:
                print(f"File not found: {file_path}")

        if max_num_rows:
            random.shuffle(data)  # Shuffle the data randomly

            # Select the specified number of entries
            data = data[:max_num_rows]

        self.df = pd.concat([pd.DataFrame(data)])

        self.text = self.df.text.values
        self.labels = self.df.label.values

    def tokenize_data(self, data):
        # Tokenize and preprocess text data
        tokenized_data = self.tokenizer(
            data,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
        )
        return {
            'input_ids': torch.tensor([tokenized_data['input_ids']]),
            'attention_mask': torch.tensor([tokenized_data['attention_mask']]),
        }

    def preprocess_data(self):
        # Preprocess data and prepare tensors
        token_id = []
        attention_masks = []

        for sample in self.text:
            encoding_dict = self.tokenize_data(sample)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        # convert data and labels to tensors of pytorch
        self.token_id = torch.cat(token_id, dim=0)
        self.attention_masks = torch.cat(attention_masks, dim=0)
        self.labels_tensor = torch.tensor(self.labels)

    def split_data(self):
        # Split data into training and validation sets stratified by labels
        train_idx, val_idx = train_test_split(
            np.arange(len(self.labels_tensor)),
            test_size=self.val_ratio,
            shuffle=True,
            stratify=self.labels_tensor
        )

        train_set = TensorDataset(self.token_id[train_idx],
                                  self.attention_masks[train_idx],
                                  self.labels_tensor[train_idx])

        val_set = TensorDataset(self.token_id[val_idx],
                                self.attention_masks[val_idx],
                                self.labels_tensor[val_idx])

        self.train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=self.batch_size
        )

        self.val_dataloader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set),
            batch_size=self.batch_size
        )

    def create_scheduler(self):
        # Create a linear learning rate scheduler with warm-up
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    def load_checkpoint(self, train_from_scratch=True, path_to_model=None, map_location='cpu'):
        if train_from_scratch:
            self.starting_epoch = 0
        else:
            if path_to_model:
                if os.path.isfile(path_to_model):
                    self.pretrained_model = torch.load(path_to_model, map_location=map_location)

                    if self.pretrained_model:
                        self.starting_epoch = 0
                        self.model.load_state_dict(self.pretrained_model['model_state_dict'])
                        self.optimizer.load_state_dict(self.pretrained_model['optimizer_state_dict'])
                else:
                    print('No saved model, so fine-tuning from scratch')
                    self.starting_epoch = 0
            else:
                print('No path to model provided')
                print('Fine-tuning from scratch')
                self.starting_epoch = 0

    def evaluate(self, dataloader):
        # Evaluate the model's performance

        # Set the model to evaluation mode
        self.model.eval()

        total_loss = 0
        total_accuracy = 0

        for batch in dataloader:
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            # Disable gradient calculation during evaluation
            with torch.no_grad():
                # Forward pass through the model to get predictions and loss
                outputs = self.model(**inputs)

                # Extract the loss and logits (raw outputs of the model)
                loss = outputs[0]
                logits = outputs[1]

                # Accumulate the total loss for the entire data
                total_loss += loss.item()

                # Detach the logits from the computation graph and convert them to a NumPy array
                logits = logits.detach().cpu().numpy()

                # Convert the labels tensor to a NumPy array
                label_ids = inputs['labels'].cpu().numpy()

                # Calculate the accuracy for the current batch and accumulate it
                total_accuracy += self.flat_accuracy(logits, label_ids)

        # Calculate average loss and accuracy across all batches
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        return avg_loss, avg_accuracy

    def flat_accuracy(self, preds, labels):
        # Calculate flat accuracy for classification
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, train_from_scratch=True, path_to_model=None, map_location='cpu', data_paths=None, save=True):
        if data_paths:
            self.load_data(data_paths, max_num_rows=30000)
            self.preprocess_data()
            self.split_data()
            self.create_scheduler() # it is dependant of the data, so it cannot be initialised directly with the model

            self.load_checkpoint(train_from_scratch, path_to_model, map_location)

            training_stats = []

            print("starting_epoch ", self.starting_epoch)
            print("epochs ", self.epochs)

            for epoch_i in range(self.starting_epoch, self.epochs):
                total_train_loss = 0
                self.model.train()

                for step, batch in enumerate(self.train_dataloader):
                    if step % 40 == 0 and not step == 0:
                        print('Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.'.format(step, len(self.train_dataloader),
                                                                               total_train_loss / step))

                    batch = tuple(b.to(self.device) for b in batch)
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2],
                              }

                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                avg_train_loss = total_train_loss / len(self.train_dataloader)
                training_loss, training_accuracy = self.evaluate(self.train_dataloader)
                validation_loss, validation_accuracy = self.evaluate(self.val_dataloader)

                print("")
                print("Epoch: {}/{}".format(epoch_i + 1, self.epochs))
                print("Training Loss: {}".format(avg_train_loss))
                print("Training Accuracy: {}".format(training_accuracy))
                print("Validation Loss: {}".format(validation_loss))
                print("Validation Accuracy: {}".format(validation_accuracy))
                print("")

                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Training Accuracy': training_accuracy,
                        'Validation Loss': validation_loss,
                        'Validation Accuracy': validation_accuracy
                    }
                )

            if save:
                current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                LLM_file_name = f"quantum_LLM_{current_date}.pth"
                directory = os.path.dirname(os.path.abspath(__file__)) + "/model/"

                torch.save({
                    'num_labels': self.num_labels,
                    'epoch': epoch_i + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': training_loss,
                }, directory + LLM_file_name)
        else:
            print("Model couldn't train because no data was provided.")

if __name__ == "__main__":
    # PARAMETERS
    data_paths = ['./data/gate_questions/generated_questions.txt',
                  './data/gate_questions/dataset_tsp_for_classification.txt']
    path_to_pretrained_model = './model/quantum_LLM.pth'
    val_ratio = 0.2
    batch_size = 16  # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
    epochs = 4  # Set the number of training epochs
    # batch_size = 32  # Set the batch size for the data loader
    num_labels = 4

    # logging.set_verbosity_info()

    classifier = BertTextClassifier(val_ratio, batch_size, epochs, num_labels)
    classifier.set_seed()
    classifier.set_device()
    classifier.train(train_from_scratch=True, data_paths=data_paths, path_to_model=None, save=True)
