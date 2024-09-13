import os

from django.apps import AppConfig
from chatbot.interface import Chatbot, FileManager
import chatbot.logic_engine as le
from chatbot.LLM_QA import BertQA
from chatbot.TSP_QAmodel import TSPQAModel

class MessageConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'message'

    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # # Create an instance of the Chatbot class, passing the data folder path
    # data_folder_path = f'{base_directory}/chatbot/data/gate_questions/'
    # chatbot = Chatbot(data_folder_path, le)

    # Initialize the chatbot with the specified checkpoint file
    data_folder_path = f'{base_directory}/chatbot/data/gate_questions/'
    classifbert = Chatbot(data_folder_path, le)

    # Initialize the chatbot with the specified checkpoint file
    checkpoint_folder_path = f'{base_directory}/chatbot/model/'
    file_manager = FileManager(checkpoint_folder_path)
    file_manager.get_latest_file()

    checkpoint_path = checkpoint_folder_path + file_manager.file_name

    classifbert.initialize(checkpoint_path, map_location='cpu', retraining_bound=20)

    bert_qa = BertQA(train_from_scratch=False)

    # TSP QA model
    tspQAmodel = TSPQAModel("bert-base-cased")


