import time
import json
from openai import OpenAI

class OpenAIFineTuner:
    """
    Class to fine tune OpenAI models
    """
    def __init__(self, training_file_path, model_name, suffix, open_api_client):
        self.training_file_path = training_file_path
        self.model_name = model_name
        self.suffix = suffix
        self.open_api_client = open_api_client
        self.file_object = None
        self.fine_tuning_job = None
        self.model_id = None

    def create_openai_file(self):
        self.file_object = self.open_api_client.files.create(
            file=open(self.training_file_path, "rb"),
            purpose="fine-tune",
        )

    def wait_for_file_processing(self, sleep_time=20):
        while self.file_object.status != 'processed':
            time.sleep(sleep_time)
            self.file_object.refresh()
            print("File Status: ", self.file_object.status)

    def create_fine_tuning_job(self):
        self.fine_tuning_job = self.open_api_client.fine_tuning.jobs.create(
            training_file=self.file_object.id,
            model=self.model_name,
            suffix=self.suffix,
        )

    def wait_for_fine_tuning(self, sleep_time=45):
        while self.fine_tuning_job.status != 'succeeded':
            time.sleep(sleep_time)
            
            # refresh the job status
            self.fine_tuning_job.status = self.open_api_client.fine_tuning.jobs.retrieve(self.fine_tuning_job.id).status
            print("Job Status: ", self.fine_tuning_job.status)

    def retrieve_fine_tuned_model(self):
        self.model_id = self.open_api_client.fine_tuning.jobs.retrieve(self.fine_tuning_job.id).fine_tuned_model
        return self.model_id

    def fine_tune_model(self):
        self.create_openai_file()
        self.wait_for_file_processing()
        self.create_fine_tuning_job()
        self.wait_for_fine_tuning()
        return self.retrieve_fine_tuned_model()
