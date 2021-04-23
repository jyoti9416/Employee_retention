from datetime import datetime
import random

class Config:
    """
    Class for configuration instance attributes
    """

    def __init__(self):
        self.training_data_path = 'data/training_data'
        self.training_database = 'training'
        self.prediction_data_path = 'data/prediction_data'
        self.prediction_database = 'prediction'

    def get_run_id(self):
        """
        it will generate unique run IDs

        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime('%H%M%S')

        return str(self.date)+'_'+str(self.current_time)+'_'+str(random.randint(100000000,999999999))