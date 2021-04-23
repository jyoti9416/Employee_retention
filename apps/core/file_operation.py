import pickle
import os
import shutil
from apps.core.logger import Logger

class FileOperation:
    """
    class for file operation
    """
    def __init__(self,run_id,data_path,mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id,'FileOperation',mode)

    def save_model(self,model,file_name):
        try:
            self.logger.info('start of Save models script')
            path = os.path.join('apps/models/',file_name)

            if os.path.isdir(path):
                shutil.rmtree('apps/models')
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path + '/' + file_name+'.sav','wb') as f:
                pickle.dump(model,f)

            self.logger.info('Model File' + file_name + 'saved')
            self.logger.info('End of Save Models')
            return 'success'
        except Exception as e:
            self.logger.exception('Exception raised while saving models: %s'%e)
            raise Exception()

    def load_model(self,cluster_number):
        try:
            self.logger.info('start of finding correct model')
            self.cluster_number = cluster_number
            self.folder_name = 'apps/models'
            self.list_of_files = os.listdir(self.folder_name)

            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue

            self.model_name = self.model_name.split('.')[0]
            self.logger.info('Found correct model')

            return self.model_name
        except Exception as e:
            self.logger.info('Exception raised while finding correct model'+str(e))
            raise Exception()