import pandas as pd
import numpy as np
import json
from sklearn.impute import KNNImputer
from apps.core.logger import Logger


class Preprocessor:
    def __init__(self, run_id, data_path, mode):
        self.run_id = run_id
        self.data_path = data_path
        self.logger = Logger(self.run_id, 'Preprocessor', mode)

    def get_data(self):
        try:
            self.logger.info('start reading dataset')
            self.data = pd.read_csv(self.data_path + '_validation/InputFile.csv')
            self.logger.info('end reading dataset')
            return self.data
        except Exception as e:
            self.logger.exception('Exception raised while reading dataset: %s' + str(e))
            raise Exception()

    def drop_columns(self, data, columns):
        self.data = data
        self.columns = columns

        try:
            self.logger.info('Start of Droping Columns...')
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger.info('End of Droping Columns...')
            return self.useful_data
        except Exception as e:
            self.logger.exception('Exception raised while dropping columns: %s' + str(e))
            raise Exception()

    def is_null_present(self, data: pd.DataFrame):
        self.null_present = False
        try:
            self.logger.info('Start of finding missing values...')
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present:
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing_values_count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(self.data_path + '_validation/' + 'null_values.csv')
            self.logger.info('End of finding missing values...')
            return self.null_present
        except Exception as e:
            self.logger.exception('Exception raised while dropping columns: %s' + str(e))
            raise Exception()

    def impute_missing_values(self, data: pd.DataFrame):
        self.data = data
        try:
            self.logger.info('Start of Imputing missing values...')
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger.info('End of Imputing missing values...')
            return self.new_data
        except Exception as e:
            self.logger.exception('Exception raised while imputing missing values: %s' + str(e))
            raise Exception()

    def feature_encoding(self, data: pd.DataFrame):
        try:
            self.logger.info('Start of Feature encoding ...')
            self.new_data = data.select_dtypes(include=['object']).copy()
            for col in self.new_data.columns:
                self.new_data = pd.get_dummies(self.new_data, columns=[col], prefix=[col], drop_first=True)
            self.logger.info('End of Imputing missing values...')
            return self.new_data
        except Exception as e:
            self.logger.exception('Exception raised while Feature encoding: %s' + str(e))
            raise Exception()

    def split_features_label(self, data, label_name):
        self.data = data
        try:
            self.logger.info('Start of splitting features and label ...')
            self.X = self.data.drop(labels=label_name, axis=1)
            self.y = self.data[label_name]
            self.logger.info('End of Feature encoding ...')

            return self.X, self.y
        except Exception as e:
            self.logger.exception('Exception raised while splitting feature and labels: %s' + str(e))
            raise Exception()

    def final_predictset(self, data: pd.DataFrame):
        try:
            self.logger.info('Start of building final_predictset ...')
            with open('apps/database/columns.json', 'r') as f:
                data_columns = json.load(f)['data_columns']
                f.close()
            df = pd.DataFrame(data=None, columns=data_columns)
            df_new = pd.concat([df, data], ignore_index=True, sort=False)
            data_new = df_new.fillna(0)
            self.logger.info('End of building final_predictset ...')
            return data_new
        except ValueError:
            self.logger.exception('ValueError raised while building final predictset')
            raise ValueError
        except KeyError:
            self.logger.exception('KeyError raised while building final predictset')
            raise KeyError
        except Exception as e:
            self.logger.exception('Exception raised while building final predictset: %s' % e)
            raise e

    def preprocess_trainset(self):
        try:
            self.logger.info('Start of building preprocessing training set ...')
            data = self.get_data()
            data = self.drop_columns(data, ['empid'])
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation
            # create separate features and labels
            self.X, self.y = self.split_features_label(data, label_name='left')
            self.logger.info('End of building final_predictset ...')
            return self.X, self.y
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predictset(self):
        try:
            self.logger.info('Start of building preprocessing training set ...')
            data = self.get_data()
            # data = self.drop_columns(data,['empid'])
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation
            # create separate features and labels
            self.X, self.y = self.split_features_label(data, label_name='left')
            self.logger.info('End of building final_predictset ...')
            return self.X, self.y
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception

    def preprocess_predict(self, data):
        try:
            self.logger.info('Start of Preprocessing...')
            cat_df = self.feature_encoding(data)
            data = pd.concat([data, cat_df], axis=1)
            # drop categorical column
            data = self.drop_columns(data, ['salary'])
            # check if missing values are present in the data set
            is_null_present = self.is_null_present(data)
            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = self.impute_missing_values(data)  # missing value imputation

            data = self.final_predictset(data)
            self.logger.info('End of Preprocessing...')
            return data
        except Exception:
            self.logger.exception('Unsuccessful End of Preprocessing...')
            raise Exception
