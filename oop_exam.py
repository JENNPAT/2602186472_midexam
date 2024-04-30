import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

#designing OOP for the best model (model6)
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path 
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

        
# ModelHandler Class
class ModelHandler:
    def __init__(self, data, input_data, output_data):
        self.data = data
        self.input_data = input_data 
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        
    def fillna(self, col):
        value = np.mean(self.input_data[col])
        self.input_data[col].fillna(value, inplace=True)

    #removing some unecessary columns
    def remove(self,col):
        self.input_data.drop(columns=col, inplace=True)
        self.output_data.drop(columns=col, inplace=True)

    #i use label encoding method for converting the categorical columns
    def feature_encode(self):
        label_encoder = preprocessing.LabelEncoder()
        for col in self.input_data.select_dtypes("object"):
            self.input_data[col] = label_encoder.fit_transform(self.input_data[col])
        
    def split_data(self, test_size=0.3, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))
               
    def createModel(self):
        self.model = XGBClassifier(gamma= 0, max_depth= 3, n_estimators = 100)

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)  # Use pickle to write the model to the file

    def display_head(self, n=5):
        if self.data is not None:
            print(self.data.head(n))
        else:
            print("Data belum dimuat. Silakan muat data terlebih dahulu menggunakan metode load_data().")


#data handling
file_path = "data_D.csv"
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')
data = data_handler.data
input_df = data_handler.input_df
output_df = data_handler.output_df

#model handling
model_handler = ModelHandler(data, input_df, output_df)
model_handler.fillna("CreditScore")
model_handler.remove("Unnamed: 0")
model_handler.remove("id")
model_handler.remove("CustomerId")
model_handler.remove("Surname")
model_handler.feature_encode()
model_handler.split_data()

print("The best model (model8)")
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()

model_handler.save_model_to_file('trained_model.pkl') 









