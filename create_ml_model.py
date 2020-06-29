import xgboost as xgb
import pandas as pd
import numpy as np

def create_xb_model():
    #do things
    x_data = pd.read_csv('assembled_stat_matrix.csv')
    y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
    print(x_data)
    print(y_label)

def predict():
    #do predict things
    x = 2

def main():
    create_xb_model()
    #have a create prev5 and combine 5 to do the predict
    predict()

if __name__ == '__main__':
    main()
