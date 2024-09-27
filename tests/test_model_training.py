
import pandas as pd
from model_training import load_data, train_all_models

def test_load_data():
    # Assuming the data file exists for testing
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert 'price' in data.columns

