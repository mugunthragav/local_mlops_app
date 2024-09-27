
import pandas as pd
from model_training import load_data, train_all_models

def test_load_data():
    # Assuming the data file exists for testing
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert 'price' in data.columns

def test_train_all_models():
    data = load_data()
    best_model_name, best_model = train_all_models(data)
    assert best_model_name is not None
    assert best_model is not None
