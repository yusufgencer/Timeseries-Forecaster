import sys, os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'streamlit_app'))
from ml_processor import MLModelSelector

def test_prepare_data_split_date_no_overlap():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({'y': range(10), 'x': range(10)}, index=dates)
    selector = MLModelSelector(df, target_column='y', hyperparameter_mode='Manual', split_date='2020-01-06')
    assert selector.X_train.index.max() < pd.Timestamp('2020-01-06')
    assert selector.X_test.index.min() >= pd.Timestamp('2020-01-06')
    assert len(selector.X_train) + len(selector.X_test) == len(df)
