import pandas as pd
import holidays
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import streamlit as st

class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def create_lag_features(self, datetime_column, columns, lags):
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        for column in columns:
            for lag in lags:
                self.data[f'{column}_lag_{lag}'] = self.data[column].shift(lag)

    def analyze_autocorrelation(self, column, nlags=40, alpha=None):
        """
        Plot ACF and PACF for a given column.
        """
        series = self.data[column]
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        plot_acf(series, ax=ax[0], lags=nlags, alpha=alpha)  # Confidence intervals are drawn at alpha
        plot_pacf(series, ax=ax[1], lags=nlags, method='ywm', alpha=alpha)  # method='ywm' is recommended for PACF

        plt.show()

    def add_pacf_based_lags(self, datetime_column, column, min_lag=14, max_lag=40):
        """
        Add lag features based on partial autocorrelation significance.
        """
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        pacf_values = pacf(self.data[column], nlags=max_lag, method='ywm')

        # Identifying significant lags: where PACF value is above threshold (e.g., 0.2)
        significant_lags = [i for i, value in enumerate(pacf_values) if abs(value) > 0.2 and i >= min_lag]

        for lag in significant_lags:
            self.data[f'{column}_pacf_lag_{lag}'] = self.data[column].shift(lag)

        st.write(f"Significant PACF lags for {column}: {significant_lags}")

    def create_rolling_window_features(self, columns, window_size):
        for column in columns:
            self.data[f'{column}_rolling_mean_{window_size}'] = self.data[column].rolling(window=window_size).mean()
            self.data[f'{column}_rolling_sum_{window_size}'] = self.data[column].rolling(window=window_size).sum()

    def create_ema_features(self, columns, span):
        """
        Create Exponential Moving Average (EMA) features.

        :param columns: List of column names to apply EMA.
        :param span: The span parameter for the EMA, which defines the degree of weighting decrease.
        """
        for column in columns:
            self.data[f'{column}_ema_{span}'] = self.data[column].ewm(span=span, adjust=False).mean()

    def create_rolling_variance_std_features(self, columns, window_size):
        """
        Create rolling variance and standard deviation features.

        :param columns: List of column names to apply rolling computations.
        :param window_size: The size of the moving window.
        """
        for column in columns:
            self.data[f'{column}_rolling_variance_{window_size}'] = self.data[column].rolling(window=window_size).var()
            self.data[f'{column}_rolling_std_{window_size}'] = self.data[column].rolling(window=window_size).std()

    def create_rolling_min_max_features(self, columns, window_size):
        """
        Create rolling minimum and maximum features.

        :param columns: List of column names to apply rolling computations.
        :param window_size: The size of the moving window.
        """
        for column in columns:
            self.data[f'{column}_rolling_min_{window_size}'] = self.data[column].rolling(window=window_size).min()
            self.data[f'{column}_rolling_max_{window_size}'] = self.data[column].rolling(window=window_size).max()

    def create_time_based_features(self, datetime_column, features_to_create):
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        
        if 'year' in features_to_create:
            self.data['year'] = self.data[datetime_column].dt.year
        if 'quarter' in features_to_create:  # Added 'quarter'
            self.data['quarter'] = self.data[datetime_column].dt.quarter
        if 'month' in features_to_create:
            self.data['month'] = self.data[datetime_column].dt.month
        if 'week_of_year' in features_to_create:  # Added 'week_of_year'
            self.data['week_of_year'] = self.data[datetime_column].dt.isocalendar().week
        if 'day' in features_to_create:
            self.data['day'] = self.data[datetime_column].dt.day
        if 'weekday' in features_to_create:
            self.data['weekday'] = self.data[datetime_column].dt.weekday
        if 'is_weekend' in features_to_create:  # Added 'is_weekend'
            self.data['is_weekend'] = self.data[datetime_column].dt.weekday.isin([5, 6]).astype(int)
        if 'hour' in features_to_create:
            self.data['hour'] = self.data[datetime_column].dt.hour


    def add_holidays(self, datetime_column, country_code):
        # Ensure country_holidays is properly initialized with the country's code.
        country_holidays = holidays.CountryHoliday(country_code)

        # Convert datetime_column to datetime format if it's not already.
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])

        # Apply a function to check if each date is a holiday.
        # The try-except block catches any errors that might occur during the process.
        def check_if_holiday(date):
            try:
                # Ensure the date is in the correct format by converting it to a date (if it's a datetime object).
                # This step may need adjustment based on your specific date format and the requirements of the holidays library.
                date = date.date() if hasattr(date, 'date') else date
                return int(date in country_holidays)
            except Exception as e:
                print(f"Error processing date {date}: {e}")
                return 0

        self.data['is_holiday'] = self.data[datetime_column].apply(check_if_holiday)


def acf_pacf_plots(fe):
    target_column = st.selectbox('Select column to analyze', options=fe.data.columns)
    nlags = st.number_input('Number of lags', min_value=1, max_value=100, value=40)
    # Generate and display ACF and PACF plots
    fe.analyze_autocorrelation(target_column, nlags=nlags)
    st.pyplot(plt)  # Assuming analyze_autocorrelation method plots the figure or use st.plotly_chart for Plotly

def pacf_selected_lags(fe, datetime_column):
    column_to_analyze_pacf = st.selectbox('Select a column for PACF-based Lags', options=fe.data.columns)
    max_lag = st.number_input('Max lags to consider for PACF', min_value=1, max_value=24*7, value=72)
    min_lag = st.number_input('Min lags to consider for PACF', min_value=1, max_value=24*7, value=24)
    auto_apply_pacf = st.button('Analyze and Autoselect Lags using PACF')
    if auto_apply_pacf:
        fe.add_pacf_based_lags(datetime_column, column_to_analyze_pacf, min_lag, max_lag)
        st.success(f"Significant Lags were used to automatically create new Lag Features based on PACF for {column_to_analyze_pacf}.")

def create_lags(fe, datetime_column):
    columns_to_lag = st.multiselect('Select columns for lag features', options=fe.data.columns)
    lags = st.text_input('Enter lag periods (comma-separated, e.g., 1,2,3)', '1')
    apply_lags = st.button('Apply Lag Features', key='apply_basic_lags')
    if apply_lags and columns_to_lag and lags:
        lag_periods = [int(lag) for lag in lags.split(',')]
        fe.create_lag_features(datetime_column, columns_to_lag, lag_periods)
        st.success("Lag features applied.")
        st.write(fe.data.head())

def rolling_window(fe):
    columns_to_roll = st.multiselect('Select columns for rolling features', options=fe.data.columns)
    window_size = st.text_input('Enter rolling window size', '24')

    # Basic Rolling Window Features
    apply_basic_rolling = st.checkbox("Apply Basic Rolling Mean and Sum")

    # Advanced Rolling Window Features
    st.markdown("#### Advanced Rolling Window Features")

    # EMA Features
    apply_ema = st.checkbox("Apply EMA")
    span_for_ema = st.number_input('Enter span for EMA', min_value=2, value=12, step=1, format='%d') if apply_ema else 12

    # Variance and STD Features
    apply_variance_std = st.checkbox("Apply Rolling Variance and STD")

    # Min and Max Features
    apply_min_max = st.checkbox("Apply Rolling Min and Max")

    if st.button('Apply Selected Rolling Window Features'):
        if apply_basic_rolling and columns_to_roll and window_size.isdigit():
            fe.create_rolling_window_features(columns_to_roll, int(window_size))
            st.success("Basic Rolling window features applied.")

        if apply_ema and columns_to_roll:
            fe.create_ema_features(columns_to_roll, span_for_ema)
            st.success("EMA features applied.")
        
        if apply_variance_std and columns_to_roll and window_size.isdigit():
            fe.create_rolling_variance_std_features(columns_to_roll, int(window_size))
            st.success("Rolling Variance and STD features applied.")
            
        if apply_min_max and columns_to_roll and window_size.isdigit():
            fe.create_rolling_min_max_features(columns_to_roll, int(window_size))
            st.success("Rolling Min and Max features applied.")
        
        st.write(fe.data.head())

def time_features(fe, datetime_column):
    options = ['year', 'quarter', 'month', 'week_of_year', 'day', 'weekday', 'hour', 'is_weekend']
    selected_time_features = st.multiselect('Select time-based features to extract', options=options, default=options)
    apply_time_features = st.button('Apply Time-Based Features')
    if apply_time_features and selected_time_features:
        fe.create_time_based_features(datetime_column, selected_time_features)
        st.success("Selected time-based features applied.")
        st.write(fe.data.head())

def add_holidays(fe, datetime_column):
    country_code = st.selectbox('Select country', options=list(holidays.list_supported_countries().keys()))
    apply_holidays = st.button('Apply Holidays')
    if apply_holidays:
        fe.add_holidays(datetime_column, country_code)
        st.success("Holidays added.")
        st.write(fe.data.head())