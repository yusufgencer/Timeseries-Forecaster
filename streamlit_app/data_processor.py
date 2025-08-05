import pandas as pd
import streamlit as st
import os

class DataProcessor:
    def __init__(self):
        """
        Initialize an instance of DataProcessor.
        """
        self.data = None

    def import_and_merge_data(self):
        """
        Import and merge CSV files uploaded by the user.
        """
        uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
        if uploaded_files:
            dfs = []
            datetime_columns = []
            for i, file in enumerate(uploaded_files):
                df = pd.read_csv(file)
                datetime_column = st.selectbox(f'Select datetime column for File {i+1}', options=df.columns, key=f'datetime_column_{i}')
                datetime_columns.append(datetime_column)
                df[datetime_column] = pd.to_datetime(df[datetime_column])
                dfs.append(df)
            self.data = self.merge_dataframes(dfs, datetime_columns)

        else:
            self.data = None

    def merge_dataframes(self, dfs, datetime_columns):
        """
        Merge multiple dataframes based on datetime columns.
        
        Parameters:
        - dfs (list): List of dataframes to merge.
        - datetime_columns (list): List of datetime columns for merging.

        Returns:
        - merged_df (DataFrame): Merged dataframe.
        """
        merged_df = dfs[0]
        for i, df in enumerate(dfs[1:]):
            merged_df = pd.merge(merged_df, df, left_on=datetime_columns[i], right_on=datetime_columns[i+1], how='inner')
        return merged_df

    def preprocess_data(self, columns_to_drop):
        """
        Preprocess data by dropping selected columns.
        
        Parameters:
        - columns_to_drop (list): List of column names to drop.

        Returns:
        - self.data (DataFrame): Preprocessed dataframe.
        """
        if self.data is not None:
            self.data = self.data.drop(columns=columns_to_drop)
            return True
        else:
            return False
    
    def save_data_to_csv(self, filename):
        """
        Save dataframe to a CSV file.
        
        Parameters:
        - filename (str): Name of the CSV file.
        """
        if self.data is not None:
            self.data.to_csv(filename, index=False)

    def read_data_from_csv(self, filename):
        """
        Read data from a CSV file.
        
        Parameters:
        - filename (str): Name of the CSV file.
        """
        try:
            self.data = pd.read_csv(filename)
        except FileNotFoundError:
            st.error(f"File {filename} not found.")

    def clear_data(self, filename):
        """Clear the current DataFrame and delete the provided file if it exists."""
        self.data = None  # Clear the DataFrame
        if os.path.exists(filename):
            os.remove(filename)

    def drop_rows_in_time_range(self, datetime_column, start_date, end_date):
        """
        Drop rows from the dataset that fall within the specified time range.

        :param datetime_column: Name of the column containing datetime information.
        :param start_date: The start date of the range to drop rows, inclusive.
        :param end_date: The end date of the range to drop rows, inclusive.
        :return: Boolean indicating whether any rows were dropped.
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Ensure datetime_column is in datetime format
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        
        # Get the initial length of the DataFrame
        initial_length = len(self.data)
        
        # Filter rows that are outside the specified range
        self.data = self.data[~((self.data[datetime_column] >= start_date) & (self.data[datetime_column] <= end_date))]
        
        # Get the new length of the DataFrame after filtering
        new_length = len(self.data)
        
        # Check if the DataFrame length has changed (meaning rows were dropped)
        rows_dropped = new_length < initial_length
        
        return rows_dropped

    def rename_column(self, old_name, new_name):
        """
        Rename a column in the dataframe.

        Parameters:
        - old_name (str): The current name of the column.
        - new_name (str): The new name for the column.
        """
        if self.data is not None and old_name in self.data.columns:
            self.data.rename(columns={old_name: new_name}, inplace=True)
            return True
        else:
            st.warning(f"Column '{old_name}' not found in the dataframe.")
            return False

      
def droping_columns(data_processor, filename):
    data_processor.read_data_from_csv(filename)
    columns_to_drop = st.multiselect('Select columns to drop', data_processor.data.columns)
    if columns_to_drop:
        if st.button("Drop selected columns"):
            if data_processor.preprocess_data(columns_to_drop):
                data_processor.save_data_to_csv(filename)
                st.success("Selected rows within the time range have been successfully dropped.")
            else:
                st.error("No rows were dropped. Please check the selected time range and datetime column.")

def droping_rows(data_processor, filename):
    data_processor.read_data_from_csv(filename)
    st.write("Select time range and datetime column")
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')
    datetime = st.selectbox('Select datetime', data_processor.data.columns)
    if st.button('Drop Rows'):
        success_rows = data_processor.drop_rows_in_time_range(datetime, start_date=start_date, end_date=end_date)
        if success_rows:
            data_processor.save_data_to_csv(filename)
            st.success("Selected rows within the time range have been successfully dropped.")
        else:
            st.error("No rows were dropped. Please check the selected time range and datetime column.")

def rename_column(data_processor, filename):
    data_processor.read_data_from_csv(filename)
    old_column = st.selectbox('Select column to rename', data_processor.data.columns)
    new_column = st.text_input("Write new name for selected column")
    if st.button("Apply rename changes"):
        if data_processor.rename_column(old_column, new_column):
            data_processor.save_data_to_csv(filename)
            st.success("Selected rows within the time range have been successfully dropped.")
        else:
            st.error("No changes applied please be sure selected columns and write new name")

