import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from data_processor import DataProcessor, droping_rows, droping_columns, rename_column
from data_analyzer import (DataAnalyzer, correlation_matrix, distribution, outlier_visualization,
                           outlier_statistics, feature_target_correlations, pca_visualization)
from feature_engineer import (FeatureEngineering, acf_pacf_plots, pacf_selected_lags, create_lags,
                              rolling_window, time_features, add_holidays)
from ml_processor import MLProcessor
import streamlit.components.v1 as stc
import numpy as np
import os
import holidays
import matplotlib.pyplot as plt


def plot_data(data, selected_columns, datetime_column):
    fig = go.Figure()

    for column in selected_columns:
        fig.add_trace(go.Scatter(x=data[datetime_column], y=data[column], mode='lines', name=column))

    fig.update_layout(title='Selected Columns Visualization',
                      xaxis_title=datetime_column,
                      yaxis_title='Value')

    st.plotly_chart(fig)

def main():
    image = "streamlit_app/timelogo.png" 
    st.title('Timeseries Forecaster App')
    data_processor = DataProcessor()
 
    with st.sidebar:
        st.image(image, use_column_width=True)
        selected = option_menu("ML Process", ["Data Preprocessing", "Data Analysis", "Feature Engineering", "Machine Learning"],  icons=["cloud-upload", "bar-chart-fill", "gear-fill", "robot"], menu_icon="list-task", default_index=0)
    
    if selected == "Data Preprocessing":
        st.header('Data Preprocessing')
        data_processor.import_and_merge_data()
        if data_processor.data is not None:
                data_processor.save_data_to_csv("dataset.csv")

        if os.path.exists("dataset.csv") and os.path.getsize("dataset.csv") > 0:
            data_processor.read_data_from_csv("dataset.csv")

        if st.button("Delete Current Data"):
            data_processor.clear_data()
            st.success("Data has been cleared.")

        st.subheader("Drop or Rename")
        with st.expander("Drop selected rows or columns"):
            if data_processor.data is not None:
                data_processor.save_data_to_csv("dataset.csv")
                if st.checkbox("Drop selected columns"):
                    droping_columns(data_processor)
                if st.checkbox("Drop selected time range rows"):
                    droping_rows(data_processor)
                if st.checkbox("Rename selected column"):
                    rename_column(data_processor)
                st.write(data_processor.data)    
            else:
                st.warning("Please upload a CSV file to enable data manipulation and visualization features.")

        st.subheader('Select Columns for Visualization')
        if data_processor.data is not None:
            selected_columns = st.multiselect('Select columns', data_processor.data.columns)
            # Check if datetime column is selected
            datetime_column = st.selectbox('Select datetime column', options=data_processor.data.columns)
            if selected_columns:
                # Plot selected columns
                st.subheader('Visualization')
                # Pass preprocessed data to plot function
                plot_data(data_processor.data, selected_columns, datetime_column)
        else:
            st.warning("To visualize data, please upload a CSV file containing the relevant columns.")
        

    elif selected == "Data Analysis":
        st.header('Data Analysis') 
        data_processor.read_data_from_csv("dataset.csv")
        if data_processor.data is not None:
            analyzer = DataAnalyzer(data_processor.data)
            
            st.subheader("Summary report")
            summary_report = analyzer.summary_report()
            st.write(summary_report)

            st.subheader("Correlation Matrix Visualization")  
            with st.expander("Expand for detailed visualization options"):
                correlation_matrix(analyzer)
                
            st.subheader("Data Distribution Visualization")
            with st.expander("Unfold to tailor distribution plots to your analysis needs"):
                distribution(analyzer)

            st.subheader("Outliers Visualization")
            with st.expander("Click to customize and explore outlier visualizations"):
                outlier_visualization(analyzer)

            st.subheader("Outlier Statistics")
            with st.expander("Dive into detailed statistics to uncover insights on outliers"):
                outlier_statistics(analyzer)

            st.subheader("Feature-Target Correlation Analysis")
            if analyzer.data is not None:
                with st.expander("Unlock Insights: Analyze Feature-Target Correlations"):
                    feature_target_correlations(analyzer)

            st.subheader("PCA Visualization")
            if analyzer.data is not None:
                with st.expander("Customize and Visualize: Apply PCA to Your Data"):
                    pca_visualization(analyzer)

    elif selected == "Feature Engineering":
        st.header('Feature Engineering')

        if os.path.exists("dataset.csv"):
            data_processor.read_data_from_csv("dataset.csv")

            if not data_processor.data.empty:
                fe = FeatureEngineering(data_processor.data)

                datetime_column = st.selectbox('Select datetime column', options=fe.data.columns)
                fe.data[datetime_column] = pd.to_datetime(fe.data[datetime_column])

                st.write(fe.data)

                st.subheader("Analyze Autocorrelation and PACF")
                with st.expander("Show ACF and PACF Plots"):
                    acf_pacf_plots(fe)

                st.subheader("Add PACF-based Lags")
                with st.expander("Use PACF to Auto-select Lag Features"):
                    pacf_selected_lags(fe)

                st.subheader("Lag Features")
                with st.expander("Create Lag Features"):
                    create_lags(fe)

                st.subheader("Rolling Window Features")
                with st.expander("Create Rolling Window Features"):
                    rolling_window(fe)

                st.subheader("Time-Based Features")
                with st.expander("Extract Time-Based Features"):
                    time_features(fe)

                st.subheader("Add Holidays")
                with st.expander("Add Holidays"):
                    add_holidays(fe)

                data_processor.data = fe.data
                data_processor.save_data_to_csv("dataset.csv")
            else:
                st.warning("No data found. Please check your 'dataset.csv'.")
        else:
            st.warning("Dataset file not found. Please upload or create a dataset.")

    elif selected == "Machine Learning":
        st.header("Machine Learning")

if __name__ == "__main__":
    main()