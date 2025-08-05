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
from ml_processor import MLModelSelector
import streamlit.components.v1 as stc
import numpy as np
import os
import holidays
import matplotlib.pyplot as plt
import uuid


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
    if "data_file" not in st.session_state:
        st.session_state["data_file"] = f"dataset_{uuid.uuid4().hex}.csv"
    data_file = st.session_state["data_file"]
    data_processor = DataProcessor()
 
    with st.sidebar:
        st.image(image, use_column_width=True)
        selected = option_menu("ML Process", ["Data Preprocessing", "Data Analysis", "Feature Engineering", "Machine Learning"],  icons=["cloud-upload", "bar-chart-fill", "gear-fill", "robot"], menu_icon="list-task", default_index=0)
    
    if selected == "Data Preprocessing":
        st.header('Data Preprocessing')
        data_processor.import_and_merge_data()
        if data_processor.data is not None:
                data_processor.save_data_to_csv(data_file)

        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            data_processor.read_data_from_csv(data_file)

        if st.button("Delete Current Data"):
            data_processor.clear_data(data_file)
            st.success("Data has been cleared.")

        st.subheader("Drop or Rename")
        with st.expander("Drop selected rows or columns"):
            if data_processor.data is not None:
                data_processor.save_data_to_csv(data_file)
                if st.checkbox("Drop selected columns"):
                    droping_columns(data_processor, data_file)
                if st.checkbox("Drop selected time range rows"):
                    droping_rows(data_processor, data_file)
                if st.checkbox("Rename selected column"):
                    rename_column(data_processor, data_file)
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
        data_processor.read_data_from_csv(data_file)
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

        if os.path.exists(data_file):
            data_processor.read_data_from_csv(data_file)

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
                    pacf_selected_lags(fe, datetime_column)

                st.subheader("Lag Features")
                with st.expander("Create Lag Features"):
                    create_lags(fe, datetime_column)

                st.subheader("Rolling Window Features")
                with st.expander("Create Rolling Window Features"):
                    rolling_window(fe)

                st.subheader("Time-Based Features")
                with st.expander("Extract Time-Based Features"):
                    time_features(fe, datetime_column)

                st.subheader("Add Holidays")
                with st.expander("Add Holidays"):
                    add_holidays(fe, datetime_column)

                data_processor.data = fe.data
                data_processor.save_data_to_csv(data_file)
            else:
                st.warning("No data found. Please check your uploaded dataset.")
        else:
            st.warning("Dataset file not found. Please upload or create a dataset.")

    elif selected == "Machine Learning":
        st.header("Machine Learning")

        if os.path.exists(data_file):
            # Load dataset
            data_processor.read_data_from_csv(data_file)
            data = data_processor.data

            # Clean column names
            data.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in data.columns]

            datetime_column = st.selectbox('Select Datetime Column (for index)', options=data.columns)
            # Set the selected column as the datetime index
            data[datetime_column] = pd.to_datetime(data[datetime_column])
            data.set_index(datetime_column, inplace=True)
            
            target_column = st.selectbox('Select Target Column', options=[col for col in data.columns if col != datetime_column])

           
            st.subheader("Hyperparameters")
            hyperparameter_mode = st.selectbox("Hyperparameter Mode", ["Manual", "Auto"])
            if hyperparameter_mode == "Manual":
                # Example for XGBoost hyperparameters
                xgb_learning_rate = st.number_input('XGBoost Learning Rate', value=0.1, min_value=0.0001, max_value=1.0, step=0.01)
                xgb_max_depth = st.number_input('XGBoost Max Depth', value=5, min_value=1, max_value=10, step=1)
                xgb_n_estimators = st.number_input('XGBoost N Estimators', value=100, min_value=10, max_value=1000, step=10)

                # For LightGBM hyperparameters
                lgbm_learning_rate = st.number_input('LightGBM Learning Rate', value=0.1, min_value=0.0001, max_value=1.0, step=0.01)
                lgbm_max_depth = st.number_input('LightGBM Max Depth', value=-1, min_value=-1, max_value=50, step=1)  # -1 for no limit
                lgbm_n_estimators = st.number_input('LightGBM N Estimators', value=100, min_value=10, max_value=1000, step=10)

                # For CatBoost hyperparameters
                catboost_learning_rate = st.number_input('CatBoost Learning Rate', value=0.1, min_value=0.0001, max_value=1.0, step=0.01)
                catboost_depth = st.number_input('CatBoost Depth', value=6, min_value=1, max_value=16, step=1)
                catboost_iterations = st.number_input('CatBoost Iterations', value=1000, min_value=10, max_value=5000, step=10)

                # Collecting the hyperparameters into dictionaries for each model
                lgbm_params = {'learning_rate': lgbm_learning_rate, 'max_depth': lgbm_max_depth, 'n_estimators': lgbm_n_estimators}
                catboost_params = {'learning_rate': catboost_learning_rate, 'depth': catboost_depth, 'iterations': catboost_iterations}
                xgb_params = {'learning_rate': xgb_learning_rate, 'max_depth': xgb_max_depth, 'n_estimators': xgb_n_estimators}
                

                # Initialize model selector with the dataset and hyperparameters
                model_selector = MLModelSelector(data, target_column=target_column, hyperparameter_mode=hyperparameter_mode, xgb_params=xgb_params, lgbm_params=lgbm_params, catboost_params=catboost_params)
            else:
                # Indicate automatic hyperparameter tuning will be applied
                st.write("Automatic hyperparameter tuning will be applied for all models.")
                # Initialize model selector with the dataset and hyperparameters
                model_selector = MLModelSelector(data, target_column=target_column, hyperparameter_mode=hyperparameter_mode)
        

            st.subheader("Train and Evaluate")
            if st.button('Train Models and Evaluate'):
                with st.spinner('Training and Evaluating...'):
                    model_selector.fit()
                    # Using the new method that returns both metrics and predictions
                    mse_table, predictions_df = model_selector.evaluate_and_predict_models()
                    st.success('Training and evaluation completed successfully!')
                    
                    # Displaying the metrics
                    st.write("Model Evaluation Metrics:")
                    st.table(mse_table)
                    
                    # Displaying the predictions
                    st.write("Model Predictions vs. Actual Values:")
                    st.table(predictions_df)

        else:
            st.warning("Dataset file not found. Please upload or create a dataset.")



            

if __name__ == "__main__":
    main()
