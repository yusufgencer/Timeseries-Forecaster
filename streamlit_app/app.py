import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Function to import and merge data
def import_and_merge_data():
    uploaded_files = st.sidebar.file_uploader("Upload CSV Files", accept_multiple_files=True)
    if uploaded_files:
        dfs = []
        datetime_columns = []
        for i, file in enumerate(uploaded_files):
            df = pd.read_csv(file)
            datetime_column = st.sidebar.selectbox(f'Select datetime column for File {i+1}', options=df.columns, key=f'datetime_column_{i}')
            datetime_columns.append(datetime_column)
            dfs.append(df)
        data = merge_dataframes(dfs, datetime_columns)
        return data
    else:
        return None

# Function to merge dataframes based on datetime column
def merge_dataframes(dfs, datetime_columns):
    # Merge dataframes based on the datetime column
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:]):
        merged_df = pd.merge(merged_df, df, on=datetime_columns[i+1], how='outer')
    return merged_df

# Function to plot data
def plot_data(data, selected_columns, datetime_column):
    fig = go.Figure()

    for column in selected_columns:
        fig.add_trace(go.Scatter(x=data[datetime_column], y=data[column], mode='lines', name=column))

    fig.update_layout(title='Selected Columns Visualization',
                      xaxis_title=datetime_column,
                      yaxis_title='Value')

    st.plotly_chart(fig)


# Function to plot correlation matrix as heatmap
def plot_correlation_heatmap(data, selected_features):
    # Calculate correlation matrix for selected features
    corr_matrix = data[selected_features].corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='Viridis')
    fig.update_layout(title='Correlation Matrix Heatmap')
    st.plotly_chart(fig)


# Main function
def main():
    st.title('XGBoost Forecaster App')

    st.sidebar.header('Import and Merge Data')
    data = import_and_merge_data()

    if data is not None:
        # Display imported data
        st.subheader('Imported and Merged Data')
        # Sidebar - Drop selected columns
        drop_columns = st.sidebar.checkbox("Drop selected columns")
        if drop_columns:
            columns_to_drop = st.sidebar.multiselect('Select columns to drop', data.columns)
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)

        st.write(data.head())

        # Sidebar - Select columns for visualization
        st.sidebar.header('Select Columns for Visualization')
        selected_columns = st.sidebar.multiselect('Select columns', data.columns)

        # Check if datetime column is selected
        datetime_column = st.sidebar.selectbox('Select datetime column', options=data.columns)

        if selected_columns:
            # Plot selected columns
            st.subheader('Visualization')
            plot_data(data, selected_columns, datetime_column)

        data_analysis = st.sidebar.checkbox("Data Analysis")

        if data_analysis:
            st.sidebar.subheader('Data Analysis Tools')
            selected_features = st.sidebar.multiselect('Select features for analysis', data.columns)

            if st.sidebar.button('Summary Statistics'):
                st.subheader('Summary Statistics')
                if data is not None and selected_features:
                    st.write(data[selected_features].describe())

            if st.sidebar.button('Correlation Matrix Heatmap'):
                st.subheader('Correlation Matrix Heatmap')
                if data is not None and selected_features:
                    plot_correlation_heatmap(data, selected_features)



if __name__ == '__main__':
    main()
