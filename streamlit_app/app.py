import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Function to import and merge data
def import_and_merge_data():
    uploaded_files = st.sidebar.file_uploader("Upload CSV Files", accept_multiple_files=True)
    if uploaded_files:
        dfs = []
        datetime_columns = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            datetime_column = st.sidebar.selectbox(f'Select datetime column for {file.name}', options=df.columns)
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

# Main function
def main():
    st.title('Your Streamlit App')

    # Sidebar - Import and merge data
    st.sidebar.header('Import and Merge Data')
    data = import_and_merge_data()
    
    if data is not None:
        # Display imported data
        st.subheader('Imported and Merged Data')
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

if __name__ == '__main__':
    main()
