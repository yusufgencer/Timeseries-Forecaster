import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize an instance of DataAnalyzer.
        
        Parameters:
        - dataframe (DataFrame): The DataFrame to analyze.
        """
        self.data = dataframe

    def summary_report(self):
        """
        Generate an enhanced summary report for the DataFrame.
        
        Returns:
        - summary (DataFrame): Enhanced summary report.
        """
        summary = self.data.describe().transpose()
        summary['mode'] = self.data.mode().iloc[0]
        summary['missing_values'] = self.data.isnull().sum()
        return summary
       
    def plot_correlation_matrix(self, selected_features):
        """
        Plot a correlation matrix heatmap for selected features.
        
        Parameters:
        - selected_features (list): List of feature names to include in the correlation matrix.
        """
        # Calculate correlation matrix for selected features
        corr_matrix = self.data[selected_features].corr()

        # Create a matplotlib figure for the heatmap
        plt.figure(figsize=(10, 8))

        # Plotting the heatmap using seaborn with the "Reds" colormap
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
                    cbar_kws={"shrink": .82}, linewidths=.5, 
                    annot_kws={"size": 10})

        plt.title('Correlation Matrix Heatmap', size=18)
        
        # Use Streamlit to display the matplotlib figure
        st.pyplot(plt)

    def plot_data_distributions(self, selected_features):
        """
        Plot data distributions for each selected numeric feature.

        Parameters:
        - selected_features (list): List of feature names to plot their distributions.
        """
        for feature in selected_features:
            # Check if the feature is numeric
            if self.data[feature].dtype in ['int64', 'float64']:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[feature], kde=True, bins=30, color='red')
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                st.pyplot(plt)
            else:
                st.warning(f"{feature} is not numeric and cannot be plotted.")


    def plot_outliers(self, selected_features):
        """
        Plot outliers for each selected numeric feature using box plots.

        Parameters:
        - selected_features (list): List of feature names to plot their outliers.
        """
        for feature in selected_features:
            # Check if the feature is numeric
            if self.data[feature].dtype in ['int64', 'float64']:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.data[feature], color='salmon')
                plt.title(f'Outliers in {feature}')
                plt.xlabel(feature)
                st.pyplot(plt)
            else:
                st.warning(f"{feature} is not numeric and cannot be plotted for outliers.")


    def outlier_statistics(self, selected_features):
        """
        Calculate statistics for outliers in selected numeric features.

        Parameters:
        - selected_features (list): List of feature names to analyze for outliers.

        Returns:
        - outlier_stats (dict): Dictionary containing statistics for outliers in each selected feature.
          The keys are feature names, and the values are dictionaries containing the following statistics:
            - 'count': Number of outliers.
            - 'mean': Mean value of outliers.
            - 'std': Standard deviation of outliers.
        """
        outlier_stats = {}
        for feature in selected_features:
            if self.data[feature].dtype in ['int64', 'float64']:
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[feature] < (Q1 - 1.5 * IQR)) | (self.data[feature] > (Q3 + 1.5 * IQR))]
                
                # Calculate statistics for outliers
                if not outliers.empty:
                    outlier_stats[feature] = {
                        'count': outliers[feature].count(),
                        'mean': outliers[feature].mean(),
                        'std': outliers[feature].std()
                    }
                else:
                    outlier_stats[feature] = 'No outliers detected'
            else:
                outlier_stats[feature] = 'Feature is non-numeric and cannot be analyzed for outliers'
        
        return outlier_stats

    
    def feature_target_correlation(self, target_column, method='pearson', encode_categorical=True):
        """
        Calculate the correlation between each feature and the target variable.

        Parameters:
        - target_column: The name of the target column.
        - method: Method of correlation (default 'pearson'). Options include 'pearson', 'spearman', 'kendall'.
        - encode_categorical: Boolean indicating whether to encode categorical variables before correlation analysis.
        """
        # Ensure target_column exists
        if target_column not in self.data.columns:
            st.error(f"Target column '{target_column}' not found.")
            return

        # Prepare data
        analysis_data = self.data.copy()
        
        if encode_categorical:
            # Encode categorical variables if required
            categorical_cols = analysis_data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                analysis_data[col] = analysis_data[col].astype('category').cat.codes
        
        # Calculate correlation
        correlation_results = analysis_data.corr(method=method)[target_column].drop(target_column)
        correlation_results = correlation_results.sort_values(ascending=False)

        # Display results
        return correlation_results
    
    def pca_visualization(self, selected_features=None):
        """
        Performs PCA on selected features or the entire dataset and visualizes the first two principal components.
        
        Parameters:
        - selected_features: List of columns to include in the PCA. If None, uses all numeric features.
        """
        # If specific features are not selected, use all numeric features
        if selected_features is None:
            selected_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract the selected features
        data_subset = self.data[selected_features]
        
        # Standardizing the features before applying PCA
        data_standardized = StandardScaler().fit_transform(data_subset)
        
        # Apply PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data_standardized)
        
        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
        
        # Visualize the first two principal components
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df)
        plt.title('PCA of Selected Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Display the plot in Streamlit
        st.pyplot(plt)


def correlation_matrix(analyzer):
    numeric_columns = analyzer.data.select_dtypes(include=np.number).columns.tolist()

    # Checkbox for selecting all numeric columns to automatically generate the heatmap
    select_all = st.checkbox("Select All Numeric Columns for Correlation Matrix", key="select_all_correlation")

    if select_all:
        # If "Select All" is checked, immediately plot the correlation matrix with all numeric columns
        analyzer.plot_correlation_matrix(numeric_columns)
    else:
        # If "Select All" is not checked, allow manual selection of columns for the correlation matrix
        selected_columns = st.multiselect('Select columns for correlation', numeric_columns, key="multiselect_correlation")
        if st.button('Generate Correlation Matrix', key="generate_correlation"):
            if selected_columns:
                analyzer.plot_correlation_matrix(selected_columns)
            else:
                st.warning("Please select at least one column.")

def distribution(analyzer):
    numeric_columns = analyzer.data.select_dtypes(include=np.number).columns.tolist()
    select_all = st.checkbox("Select All Numeric Columns to visualize their distribution", key='select_all_distribution')

    if select_all:
        analyzer.plot_data_distributions(numeric_columns)
    else:
        selected_columns = st.multiselect('Select numeric columns to visualize their distribution', numeric_columns, key='multiselect_distribution')
        if selected_columns:
            analyzer.plot_data_distributions(selected_columns)


def outlier_visualization(analyzer):
    numeric_columns = analyzer.data.select_dtypes(include=np.number).columns.tolist()
    select_all_outliers = st.checkbox("Select All Numeric Columns to visualize outliers")

    if select_all_outliers:
        # If "Select All" is checked, immediately plot outliers for all numeric columns
        analyzer.plot_outliers(numeric_columns)
    else:
        # If "Select All" is not checked, allow manual selection of columns for outliers visualization
        selected_columns_for_outliers = st.multiselect('Select numeric columns to visualize outliers', numeric_columns, default=numeric_columns[0])

        if selected_columns_for_outliers:
            analyzer.plot_outliers(selected_columns_for_outliers)

def outlier_statistics(analyzer):
    numeric_columns = analyzer.data.select_dtypes(include=np.number).columns.tolist()
    selected_columns_for_outliers_stats = st.multiselect('Select numeric columns to calculate outlier statistics', numeric_columns , default=numeric_columns[0])

    if selected_columns_for_outliers_stats:
        outlier_stats = analyzer.outlier_statistics(selected_columns_for_outliers_stats)
        for feature, stats in outlier_stats.items():
            if isinstance(stats, dict):
                st.markdown(f"**{feature}**")
                st.write(f"Count: {stats['count']}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            else:
                st.write(f"**{feature}**: {stats}")

def feature_target_correlations(analyzer):
    target_column = st.selectbox('Select the target variable for correlation analysis:', analyzer.data.columns)
    encode_categorical = st.checkbox("Encode categorical variables?", value=True)
    correlation_method = st.selectbox('Select correlation method:', ['pearson', 'spearman', 'kendall'], index=0)
    if st.button('Calculate Feature-Target Correlations'):
        correlation_results = analyzer.feature_target_correlation(target_column, method=correlation_method, encode_categorical=encode_categorical)
        if correlation_results is not None:
            st.write(correlation_results)

def pca_visualization(analyzer):
    numeric_columns = analyzer.data.select_dtypes(include=np.number).columns.tolist()
    selected_columns_for_pca = st.multiselect('Select features for PCA (leave empty to use all numeric features):', numeric_columns)
    if st.button('Perform PCA'):
        analyzer.pca_visualization(selected_columns_for_pca if selected_columns_for_pca else None)