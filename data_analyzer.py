import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
import os


class DataAnalyzer:
    """
    A utility class for data analysis operations including loading, cleaning,
    and visualizing data from various sources.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the DataAnalyzer with an optional data path."""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.
        
        Args:
            filename: Name of the CSV file to load
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing the loaded data
        """
        if self.data_path:
            full_path = os.path.join(self.data_path, filename)
        else:
            full_path = filename
            
        try:
            self.data = pd.read_csv(full_path, **kwargs)
            return self.data
        except FileNotFoundError:
            print(f"Error: File {full_path} not found.")
            return pd.DataFrame()
    
    def clean_data(self, columns_to_drop: List[str] = None, 
                  fill_na: Dict[str, Union[str, int, float]] = None) -> pd.DataFrame:
        """
        Clean the loaded data by removing specified columns and handling missing values.
        
        Args:
            columns_to_drop: List of column names to remove
            fill_na: Dictionary mapping column names to values for filling NAs
            
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        # Create a copy to avoid modifying the original
        cleaned_data = self.data.copy()
        
        # Drop specified columns
        if columns_to_drop:
            cleaned_data = cleaned_data.drop(columns=columns_to_drop, errors='ignore')
        
        # Fill NA values in specified columns
        if fill_na:
            for col, value in fill_na.items():
                if col in cleaned_data.columns:
                    cleaned_data[col] = cleaned_data[col].fillna(value)
        
        self.processed_data = cleaned_data
        return cleaned_data
    
    def calculate_statistics(self, numeric_only: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for each column in the processed data.
        
        Args:
            numeric_only: If True, only calculate stats for numeric columns
            
        Returns:
            Dictionary with column names as keys and stats as values
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        result = {}
        data = self.processed_data.select_dtypes(include=['number']) if numeric_only else self.processed_data
        
        for column in data.columns:
            col_data = data[column]
            result[column] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
        
        return result
    
    def plot_histogram(self, column: str, bins: int = 10, 
                       title: Optional[str] = None, 
                       figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot a histogram for the specified column.
        
        Args:
            column: The column name to plot
            bins: Number of bins in the histogram
            title: Optional title for the plot
            figsize: Figure size as a tuple (width, height)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        if column not in self.processed_data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        plt.figure(figsize=figsize)
        plt.hist(self.processed_data[column].dropna(), bins=bins, alpha=0.7)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(title or f'Histogram of {column}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def find_correlations(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Find correlations between numeric columns above the specified threshold.
        
        Args:
            threshold: Minimum absolute correlation value to include
            
        Returns:
            DataFrame with pairs of columns and their correlation values
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data() first.")
        
        numeric_data = self.processed_data.select_dtypes(include=['number'])
        corr_matrix = numeric_data.corr()
        
        # Create pairs of columns with their correlation values
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    corr_pairs.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': correlation
                    })
        
        return pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)


# Example usage
if __name__ == "__main__":
    analyzer = DataAnalyzer("data/")
    df = analyzer.load_csv("sample_data.csv")
    
    cleaned_df = analyzer.clean_data(
        columns_to_drop=["id", "unused_column"],
        fill_na={"age": 30, "income": 0}
    )
    
    stats = analyzer.calculate_statistics()
    print("Data Statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
    
    # Find strong correlations
    correlations = analyzer.find_correlations(threshold=0.7)
    print("\nStrong correlations:")
    print(correlations)
    
    # Plot histogram for a numeric column
    if "age" in cleaned_df.columns:
        analyzer.plot_histogram("age", bins=15, title="Age Distribution")
