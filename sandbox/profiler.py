import time
import psutil
import tracemalloc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple
import itertools

# Import the main preprocessing function
from your_preprocessing_script import preprocess_corpus

class ProcessingProfiler:
    def __init__(self, output_dir: str = "profiling_results"):
        """
        Initialize the profiler.
        
        Args:
            output_dir: Directory to store profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.param_combinations = []
        
    def _measure_memory(self, pid: int = None) -> float:
        """Measure memory usage in MB"""
        if pid is None:
            pid = os.getpid()
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def profile_single_run(self, **kwargs) -> Dict[str, Any]:
        """
        Profile a single run of the preprocessing function with given parameters.
        
        Args:
            **kwargs: Parameters to pass to the preprocessing function
            
        Returns:
            Dictionary with profiling results
        """
        print(f"\nProfiling with parameters: {kwargs}")
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self._measure_memory()
        start_time = time.time()
        
        # Run the preprocessing function
        preprocess_corpus(**kwargs)
        
        # Measure execution time
        execution_time = time.time() - start_time
        
        # Measure memory usage
        peak_memory = self._measure_memory() - initial_memory
        _, peak_traced = tracemalloc.get_traced_memory()
        peak_traced_mb = peak_traced / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        
        result = {
            "params": kwargs,
            "execution_time_seconds": execution_time,
            "peak_memory_mb": peak_memory,
            "peak_traced_memory_mb": peak_traced_mb
        }
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        
        return result
    
    def profile_parameter_grid(self, 
                          input_dir: str, 
                          output_dir: str,
                          model_names: List[str] = ["en_core_web_sm"],
                          batch_sizes: List[int] = [50, 100, 200],
                          n_processes_list: List[int] = [1, 2, 4, 8],
                          disable_options: List[List[str]] = [None, ["ner"], ["parser"], ["ner", "parser"]]) -> None:
        """
        Profile preprocessing with different parameter combinations.
        
        Args:
            input_dir: Input directory with text files
            output_dir: Output directory for processed files
            model_names: List of spaCy models to test
            batch_sizes: List of batch sizes to test
            n_processes_list: List of process counts to test
            disable_options: List of pipeline components to disable
        """
        # Generate all combinations of parameters
        base_params = {"input_dir": input_dir, "output_dir": output_dir}
        
        # Clean up disable_options for better representation
        clean_disable_options = []
        for option in disable_options:
            if option is None:
                clean_disable_options.append([])
            else:
                clean_disable_options.append(option)
        
        # Generate all parameter combinations
        all_combinations = list(itertools.product(
            model_names, 
            batch_sizes, 
            n_processes_list, 
            clean_disable_options
        ))
        
        self.param_combinations = all_combinations
        total_runs = len(all_combinations)
        
        print(f"Running profiling with {total_runs} parameter combinations")
        
        for i, (model, batch_size, n_process, disable) in enumerate(all_combinations):
            print(f"\nRun {i+1}/{total_runs}")
            
            # Prepare temp output directory for this run
            temp_output_dir = Path(output_dir) / f"temp_run_{i}"
            temp_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Set parameters for this run
            params = {
                **base_params,
                "model_name": model,
                "batch_size": batch_size,
                "n_process": n_process,
                "disable": disable,
                "output_dir": str(temp_output_dir)
            }
            
            # Run profiling
            result = self.profile_single_run(**params)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
    
    def save_results(self) -> None:
        """Save profiling results to JSON"""
        results_file = self.output_dir / "profiling_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self) -> None:
        """Load profiling results from JSON"""
        results_file = self.output_dir / "profiling_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                self.results = json.load(f)
    
    def _create_result_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for easier analysis"""
        data = []
        
        for result in self.results:
            params = result["params"]
            row = {
                "model": params["model_name"],
                "batch_size": params["batch_size"],
                "n_process": params["n_process"],
                "disable": ",".join(params["disable"]) if params["disable"] else "none",
                "execution_time": result["execution_time_seconds"],
                "peak_memory": result["peak_memory_mb"],
                "peak_traced_memory": result.get("peak_traced_memory_mb", 0)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_results(self) -> None:
        """Generate plots for profiling results"""
        if not self.results:
            print("No results to plot")
            return
        
        df = self._create_result_dataframe()
        
        # Create figure directory
        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        
        # Plot 1: Execution time vs batch size for different process counts
        self._plot_param_comparison(
            df, 
            x_param="batch_size", 
            y_metric="execution_time", 
            group_param="n_process",
            title="Execution Time vs Batch Size",
            ylabel="Execution Time (seconds)",
            filename="execution_time_vs_batch_size.png"
        )
        
        # Plot 2: Memory usage vs batch size for different process counts
        self._plot_param_comparison(
            df, 
            x_param="batch_size", 
            y_metric="peak_memory", 
            group_param="n_process",
            title="Memory Usage vs Batch Size",
            ylabel="Peak Memory (MB)",
            filename="memory_vs_batch_size.png"
        )
        
        # Plot 3: Execution time vs process count for different batch sizes
        self._plot_param_comparison(
            df, 
            x_param="n_process", 
            y_metric="execution_time", 
            group_param="batch_size",
            title="Execution Time vs Process Count",
            ylabel="Execution Time (seconds)",
            filename="execution_time_vs_processes.png"
        )
        
        # Plot 4: Memory usage vs process count for different batch sizes
        self._plot_param_comparison(
            df, 
            x_param="n_process", 
            y_metric="peak_memory", 
            group_param="batch_size",
            title="Memory Usage vs Process Count",
            ylabel="Peak Memory (MB)",
            filename="memory_vs_processes.png"
        )
        
        # Plot 5: Execution time for different disabled components
        self._plot_param_comparison(
            df, 
            x_param="disable", 
            y_metric="execution_time", 
            group_param="model",
            title="Execution Time vs Disabled Components",
            ylabel="Execution Time (seconds)",
            filename="execution_time_vs_disabled.png",
            use_bar=True
        )
        
        # Plot 6: Memory for different disabled components
        self._plot_param_comparison(
            df, 
            x_param="disable", 
            y_metric="peak_memory", 
            group_param="model",
            title="Memory Usage vs Disabled Components",
            ylabel="Peak Memory (MB)",
            filename="memory_vs_disabled.png",
            use_bar=True
        )
        
        # Plot 7: Heatmap of execution time for batch size vs process count
        self._plot_heatmap(
            df,
            x_param="batch_size",
            y_param="n_process",
            z_metric="execution_time",
            title="Execution Time Heatmap",
            filename="execution_time_heatmap.png"
        )
        
        # Plot 8: Heatmap of memory usage for batch size vs process count
        self._plot_heatmap(
            df,
            x_param="batch_size",
            y_param="n_process",
            z_metric="peak_memory",
            title="Memory Usage Heatmap",
            filename="memory_heatmap.png"
        )
        
        print(f"Plots saved to {figure_dir}")
    
    def _plot_param_comparison(
        self, 
        df: pd.DataFrame, 
        x_param: str, 
        y_metric: str, 
        group_param: str,
        title: str,
        ylabel: str,
        filename: str,
        use_bar: bool = False
    ) -> None:
        """Plot comparison of a parameter's effect on a metric"""
        plt.figure(figsize=(10, 6))
        
        if use_bar:
            # Group by the parameters
            grouped = df.groupby([x_param, group_param])[y_metric].mean().reset_index()
            
            # Get unique values for parameters 
            x_values = sorted(df[x_param].unique())
            group_values = sorted(df[group_param].unique())
            
            # Set width and positions for bars
            bar_width = 0.8 / len(group_values)
            
            # Plot bars for each group
            for i, group in enumerate(group_values):
                group_data = grouped[grouped[group_param] == group]
                positions = np.arange(len(x_values)) + i * bar_width - (len(group_values) - 1) * bar_width / 2
                plt.bar(positions, group_data[y_metric], width=bar_width, label=f"{group_param}={group}")
            
            plt.xticks(np.arange(len(x_values)), x_values)
        else:
            # For each group value, plot a line
            for group_value in sorted(df[group_param].unique()):
                group_df = df[df[group_param] == group_value]
                
                # Group by x_param and calculate mean of the metric
                plot_data = group_df.groupby(x_param)[y_metric].mean()
                
                plt.plot(plot_data.index, plot_data.values, 'o-', label=f"{group_param}={group_value}")
        
        plt.title(title)
        plt.xlabel(x_param)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "figures" / filename)
        plt.close()
    
    def _plot_heatmap(
        self, 
        df: pd.DataFrame, 
        x_param: str, 
        y_param: str, 
        z_metric: str,
        title: str,
        filename: str
    ) -> None:
        """Plot heatmap of two parameters' effect on a metric"""
        # Create pivot table
        pivot = df.pivot_table(
            index=y_param, 
            columns=x_param, 
            values=z_metric,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(pivot, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, label=z_metric)
        
        # Set ticks and labels
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                plt.text(j, i, f"{value:.1f}", ha='center', va='center', 
                         color='white' if value > pivot.values.mean() else 'black')
        
        plt.title(title)
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "figures" / filename)
        plt.close()
    
    def generate_report(self) -> None:
        """Generate a summary report of the profiling results"""
        if not self.results:
            print("No results to report")
            return
            
        df = self._create_result_dataframe()
        
        # Find optimal configurations
        fastest_config = df.loc[df['execution_time'].idxmin()]
        lowest_memory_config = df.loc[df['peak_memory'].idxmin()]
        
        # Calculate efficiency (time * memory)
        df['efficiency'] = df['execution_time'] * df['peak_memory']
        most_efficient_config = df.loc[df['efficiency'].idxmin()]
        
        # Generate report
        report = f"""# Text Preprocessing Profiling Report

## Summary of Results
- **Total configurations tested:** {len(df)}
- **Models tested:** {', '.join(df['model'].unique())}
- **Batch sizes tested:** {', '.join(map(str, sorted(df['batch_size'].unique())))}
- **Process counts tested:** {', '.join(map(str, sorted(df['n_process'].unique())))}
- **Pipeline components disabled:** {', '.join(df['disable'].unique())}

## Optimal Configurations

### Fastest Configuration
- **Model:** {fastest_config['model']}
- **Batch size:** {fastest_config['batch_size']}
- **Process count:** {fastest_config['n_process']}
- **Disabled components:** {fastest_config['disable']}
- **Execution time:** {fastest_config['execution_time']:.2f} seconds
- **Memory usage:** {fastest_config['peak_memory']:.2f} MB

### Lowest Memory Usage Configuration
- **Model:** {lowest_memory_config['model']}
- **Batch size:** {lowest_memory_config['batch_size']}
- **Process count:** {lowest_memory_config['n_process'