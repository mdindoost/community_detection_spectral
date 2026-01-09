"""
Results management for experiment outputs.
Handles incremental folder creation and meta.json storage.
"""
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import RESULTS_DIR
else:
    from ..config import RESULTS_DIR


def get_next_run_folder(base_dir: Optional[Path] = None) -> Path:
    """
    Get the next incremental run folder.
    
    Folders are named 1, 2, 3, etc. This finds the highest existing
    folder number and returns the next one.
    
    Parameters
    ----------
    base_dir : Path, optional
        Base results directory. Defaults to RESULTS_DIR from config.
        
    Returns
    -------
    run_folder : Path
        Path to the new run folder (created)
    """
    if base_dir is None:
        base_dir = RESULTS_DIR
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find highest existing folder number
    existing_folders = [
        int(f.name) for f in base_dir.iterdir() 
        if f.is_dir() and f.name.isdigit()
    ]
    
    next_idx = max(existing_folders, default=0) + 1
    
    run_folder = base_dir / str(next_idx)
    run_folder.mkdir(parents=True, exist_ok=True)
    
    return run_folder


class ResultsManager:
    """
    Manages experiment results storage.
    
    Creates incremental run folders and stores:
    - meta.json with runtime info and metrics
    - Community membership files after each Leiden run
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize results manager and create run folder.
        
        Parameters
        ----------
        base_dir : Path, optional
            Base results directory
        """
        self.run_folder = get_next_run_folder(base_dir)
        self.meta: Dict[str, Any] = {
            'run_id': int(self.run_folder.name),
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
        self._save_meta()
    
    def _save_meta(self):
        """Save meta.json to run folder."""
        meta_path = self.run_folder / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self.meta, f, indent=2)
    
    def set_dataset_info(
        self,
        dataset_name: str,
        n_nodes: int,
        n_edges: int,
        n_components: int
    ):
        """
        Store dataset information in meta.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        n_nodes : int
            Number of nodes
        n_edges : int
            Number of edges
        n_components : int
            Number of connected components
        """
        self.meta['dataset'] = {
            'name': dataset_name,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_components': n_components
        }
        self._save_meta()
    
    def save_communities(
        self,
        membership: List[int],
        name: str
    ) -> Path:
        """
        Save community membership to file.
        
        Parameters
        ----------
        membership : list
            Community membership for each node
        name : str
            Name for the file (e.g., "original", "dspar_paper_0.75")
            
        Returns
        -------
        filepath : Path
            Path to saved file
        """
        filename = f"communities_{name}.txt"
        filepath = self.run_folder / filename
        
        # Save using numpy for speed
        import numpy as np
        mem_arr = np.array(membership, dtype=np.int64)
        node_ids = np.arange(len(mem_arr), dtype=np.int64)
        data = np.column_stack([node_ids, mem_arr])
        np.savetxt(filepath, data, fmt='%d', delimiter='\t')
        
        return filepath
    
    def add_original_result(
        self,
        membership: List[int],
        modularity: float,
        n_communities: int,
        leiden_time: float,
        total_intra: int,
        total_inter: int
    ):
        """
        Add original graph Leiden results.
        
        Parameters
        ----------
        membership : list
            Community membership
        modularity : float
            Modularity score
        n_communities : int
            Number of communities
        leiden_time : float
            Leiden runtime in seconds
        total_intra : int
            Number of intra-community edges
        total_inter : int
            Number of inter-community edges
        """
        # Save communities
        self.save_communities(membership, "original")
        
        self.meta['original'] = {
            'n_communities': n_communities,
            'modularity': modularity,
            'leiden_time': leiden_time,
            'total_intra_edges': total_intra,
            'total_inter_edges': total_inter
        }
        self._save_meta()
    
    def add_experiment_result(
        self,
        method: str,
        param: str,
        n_edges_sparse: int,
        edge_pct: float,
        n_components: int,
        n_communities: int,
        modularity: float,
        nmi: float,
        ari: float,
        intra_pct: float,
        inter_pct: float,
        ratio: float,
        sparsify_time: float,
        leiden_time: float,
        membership: List[int]
    ):
        """
        Add sparsification experiment result.
        
        Parameters
        ----------
        method : str
            Sparsification method name
        param : str
            Parameter string (e.g., "ε=1.0" or "r=0.75")
        n_edges_sparse : int
            Number of edges after sparsification
        edge_pct : float
            Percentage of edges retained
        n_components : int
            Number of connected components
        n_communities : int
            Number of communities found
        modularity : float
            Modularity score
        nmi : float
            NMI vs original
        ari : float
            ARI vs original
        intra_pct : float
            Intra-community edge preservation percentage
        inter_pct : float
            Inter-community edge preservation percentage
        ratio : float
            Inter/Intra preservation ratio
        sparsify_time : float
            Sparsification runtime in seconds
        leiden_time : float
            Leiden runtime in seconds
        membership : list
            Community membership
        """
        # Create safe filename
        safe_name = f"{method}_{param}".replace("=", "_").replace(".", "_").replace(" ", "_")
        self.save_communities(membership, safe_name)
        
        result = {
            'method': method,
            'param': param,
            'n_edges': n_edges_sparse,
            'edge_pct': edge_pct,
            'n_components': n_components,
            'n_communities': n_communities,
            'modularity': modularity,
            'nmi': nmi,
            'ari': ari,
            'intra_pct': intra_pct,
            'inter_pct': inter_pct,
            'ratio': ratio,
            'sparsify_time': sparsify_time,
            'leiden_time': leiden_time,
            'communities_file': f"communities_{safe_name}.txt"
        }
        
        self.meta['experiments'].append(result)
        self._save_meta()
    
    def add_cpm_results(self, cpm_results: List[Dict[str, Any]]):
        """
        Add CPM resolution analysis results.
        
        Parameters
        ----------
        cpm_results : list
            List of CPM analysis results
        """
        self.meta['cpm_analysis'] = cpm_results
        self._save_meta()
    
    def finalize(self):
        """Finalize results and save end time."""
        self.meta['end_time'] = datetime.now().isoformat()
        self._save_meta()
        return self.run_folder
