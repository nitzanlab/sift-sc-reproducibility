"""create a mapping."""

import logging
import warnings

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from scipy.stats import pearsonr
from jax.config import config
config.update("jax_enable_x64", True)
        
warnings.filterwarnings("ignore")



class MappingProblem:
    """
    Initialize models and datasets to run mapping.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    def filter_vars(
        self,
        adata_sc,
        adata_spatial=None,
        var_names=None,
    ):

        vars_sc = set(
            adata_sc.var_names
        ) 
        vars_sp = set(adata_spatial.var_names) if adata_spatial is not None else None
        var_names = set(var_names) if var_names is not None else None
        if var_names is None and adata_spatial is not None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return adata_sc[:, list(var_names)], adata_spatial[:, list(var_names)]
            else:
                return adata_sc, adata_spatial
        elif var_names is None:
            return adata_sc
        elif adata_spatial is not None:
            if var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
                return adata_sc[:, list(var_names)], adata_spatial[:, list(var_names)]
            else:
                raise ValueError(
                    "Some `var_names` ares missing in either `adata_sc` or `adata_spatial`."
                )
        else:
            if var_names.issubset(vars_sc):
                return adata_sc[:, list(var_names)]
            else:
                raise ValueError("Some `var_names` ares missing in `adata_sc`.")

    def _correlate(
        self, adata_sc, adata_spatial, transport_matrix, var_names=None, rotate=False,
    ) -> pd.Series:
        """
        Calculate correlation between the predicted gene expression and observed in tissue space.

        Parameters
        ----------
        transport_matrix: learnt transport_matrix - assumes [n_cell_sp X n_cells_sc]
        var_names: genes to correlate, if none correlate all.

        Returns
        -------
        corr_val: the pearsonr correlation
        """
        try:
            transport_matrix = np.asarray(transport_matrix)
            adata_sc, adata_spatial = self.filter_vars(
                adata_sc, adata_spatial, var_names
            )
            X = adata_sc.X.copy() if (self.sc_joint_key == "X") else adata_sc.layers[self.sc_joint_key].copy()
            Y = adata_spatial.X.copy()
            
            if scipy.sparse.issparse(X):
                X = X.A
            if scipy.sparse.issparse(Y):
                Y = Y.A

            sp_gex_pred = np.dot(transport_matrix, X)
            if rotate:
                num_cells, num_genes = Y.shape
                pr = np.zeros((num_cells, num_genes))
                for c in range(num_cells):
                    for gi in range(num_genes):
                        pr[c, gi] = pearsonr(sp_gex_pred[:, gi], np.roll(Y[:, gi], -c))[0]
                idx_rot = np.argmax(np.nanmean(pr, axis=1))
                pears = pr[idx_rot, :]
            else:
                pears = [
                    pearsonr(sp_gex_pred[:, gi], Y[:, gi])[0]
                    for gi, _ in enumerate(adata_spatial.var_names)
                ]
            return pd.Series(pears, index=adata_spatial.var_names)
        except Exception as e:  # noqa: B902
            logging.error(f"Unable to correlate, reason: `{e}`")
            return pd.Series([])
        
    def _accuracy(
        self, adata_sc, transport_matrix, key="ZT", rotate=False,
    ) -> pd.Series:
        """
        Calculate the accuracy of the time-point prediction.

        Parameters
        ----------
        transport_matrix: learnt transport_matrix - assumes [n_cell_sp X n_cells_sc]
        key: labels key.

        Returns
        -------
        corr_val: the pearsonr correlation
        """
        try:
            transport_matrix = np.asarray(transport_matrix)
            tps_pred = transport_matrix.argmax(0)
            tps_orig = adata_sc.obs[key].astype(float).values / 6
            num_labels = transport_matrix.shape[0]
            
            error = np.abs(tps_pred - tps_orig).mean()
            if rotate:
                error_ = np.zeros(num_labels)
                for c in range(num_labels):
                    transport_matrix_cur = np.roll(transport_matrix, -c, axis=0)
                    tps_pred = transport_matrix.argmax(0)
                    error_[c] = np.abs(tps_pred - tps_orig).mean()
                error = np.min(error_)
            
            return (num_labels - error) / num_labels
        except Exception as e:  # noqa: B902
            logging.error(f"Unable to find accuracy, reason: `{e}`")
            return np.inf


    def init_dataset(self, dataset: dict):

        import scanpy as sc

        logging.info(f"Initializing dataset: `{dataset}`")

        dataset_logger = logging.getLogger()

        self.adata_spatial = sc.read(dataset["adata_spatial"])
        self.adata_sc = sc.read(dataset["adata_sc"])


        dataset_logger.info("subset data.")
        reps = dataset.get("rep", self.adata_sc.obs["rep"].cat.categories)
        self.adata_sc = self.adata_sc[self.adata_sc.obs["rep"].isin(reps),:]

        if "subsample" in dataset:
            sc.pp.subsample(self.adata_sc, dataset["subsample"])
        
        
        dataset_logger.info("set marker and reference genes.")
        marker_genes = self.adata_spatial.var_names.intersection(
            self.adata_sc.var_names
        ) if dataset.get("marker", None) is None else self.adata_spatial.var_names[self.adata_spatial.var["type"].isin(dataset["marker"])]
        
        marker_genes = marker_genes if dataset.get("marker_names", None) is None else dataset["marker_names"]
        
        # set marker genes
        self.adata_sc.var["marker"] = False
        self.adata_sc.var.loc[
            self.adata_sc.var_names.isin(marker_genes), "marker"
        ] = True

        self.adata_spatial.var["marker"] = False
        self.adata_spatial.var.loc[
            self.adata_spatial.var_names.isin(marker_genes), "marker"
        ] = True
        
        
        reference_genes = self.adata_spatial.var_names.intersection(
            self.adata_sc.var_names
        ) if dataset.get("reference", None) is None else self.adata_spatial.var_names[self.adata_spatial.var["type"].isin(dataset["reference"])]

        reference_genes = reference_genes if dataset.get("reference_names", None) is None else dataset["reference_names"]
        
        # set reference genes
        self.adata_sc.var["reference"] = False
        self.adata_sc.var.loc[
            self.adata_sc.var_names.isin(reference_genes), "reference"
        ] = True

        self.adata_spatial.var["reference"] = False
        self.adata_spatial.var.loc[
            self.adata_spatial.var_names.isin(reference_genes), "reference"
        ] = True
        
        self.sc_key = dataset.get("sc_key", "X_scvi")
        self.sc_joint_key = dataset.get("sc_joint_key", "X")
        self.sp_key = dataset.get("sp_key", "spatial")

        sc.pp.highly_variable_genes(self.adata_sc)
        sc.pp.pca(self.adata_sc, n_comps=30, use_highly_variable=True)

        dataset_logger.info("Init dataset finished.")

    def init_model(self, solver):
        logging.info(f"Initializing model: `{solver}`")
        self.fused_penalty = solver["fused_penalty"]
        self.epsilon = solver["epsilon"]
        self.rank = solver["rank"]
        self.gamma = solver.get("gamma", None)

    def init_all(self):
        """Sequentially run the sub-initializers of the experiment."""
        self.init_model()
        self.init_dataset()

    def train(self, training: dict):
        import pathlib
        import pickle as pkl
        from time import perf_counter

        from jax.config import config

        config.update("jax_enable_x64", True)

        import ott

        logging.info(f"Training: `{training}`")
        train_logger = logging.getLogger()


        test_results = {}
        save = training.get("save", False)
        path_dir = training.get("output", None)
        path_dir = pathlib.Path(path_dir) if path_dir is not None else path_dir

        var_names = self.adata_spatial.var_names[
            self.adata_spatial.var["marker"]
        ]
        rank = -1 if self.rank is None else self.rank
        if isinstance(rank, float):
            rank = int(rank * self.adata_spatial.n_obs)
        if rank == -1:
            kwargs = {}
        else:
            kwargs = {
                "rank": rank,
                "gamma": self.gamma,
            }
        logging.info(f"Linear OT kwargs: `{kwargs}`")

        sp, sc, joint = self._create_geoms(
            adata_sc=self.adata_sc,
            adata_sp=self.adata_spatial,
            epsilon=self.epsilon,
            sc_key=self.sc_key,
            sp_key=self.sp_key,
            sc_joint_key = self.sc_joint_key,
            var_names=var_names
        )

        start = perf_counter()
        if self.fused_penalty == 0:
            prob = ott.core.quad_problems.QuadraticProblem(
                geom_xx=sp, 
                geom_yy=sc, 
                scale_cost="max_cost",
            )
            gw = ott.core.gromov_wasserstein.GromovWasserstein(
                epsilon=self.epsilon
            )
            out = gw(prob)
        else:
            prob = ott.core.quad_problems.QuadraticProblem(
                geom_xx=sp, 
                geom_yy=sc, 
                geom_xy=joint,
                scale_cost="max_cost", 
                fused_penalty=self.fused_penalty,
            )
            gw = ott.core.gromov_wasserstein.GromovWasserstein(
                epsilon=self.epsilon
            )
            out = gw(prob)
            
        time_ = perf_counter() - start

        transport_matrix = np.asarray(out.matrix)
        converged = bool(out.converged)
        corrs = self._correlate(
            adata_sc=self.adata_sc, 
            adata_spatial=self.adata_spatial,
            transport_matrix=transport_matrix,
            var_names=self.adata_spatial.var_names[self.adata_spatial.var["reference"]],
            rotate=(self.fused_penalty == 0)
        )
        
        accuracy = self._accuracy(
            adata_sc=self.adata_sc, 
            transport_matrix=transport_matrix,
            key="ZT",
            rotate=(self.fused_penalty == 0)
        )

        if save:
            with open(path_dir / f"res_{self.epsilon}_{self.fused_penalty}_ott.pkl", "wb") as sol_path:
                pkl.dump(out, sol_path)

        test_results["pears_mean"] = np.nanmean(corrs)
        test_results["pears_std"] = np.nanstd(corrs)
        test_results["accuracy"] = accuracy
        test_results["time"] = time_
        test_results["converged"] = converged
        train_logger.info(test_results)

        return test_results, out

        
    def _create_geoms(self, adata_sc: AnnData,
                      adata_sp: AnnData,
                      epsilon: float,
                      sc_key: str = "X_scvi",
                      sp_key: str = "spatial",
                      sc_joint_key: str = "X",
                      **kwargs):
        
        from ott import geometry
        from ott.geometry import pointcloud

        adata_sc, adata_sp = self.filter_vars(adata_sc, adata_sp, **kwargs)
        sc = adata_sc.obsm[sc_key]
        sp = adata_sp.obsm[sp_key]
        
        if sp.ndim == 1:
            sp = sp[:, np.newaxis]
            
        X = adata_sp.X.copy()
        Y = adata_sc.X.copy() if (sc_joint_key=="X") else adata_sc.layers[sc_joint_key].copy()
        
        if scipy.sparse.issparse(X):
            X = X.A
        if scipy.sparse.issparse(Y):
            Y = Y.A
            
        geom_xx = pointcloud.PointCloud(
            x=sp, 
            scale_cost="max_cost", 
            epsilon=epsilon,)
        geom_yy = pointcloud.PointCloud(
            x=sc, 
            scale_cost="max_cost", 
            epsilon=epsilon,)
        geom_xy = pointcloud.PointCloud(
            x=X, 
            y=Y, 
            scale_cost="max_cost",
            epsilon=epsilon,)

        return (
            geom_xx,
            geom_yy,
            geom_xy,
        )


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.

def get_mapping(init_all=False):
    print("get_mapping")
    mapping = MappingProblem(init_all=init_all)
    return mapping
