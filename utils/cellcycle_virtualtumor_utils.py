import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl

import matplotlib.pyplot as plt

def read_matrix_from_binary(file_name):
    """
    Read a matrix from a binary file defined by this module.
    Parameters
    ----------
    file_name
        the file to read
    Returns
    -------
    matrix
        the matrix form
    Notes
    -----
    This function is adapted from `Cyclum`
    """
    with open(file_name, "rb") as file:
        buffer = file.read()
    n_row = int.from_bytes(buffer[0:4], "little")
    n_col = int.from_bytes(buffer[4:8], "little")
    matrix = np.frombuffer(buffer[8:], dtype=float).reshape([n_row, n_col])
    return matrix

def read_df_from_binary(file_name_mask):
    """
    Read a data frame from a binary file defined by this module.
    Parameters
    ----------
    file_name_mask
        the file to read
    Returns
    -------
    the data frame
    Notes
    -----
    This function is adapted from `Cyclum`
    """
    data = read_matrix_from_binary(file_name_mask + "-value.bin")
    with open(file_name_mask + "-name.txt", "r") as f:
        index = f.readline().strip().split("\t")
        columns = f.readline().strip().split("\t")
    return pd.DataFrame(data=data, index=index, columns=columns)


def preprocess(input_file_mask):
    """
    Read in data and perform log transform (log2(x+1)), centering (mean = 1) and scaling (sd = 1).
    """
    tpm = read_df_from_binary(input_file_mask)
    
    sttpm = pd.DataFrame(data=skl.preprocessing.scale(np.log2(tpm.values + 1)), index=tpm.index, columns=tpm.columns)
    
    label = pd.read_csv(input_file_mask + "-label.csv", sep="\t", index_col=0)
    return sttpm, label



def plot_kernel(
    kernel,
    adata = None,
    src_key = None,
    tgt_key= None,
    groupby = None,
    save_path = None,
    show = True,
    fontsize=24,
    **kwargs,
):
    """
    Visualize the cell-cell similarity kernel.
    Parameters
    ----------
    kernel
        kernel to plot
    adata
         Optional, reference adata
    src_key
        Optional, mask for source space of kernel
    tgt_key
        Optional, mask for target space of kernel
    groupby
        the key of the observation grouping to consider.
    save_path
        path and fig_name to save the fig
    show
        If `False`, return :class:`matplotlib.pyplot.Axes`.
    **kwargs
        additional plotting arguments
    Returns
    -------
    The axes object, if ``show = False``.
    """
    
    interpolation = kwargs.pop("interpolation", True)
    if interpolation:
        from scipy import ndimage
        kernel = ndimage.gaussian_filter(kernel, sigma=1)
    vmax = kwargs.pop("vmax", np.quantile(kernel, 0.9))
    vmax = vmax if vmax > 0 else np.max(kernel)
    vmin = kwargs.pop("vmin", 0)
    color_palette = kwargs.pop("color_palette", "deep")
    cmap = kwargs.pop("cmap", "Blues")
    ncol = kwargs.pop("ncol", 2)
    figsize = kwargs.pop("figsize", None)
    if groupby is not None:
        src_idx = (
            _get_mask_idx(adata=adata, key=src_key)
            if src_key is not None
            else np.arange(adata.n_obs)
        )
        tgt_idx = (
            _get_mask_idx(adata=adata, key=tgt_key)
            if tgt_key is not None
            else np.arange(adata.n_obs)
        )
        if adata is None:
            raise ValueError("cannot use groupby without anndata reference.")
        if adata.obs[groupby].dtype.name == "category":
            labels = adata.obs[groupby].cat.categories.astype(str)
            df = pd.DataFrame(
                adata.obs[groupby].astype(str), index=adata.obs_names, columns=[groupby]
            )
        else:
            labels = adata.obs[groupby].unique()
            df = pd.DataFrame(
                adata.obs[groupby].astype(str), index=adata.obs_names, columns=[groupby]
            )

        label_pal = sns.color_palette(color_palette, labels.size)
        label_lut = dict(zip(map(str, labels), label_pal))
        row_colors = pd.Series(
            df[groupby].iloc[src_idx], index=adata.obs_names[src_idx], name=groupby
        ).map(label_lut)
        col_colors = pd.Series(
            df[groupby].iloc[tgt_idx], index=adata.obs_names[tgt_idx], name=groupby
        ).map(label_lut)

        g = sns.clustermap(
            kernel,
            row_cluster=False,
            col_cluster=False,
            standard_scale=None,
            col_colors=col_colors.to_numpy(),
            row_colors=row_colors.to_numpy(),
            linewidths=0,
            xticklabels=[],
            yticklabels=[],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            cbar_kws={"ticks":[vmin,vmax]},
            **kwargs,
        )
        if adata.obs[groupby].dtype.name == "category":
            for label in labels:
                g.ax_col_dendrogram.bar(
                    0, 0, color=label_lut[label], label=label, linewidth=0
                )

            l1 = g.ax_col_dendrogram.legend(
                title=groupby,
                loc="center",
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.9),
                bbox_transform=plt.gcf().transFigure,
            )
    else:
        g = sns.clustermap(
            kernel,
            row_cluster=False,
            col_cluster=False,
            linewidths=0,
            xticklabels=[],
            yticklabels=[],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            figsize=figsize,
        )

    g.cax.set_position([0.05, 0.2, 0.03, 0.45])
    ax = g.ax_heatmap
    ax.set_xlabel("cells", fontsize=fontsize)
    ax.set_ylabel("cells", fontsize=fontsize)

    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches="tight", transparent=True)

    if not show:
        return g.cax