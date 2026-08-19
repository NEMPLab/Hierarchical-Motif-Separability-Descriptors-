#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from define_structural_primitives import (
    get_crystalnn_motifs,
    identify_asymmetric_unit
)
from pymatgen.core import Structure


def compute_CE_feature(center_atom, neighbors, elements_properties):
    """
    计算局部基元的中心-环境（CE）特征向量。

    Parameters
    ----------
    center_atom : dict
        {'species': str, 'coords': array_like, 'index': int}
    neighbors : list of dict
        [{'species': str, 'coords': array_like, 'distance': float}, ...]
    elements_properties : pd.DataFrame
        元素性质表，行列都是元素符号

    Returns
    -------
    CE_feature : pd.Series
        带有列名的 CE 特征向量
    """
    # 中心原子属性加前缀 C_
    center_species = center_atom['species']
    center_prop = elements_properties[center_species]
    center_indexed = pd.Series(center_prop.values, index=[f"C_{attr}" for attr in center_prop.index])

    # 邻居属性按距离加权
    weights = np.array([1 / n['distance'] if n['distance'] != 0 else 0 for n in neighbors])
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.zeros_like(weights)

    env_prop = pd.Series(0, index=center_prop.index)
    for n, w in zip(neighbors, weights):
        env_prop += elements_properties[n['species']] * w

    # 邻居属性加前缀 E_
    env_indexed = pd.Series(env_prop.values, index=[f"E_{attr}" for attr in env_prop.index])

    # 拼接中心和邻居特征
    CE_feature = pd.concat([center_indexed, env_indexed])
    return CE_feature


def load_elements_properties(file_path="ElementsProperties.xlsx"):
    """
    读取元素属性表格

    Returns
    -------
    elements_properties : pd.DataFrame
        行列都是元素符号
    property_names : list
        属性名称列表，用于列重命名
    """
    elements_df = pd.read_excel(file_path, sheet_name=0, index_col=None, header=0)
    elements_list = pd.read_excel(file_path, sheet_name=1, index_col=None, header=0)
    elements = list(elements_list["element"])
    df = elements_df.loc[:, elements]
    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True)
    # 从 Properties 列获取属性名称
    property_names = elements_df["Properties"].tolist()
    return df, property_names


def extract_CE_features_from_motifs(motifs, elements_properties_file="ElementsProperties.xlsx"):
    """
    从给定结构 motif 列表计算 CE 特征

    Parameters
    ----------
    motifs : list of dict
        每个元素包含:
        {
            'central_atom': dict,
            'neighbors': list of dict,
            'coordination_environment': str,
            'distorted': bool
        }
    elements_properties_file : str
        元素属性文件路径

    Returns
    -------
    features : list of dict
        每个元素包含:
        {
            'center_index': int,
            'center_species': str,
            'coordination_environment': str,
            'distorted': bool,
            'CE_feature': pd.Series
        }
    property_names : list of str
        属性名称列表，用于后续重命名
    """
    elements_properties, property_names = load_elements_properties(elements_properties_file)
    features = []

    for motif in motifs:
        center_atom = motif['central_atom']
        CE_vec = compute_CE_feature(center_atom=center_atom,
                                    neighbors=motif['neighbors'],
                                    elements_properties=elements_properties)
        features.append({
            'center_index': center_atom['index'],
            'center_species': center_atom['species'],
            'coordination_environment': motif.get('coordination_environment', ''),
            'distorted': motif.get('distorted', False),
            'CE_feature': CE_vec
        })
    return features, property_names


def rename_CE_feature_columns(ce_df, property_names):
    """
    将 CE 特征列重命名为可读的名称 C_XXX / E_XXX

    Parameters
    ----------
    ce_df : pd.DataFrame
        CE 特征矩阵
    property_names : list of str
        每个元素对应一个属性名

    Returns
    -------
    ce_df_renamed : pd.DataFrame
        列名已替换为 C_XXX / E_XXX
    """
    n_props = len(property_names)
    C_cols = ["C_" + name for name in property_names]
    E_cols = ["E_" + name for name in property_names]

    if ce_df.shape[1] != 2 * n_props:
        raise ValueError(f"列数({ce_df.shape[1]})与属性数({n_props})不匹配")

    ce_df_renamed = ce_df.copy()
    ce_df_renamed.columns = C_cols + E_cols
    return ce_df_renamed

def weighted_CE_vector(ce_features, method='hybrid', scale_method='zscore'):
    """
    对多个基元的 CE 特征进行加权，生成一个最终结构级 CE 向量

    Parameters
    ----------
    ce_features : list of dict
        每个元素包含：
        {
            'center_index': int,
            'CE_feature': pd.Series
        }
    method : str
        加权方法：'variance', 'distance', 'hybrid'
    scale_method : str
        标准化方法：'zscore' 或 'minmax'

    Returns
    -------
    final_vector : pd.Series
        加权后的结构 CE 向量，带列名
    """

    # === Step 1: 构建特征矩阵 ===
    # 每行一个基元
    all_keys = ce_features[0]['CE_feature'].index
    X = np.array([f['CE_feature'].values for f in ce_features])

    n_motifs = len(ce_features)
    if n_motifs == 1:
        return ce_features[0]['CE_feature']

    # === Step 2: 特征标准化 ===
    if scale_method == 'zscore':
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale_method 必须是 'zscore' 或 'minmax'")

    X_scaled = scaler.fit_transform(X)

    # === Step 3: 计算方差权重 ===
    var_weights = np.var(X_scaled, axis=1)
    if np.all(var_weights == 0):
        var_weights = np.ones_like(var_weights)

    # === Step 4: 计算距离权重 ===
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    dist_weights = np.mean(dist_matrix, axis=1)

    # === Step 5: 组合权重 ===
    if method == 'variance':
        weights = var_weights
    elif method == 'distance':
        weights = dist_weights
    elif method == 'hybrid':
        weights = (var_weights + dist_weights) / 2
    else:
        raise ValueError("method 必须是 'variance', 'distance' 或 'hybrid'")

    # === Step 6: 非负化 + 归一化 ===
    weights = np.maximum(weights, 1e-8)
    weights /= np.sum(weights)

    # === Step 7: 加权求和 ===
    weighted_X = X * weights[:, np.newaxis]  # 每行乘对应权重
    final_vector_values = np.sum(weighted_X, axis=0)

    # 返回 pd.Series，保持列名
    final_vector = pd.Series(final_vector_values, index=all_keys)
    return final_vector