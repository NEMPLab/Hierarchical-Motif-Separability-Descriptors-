from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element
import networkx as nx
from itertools import combinations
from scipy.spatial import distance, KDTree
from sklearn.preprocessing import StandardScaler

def calculate_bond_angles(neighbors):
    angles = []
    n_neighbors = len(neighbors)
    if n_neighbors < 2:
        return angles
    vectors = [neighbor['vector'] for neighbor in neighbors]
    for i in range(n_neighbors):
        for j in range(i + 1, n_neighbors):
            vec1 = vectors[i]
            vec2 = vectors[j]
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
    return angles

def calculate_shape_descriptors(neighbor_vectors, prefix):
    features = {}
    if len(neighbor_vectors) < 2:
        return features
    vectors = np.array(neighbor_vectors)
    # 惯性张量
    if len(vectors) > 0:
        inertia_tensor = np.zeros((3, 3))
        for vec in vectors:
            r_sq = np.dot(vec, vec)
            inertia_tensor += np.eye(3) * r_sq - np.outer(vec, vec)
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        eigenvalues = np.sort(eigenvalues)
        features[f'{prefix}_inertia_min'] = eigenvalues[0]
        features[f'{prefix}_inertia_mid'] = eigenvalues[1]
        features[f'{prefix}_inertia_max'] = eigenvalues[2]
        features[f'{prefix}_inertia_ratio'] = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0
    # 径向分布
    distances = [np.linalg.norm(vec) for vec in neighbor_vectors]
    if distances:
        features[f'{prefix}_radial_std'] = np.std(distances)
        features[f'{prefix}_radial_asymmetry'] = np.max(distances) / np.min(distances) if np.min(distances) > 0 else 0
    # 角度分布
    if len(neighbor_vectors) >= 3:
        angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dot_product = np.dot(vectors[i], vectors[j])
                norm_i = np.linalg.norm(vectors[i])
                norm_j = np.linalg.norm(vectors[j])
                if norm_i > 0 and norm_j > 0:
                    cos_angle = np.clip(dot_product / (norm_i * norm_j), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    angles.append(angle)
        if angles:
            features[f'{prefix}_angle_diversity'] = np.std(angles)
    return features

def calculate_topological_features(neighbor_vectors, prefix):
    features = {}
    if len(neighbor_vectors) < 3:
        return features
    vectors = np.array(neighbor_vectors)
    try:
        hull = ConvexHull(vectors)
        features[f'{prefix}_convex_volume'] = hull.volume
        features[f'{prefix}_convex_area'] = hull.area
    except:
        features[f'{prefix}_convex_volume'] = 0
        features[f'{prefix}_convex_area'] = 0
    if features.get(f'{prefix}_convex_volume', 0) > 0:
        features[f'{prefix}_sphericity'] = (36 * np.pi * features[f'{prefix}_convex_volume'] ** 2) ** (1 / 3) / \
                                           features[f'{prefix}_convex_area']
    else:
        features[f'{prefix}_sphericity'] = 0
    if len(vectors) >= 3:
        inertia_tensor = np.zeros((3, 3))
        for vec in vectors:
            r_sq = np.dot(vec, vec)
            inertia_tensor += np.eye(3) * r_sq - np.outer(vec, vec)
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        eigenvalues = np.sort(eigenvalues)
        if eigenvalues[2] > 0:
            anisotropy = 1 - 3 * eigenvalues[0] / np.sum(eigenvalues)
            features[f'{prefix}_anisotropy'] = anisotropy
    return features

def calculate_symmetry_features(neighbor_vectors, prefix):
    features = {}
    if len(neighbor_vectors) < 2:
        return features
    vectors = np.array(neighbor_vectors)
    distances = [np.linalg.norm(vec) for vec in vectors]
    if distances:
        unique_distances = len(set(np.round(distances, 3)))
        features[f'{prefix}_unique_distances'] = unique_distances
        features[f'{prefix}_distance_symmetry'] = len(distances) / unique_distances if unique_distances > 0 else 0
    angles = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[j])
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i > 0 and norm_j > 0:
                cos_angle = np.clip(dot_product / (norm_i * norm_j), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
    if angles:
        unique_angles = len(set(np.round(angles, 3)))
        features[f'{prefix}_unique_angles'] = unique_angles
        features[f'{prefix}_angle_symmetry'] = len(angles) / unique_angles if unique_angles > 0 else 0
    return features


from scipy.special import sph_harm
from scipy.spatial import ConvexHull

def radial_features(center, neighbor_coords):
    dists = np.linalg.norm(neighbor_coords - center, axis=1)
    return {
        #"dist_mean": float(np.mean(dists)),
      #  "dist_std": float(np.std(dists)),
       # "dist_min": float(np.min(dists)),
      #  "dist_max": float(np.max(dists)),
        "dist_skew": float(np.mean((dists - np.mean(dists))**3) / (np.std(dists)**3 + 1e-8)),
        "dist_unique": len(np.unique(np.round(dists, 3)))
    }

def angular_features(center, neighbor_coords):
    vecs = neighbor_coords - center
    n = len(vecs)
    if n < 2:
        return {"angle_mean": 0, "angle_std": 0, "angle_min": 0, "angle_max": 0, "angle_entropy": 0}
    norms = np.linalg.norm(vecs, axis=1)
    angles = []
    for i in range(n):
        for j in range(i+1, n):
            cosang = np.dot(vecs[i], vecs[j]) / (norms[i]*norms[j] + 1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    angles = np.array(angles)
    hist, _ = np.histogram(angles, bins=12, range=(0, 180), density=True)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    return {
        #"angle_mean": float(np.mean(angles)),
        #"angle_std": float(np.std(angles)),
        #"angle_min": float(np.min(angles)),
       # "angle_max": float(np.max(angles)),
        "angle_entropy": float(entropy)
    }

def dihedral_features(center, neighbor_coords):
    vecs = neighbor_coords - center
    n = len(vecs)
    if n < 4:
        return {"dihedral_mean": 0, "dihedral_std": 0, "dihedral_entropy": 0}
    angles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    a, b, c, d = vecs[i], vecs[j], vecs[k], vecs[l]
                    n1 = np.cross(a, b)
                    n2 = np.cross(c, d)
                    if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
                        continue
                    cosang = np.dot(n1, n2) / (np.linalg.norm(n1)*np.linalg.norm(n2))
                    angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    if len(angles) == 0:
        return {"dihedral_mean": 0, "dihedral_std": 0, "dihedral_entropy": 0}
    angles = np.array(angles)
    hist, _ = np.histogram(angles, bins=10, range=(0, 180), density=True)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    return {
        "dihedral_mean": float(np.mean(angles)),
        "dihedral_std": float(np.std(angles)),
        "dihedral_entropy": float(entropy)
    }


def steinhardt_QW(center, neighbor_coords, ls=[2, 4, 6, 8]):
    vecs = neighbor_coords - center
    n = len(vecs)
    if n == 0:
        return {f"Q{l}": 0 for l in ls} | {f"W{l}": 0 for l in ls}
    rs = np.linalg.norm(vecs, axis=1)
    rs[rs == 0] = 1e-10

    thetas = np.arccos(np.clip(vecs[:, 2] / rs, -1, 1))
    phis = np.arctan2(vecs[:, 1], vecs[:, 0])

    result = {}
    for l in ls:

        ms = np.arange(-l, l + 1)
        Ylms = sph_harm(ms[:, None], l, phis[None, :], thetas[None, :])

        Q_bar_lm = np.mean(Ylms, axis=1)

        sum_sq = np.sum(np.abs(Q_bar_lm) ** 2)
        Q_l = np.sqrt((4 * np.pi / (2 * l + 1)) * sum_sq)
        result[f"Q{l}"] = float(Q_l)

        W_l_val = 0.0

        norm_factor = (sum_sq) ** 1.5 + 1e-10

        for i, m1 in enumerate(range(-l, l + 1)):
            for j, m2 in enumerate(range(-l, l + 1)):
                m3 = -m1 - m2
                if -l <= m3 <= l:
                    k = m3 + l

                    term = Q_bar_lm[i] * Q_bar_lm[j] * Q_bar_lm[k]

                    W_l_val += np.real(term)

        result[f"W{l}"] = float(W_l_val / norm_factor)

    return result
def extract_motif_features(structural_motifs):
    motif_features = []
    for motif in structural_motifs:
        features = {}
        neighbors = motif['neighbors']
        neighbor_vectors = [n['vector'] for n in neighbors]
        distances = [n['distance'] for n in neighbors]
        angles = calculate_bond_angles(neighbors)

        # ---- 保留原有特征 ----
        n_neighbors = len(neighbors)
        features['coordination'] = n_neighbors
        if distances:
            features.update({
                'avg_distance': np.mean(distances),
                'distance_std': np.std(distances) if n_neighbors > 1 else 0,
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'distance_range': np.max(distances)-np.min(distances)
            })
        else:
            features.update({k:0 for k in ['avg_distance','distance_std','min_distance','max_distance','distance_range']})

        if angles:
            features.update({
                'avg_angle': np.mean(angles),
                'angle_std': np.std(angles),
                'angle_min': np.min(angles),
                'angle_max': np.max(angles),
                'angle_range': np.max(angles)-np.min(angles)
            })
        else:
            features.update({k:0 for k in ['avg_angle','angle_std','angle_min','angle_max','angle_range']})

        # ---- 原有拓扑、对称性、形状描述符 ----
        features.update(calculate_topological_features(neighbor_vectors, prefix=''))
        features.update(calculate_symmetry_features(neighbor_vectors, prefix=''))
        features.update(calculate_shape_descriptors(neighbor_vectors, prefix=''))

        # ---- 新增局部几何特征 ----
        center = np.zeros(3)  # 假设中心在原点
        neigh_coords = np.array(neighbor_vectors)
        features.update(radial_features(center, neigh_coords))
        features.update(angular_features(center, neigh_coords))
        features.update(dihedral_features(center, neigh_coords))
        features.update(steinhardt_QW(center, neigh_coords))

        motif_features.append(features)
    return motif_features

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np

def compute_motif_weights_standardized(features_list, method='hybrid', scale_method='zscore'):
    """
    计算基元加权值（标准化 + 方差/距离加权）
    features_list: list of dicts, 每个 dict 对应一个基元的特征
    method: 'variance', 'distance', 'hybrid'
    scale_method: 'zscore' 或 'minmax'
    """
    # === Step 1: 构建特征矩阵 ===
    all_keys = sorted({k for f in features_list for k in f.keys()})
    X = np.zeros((len(features_list), len(all_keys)))
    for i, f in enumerate(features_list):
        for j, k in enumerate(all_keys):
            v = f.get(k, 0.0)
            # 替换 inf 或 nan 为 nan
            if np.isinf(v) or np.isnan(v):
                v = np.nan
            X[i, j] = v

    # === Step 2: 填充 NaN ===
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # === Step 3: 标准化 ===
    if scale_method == 'zscore':
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale_method 必须是 'zscore' 或 'minmax'")
    X_scaled = scaler.fit_transform(X)

    # === Step 4: 计算方差权重 ===
    var_w = np.var(X_scaled, axis=1)  # 每个基元内部特征的方差

    # === Step 5: 计算距离权重（基元与其他基元特征距离） ===
    mean_vec = np.mean(X_scaled, axis=0)
    dist_w = np.linalg.norm(X_scaled - mean_vec, axis=1)

    # === Step 6: 合并权重 ===
    if method == 'variance':
        weights = var_w
    elif method == 'distance':
        weights = dist_w
    elif method == 'hybrid':
        weights = var_w + dist_w
    else:
        raise ValueError("method 必须是 'variance', 'distance' 或 'hybrid'")

    return weights, all_keys, X_scaled, var_w, dist_w


def weighted_features(features_list, weights):
    """
    根据每个基元的权重，计算整个样本的加权局部特征。

    Parameters:
    - features_list: list of dict，每个基元的局部特征
    - weights: array-like, 每个基元的权重 (长度与features_list一致)

    Returns:
    - aggregated_features: dict, 样本级加权特征
    """
    if len(features_list) != len(weights):
        raise ValueError("features_list 和 weights 长度必须一致")

    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)  # 归一化

    all_keys = sorted(set().union(*[f.keys() for f in features_list]))
    aggregated_features = {}

    for key in all_keys:
        # 收集每个基元的特征值
        values = np.array([feat.get(key, 0.0) for feat in features_list], dtype=float)
        # 加权平均
        weighted_mean = np.sum(values * weights)
        aggregated_features[f'{key}_mean'] = weighted_mean
        # 加权方差（可选，反映样本内部特征差异）
        weighted_var = np.sum(weights * (values - weighted_mean) ** 2)
        aggregated_features[f'{key}_var'] = weighted_var

    return aggregated_features


def is_cation(species_str, structure, site_index):
    """
    基于氧化态判断是否为阳离子
    """
    try:
        # 尝试获取该位点的氧化态
        oxi = structure[site_index].specie.oxi_state
        return oxi > 0
    except (AttributeError, ValueError, IndexError):
        # 兜底：查表法
        typical_cations = {
            'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba',
            'Sc', 'Y', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W',
            'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg',
            'Al', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb', 'Sb', 'Bi',
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
        }
        # 去掉氧化态数字，只留元素符号
        element = "".join([c for c in species_str if c.isalpha()])
        return element in typical_cations


def identify_anions_general(motifs, structure):
    """
    动态识别体系中的阴离子类型
    """
    anion_species = set()
    for motif in motifs:
        # 检查中心原子是否是阴离子
        idx = motif['central_atom']['index']
        try:
            if structure[idx].specie.oxi_state < 0:
                anion_species.add(motif['central_atom']['species'])
        except:
            pass

        # 检查邻居
        for n in motif['neighbors']:
            species_str = n['species']
            element = "".join([c for c in species_str if c.isalpha()])
            if element in {'O', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'N', 'P'}:
                anion_species.add(species_str)
    return sorted(list(anion_species))

# ----------------------
# 阴离子桥接计数
# ----------------------
def count_shared_anions(motif_i, motif_j, anion_species, tolerance_ratio=None):
    coords_i = [n['coords'] for n in motif_i['neighbors'] if n['species'] in anion_species]
    coords_j = [n['coords'] for n in motif_j['neighbors'] if n['species'] in anion_species]

    shared_count = 0
    spatial_tolerance = 0.1  # 0.1 Å 的物理重合容差

    for c_i in coords_i:
        for c_j in coords_j:
            # 计算欧几里得距离
            dist = np.linalg.norm(c_i - c_j)
            if dist < spatial_tolerance:
                shared_count += 1
                break  # 找到了匹配，跳出内层循环，继续找下一个 i

    return shared_count, [], []

from collections import defaultdict


def analyze_type_based_connectivity(G, motif_types):
    """分析基于基元类型的连接模式"""
    features = {}

    # 统计不同类型基元之间的连接
    type_connections = defaultdict(int)
    for i, j in G.edges():
        type_i = motif_types[i]
        type_j = motif_types[j]
        connection_key = f"{type_i}_{type_j}" if type_i <= type_j else f"{type_j}_{type_i}"
        type_connections[connection_key] += 1
    # 添加最重要的连接类型
    sorted_connections = sorted(type_connections.items(), key=lambda x: x[1], reverse=True)
    for idx, (conn_type, count) in enumerate(sorted_connections[:10]):  # 前5种连接类型
        features[f'motif_connection_{conn_type}'] = count

    return features


def calculate_cation_anion_ratio(structural_motifs, structure, anion_species):
    """
    计算阳离子-阴离子比例
    """
    cation_count = 0
    anion_count = 0

    for motif in structural_motifs:
        central_species = motif['central_atom']['species']
        idx = motif['central_atom']['index']

        if is_cation(central_species, structure, idx):
            cation_count += 1
        elif central_species in anion_species:
            anion_count += 1

    return cation_count / anion_count if anion_count > 0 else 0


# ============================================================================
# 1. 通用图特征提取
# ============================================================================
def extract_graph_features(adj_matrix, prefix):
    G = nx.from_numpy_array(adj_matrix)

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    n_nodes = G.number_of_nodes()

    if n_nodes > 0:
        # 连通分量 (最重要的指标)
        n_components = nx.number_connected_components(G)
        density = nx.density(G)

        # 度统计
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        degree_std = np.std(degrees)

        # 聚类系数
        avg_clustering = nx.average_clustering(G)
        n_edges = G.number_of_edges()

        # 路径与直径 (仅在最大子图中计算)
        if n_edges > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
            avg_path = nx.average_shortest_path_length(subgraph)
        else:
            diameter = 0;
            avg_path = 0
    else:
        # 空图兜底
        return {
            f'{prefix}_nodes': 0, f'{prefix}_edges': 0,
            f'{prefix}_components': 0, f'{prefix}_avg_degree': 0
        }

    return {
        f'{prefix}_nodes': n_nodes,
        f'{prefix}_edges': G.number_of_edges(),
        f'{prefix}_density': density,
        f'{prefix}_components': n_components,
        f'{prefix}_diameter': diameter,
        f'{prefix}_avg_path_length': avg_path,
        f'{prefix}_avg_degree': avg_degree,
        f'{prefix}_max_degree': max_degree,
        f'{prefix}_degree_std': degree_std,
        f'{prefix}_avg_clustering': avg_clustering
    }


import networkx as nx
from collections import defaultdict


def auto_detect_a_site_species(structural_motifs, cn_threshold=8.0):
    species_cn_map = defaultdict(list)

    for motif in structural_motifs:
        # 获取纯净的元素符号 (去掉电荷)
        sp_str = str(motif['central_atom']['species'])
        clean_sp = "".join([c for c in sp_str if c.isalpha()])

        # 记录该原子的配位数
        cn = motif['coordination_number']
        species_cn_map[clean_sp].append(cn)

    # 2. 计算平均配位数并判断
    ignore_list = []
    print("-" * 30)
    print("[Auto-Detect] 正在侦测 A 位阳离子...")

    for sp, cn_list in species_cn_map.items():
        avg_cn = np.mean(cn_list)
        print(f"   -> 元素 {sp}: 平均 CN = {avg_cn:.2f}")

        if avg_cn > cn_threshold:
            ignore_list.append(sp)
            print(f"判定为 A 位 (CN > {cn_threshold}) -> 加入忽略列表")
        else:
            print(f"判定为骨架 B 位/其他 -> 保留")

    print(f"自动生成的忽略列表: {ignore_list}")
    print("-" * 30)

    return ignore_list



def analyze_motif_connectivity(structural_motifs, structure,
                               distance_cutoff=None,
                               ignore_species=None,
                               cation_only=True):
    # --- A. 配置参数 ---
    if ignore_species is None: ignore_species = []
    if distance_cutoff is None: distance_cutoff = 6.0  # 稍微放宽一点默认值

    print("=" * 40)
    print(" 开始通用结构连接性分析")
    print(f"截断半径: {distance_cutoff:.2f} Å")
    print(f"忽略列表 (A位/杂质): {ignore_species}")
    print(f"仅分析阳离子模式: {cation_only}")

    # --- B. 识别阴离子 (用于多面体桥连判断) ---
    anion_list = identify_anions_general(structural_motifs, structure)
    print(f"识别到的桥连阴离子: {anion_list}")

    # --- C. 构建“白名单” (Active Nodes) ---
    # 我们只对白名单里的原子建立连接，其他的全部忽略
    n_motifs = len(structural_motifs)
    valid_indices = []

    for i in range(n_motifs):
        motif = structural_motifs[i]
        idx_struct = motif['central_atom']['index']
        sp_str = str(motif['central_atom']['species'])
        clean_sp = "".join([c for c in sp_str if c.isalpha()])  # 去掉电荷

        # 1. 过滤用户指定的 (A位等)
        if clean_sp in ignore_species:
            continue

        # 2. 过滤阴离子 (如果开启 cation_only)
        if cation_only:
            # 如果它是刚才识别出的阴离子，或者是公认的阴离子，踢掉
            if clean_sp in anion_list or clean_sp in ['O', 'F', 'Cl', 'Br', 'I']:
                continue

        valid_indices.append(i)

    print(f"🔹 有效骨架节点数: {len(valid_indices)} / {n_motifs} (已过滤 A位/阴离子)")

    # --- D. 建立矩阵 (只针对有效节点连线) ---
    adj_poly = np.zeros((n_motifs, n_motifs))
    adj_cation = np.zeros((n_motifs, n_motifs))
    matched_bridges = set()

    # 双重循环只跑 valid_indices，大幅提高效率
    for k in range(len(valid_indices)):
        i = valid_indices[k]
        motif_i = structural_motifs[i]

        for m in range(k + 1, len(valid_indices)):
            j = valid_indices[m]
            motif_j = structural_motifs[j]

            # --- 连接逻辑 1: 共用阴离子 (Polyhedral) ---
            ids_i = {n['index'] for n in motif_i['neighbors']
                     if "".join([c for c in str(n['species']) if c.isalpha()]) in anion_list}
            ids_j = {n['index'] for n in motif_j['neighbors']
                     if "".join([c for c in str(n['species']) if c.isalpha()]) in anion_list}

            shared = ids_i.intersection(ids_j)
            if shared:
                count = len(shared)
                adj_poly[i, j] = count
                adj_poly[j, i] = count
                matched_bridges.update(shared)

            # --- 连接逻辑 2: 直接距离 (Cation-Cation) ---
            dist = structure.lattice.get_distance_and_image(
                motif_i['central_atom']['coords'],
                motif_j['central_atom']['coords']
            )[0]

            if dist < distance_cutoff:
                adj_cation[i, j] = 1.0
                adj_cation[j, i] = 1.0

    # --- E. 提取特征 ---
    print("🔹 正在提取图特征 (自动清洗孤立节点)...")

    # 这里的关键是：adj矩阵里，被过滤掉的原子行和列全是0
    # extract_graph_features 里的 remove_isolates 会把它们自动踢出统计
    poly_feats = extract_graph_features(adj_poly, "poly")
    cat_feats = extract_graph_features(adj_cation, "cat")

    final = {**poly_feats, **cat_feats}
    final['global_total_anion_bridged'] = len(matched_bridges)

    # 计算连接类型比例
    n_links = np.sum(adj_poly > 0) / 2
    if n_links > 0:
        final['poly_frac_corner_sharing'] = (np.sum(adj_poly == 1) / 2) / n_links
        final['poly_frac_edge_sharing'] = (np.sum(adj_poly == 2) / 2) / n_links
        final['poly_frac_face_sharing'] = (np.sum(adj_poly >= 3) / 2) / n_links
    else:
        final['poly_frac_corner_sharing'] = 0.0
        final['poly_frac_edge_sharing'] = 0.0
        final['poly_frac_face_sharing'] = 0.0

    print("✅ 分析完成")
    return final



import numpy as np
from scipy.signal import find_peaks


def detect_optimal_cutoff(structure, bin_size=0.1, plot=False):

    cation_sites = []
    # 假设你外部已经有了 is_cation 判断，这里简化演示
    for i, s in enumerate(structure):
        if s.specie.symbol not in ['O', 'F', 'Cl', 'S']:  # 简单排除常见阴离子
            cation_sites.append(s)

    if len(cation_sites) < 2:
        return 4.0  # 无法计算时的保守值

    frac_coords = [s.frac_coords for s in cation_sites]
    dists = structure.lattice.get_all_distances(frac_coords, frac_coords)
    dists = dists.flatten()
    # 过滤掉 0 距离和太近的距离(小于 2.0 Å 通常不是阳离子间距)
    dists = dists[dists > 2.0]

    # =========================================================================
    # 2. 生成直方图 (Histogram)
    # =========================================================================
    bins = np.arange(0, 10.0 + bin_size, bin_size)
    hist, bin_edges = np.histogram(dists, bins=bins)

    # =========================================================================
    # 3. 寻找峰值 (Find Peaks)
    # =========================================================================
    # 适当降低 prominence 以识别分裂的小峰
    peaks, _ = find_peaks(hist, distance=3, prominence=np.max(hist) * 0.03)

    cutoff = 4.5  # 默认保底值

    if len(peaks) > 0:
        # 获取峰的位置 (Å)
        peak_positions = bin_edges[peaks]

        # --- 策略 B 核心逻辑 ---
        target_peak_idx = peaks[0]  # 默认瞄准第一个峰
        strategy_triggered = False

        if len(peaks) >= 2:
            first_peak_pos = peak_positions[0]
            second_peak_pos = peak_positions[1]
            peak_dist = second_peak_pos - first_peak_pos

            # 【判定条件】: 如果前两个峰距离小于 1.0 Å
            if peak_dist < 1.0:
                print(f"检测到峰分裂 (Δ={peak_dist:.2f} Å < 1.0 Å)。")
                print(f"   -> 将忽略第一个峰 ({first_peak_pos:.2f} Å)，瞄准第二个峰 ({second_peak_pos:.2f} Å)。")
                target_peak_idx = peaks[1]  # 瞄准第二个峰
                strategy_triggered = True
            else:

                if hist[peaks[1]] > hist[peaks[0]] * 1.5:
                    target_peak_idx = peaks[1]

        current_ptr = np.where(peaks == target_peak_idx)[0][0]

        if current_ptr < len(peaks) - 1:
            next_peak_idx = peaks[current_ptr + 1]
            search_range = hist[target_peak_idx: next_peak_idx]

            if len(search_range) > 0:
                valley_relative = np.argmin(search_range)
                valley_idx = target_peak_idx + valley_relative
                cutoff = bin_edges[valley_idx]

                if cutoff - bin_edges[target_peak_idx] < 0.5:
                    cutoff = bin_edges[target_peak_idx] + 0.8
            else:
                cutoff = bin_edges[target_peak_idx] + 1.0
        else:
            cutoff = bin_edges[target_peak_idx] + 1.2

        if cutoff < 3.2:
            print(f"警告: 计算出的 Cutoff ({cutoff:.2f} Å) 过小，触发硬性修正。")
            cutoff = 4.5

    # (可选) 画图调试
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 3.5))
            plt.bar(bin_edges[:-1], hist, width=bin_size, alpha=0.5, color='gray', label='RDF')
            plt.plot(bin_edges[peaks], hist[peaks], "x", color='red')

            # 标记我们选中的那个“目标峰”
            if len(peaks) > 0:
                plt.plot(bin_edges[target_peak_idx], hist[target_peak_idx], "o", color='green', label="Target Peak")

            plt.axvline(x=cutoff, color='blue', linestyle='--', linewidth=2, label=f"Cutoff={cutoff:.2f}")
            plt.title(f"Strategy B: Merge Split Peaks (Cutoff = {cutoff:.2f} Å)")
            plt.legend()
            plt.show()
        except:
            pass

    return round(float(cutoff), 2)


from scipy.stats import skew, kurtosis
from collections import defaultdict


def calculate_moments(values, prefix):
    """
    通用函数：将一堆距离/角度数据转换为 4 个可解释的统计特征
    """
    if len(values) < 1:
        return {f'{prefix}_mean': 0, f'{prefix}_std': 0, f'{prefix}_skew': 0, f'{prefix}_kurt': 0}

    return {
        f'{prefix}_mean': np.mean(values),  # 尺度 (Scale)
        f'{prefix}_std': np.std(values),  # 畸变 (Distortion)
        f'{prefix}_skew': skew(values),  # 偏向 (Asymmetry)
        f'{prefix}_kurt': kurtosis(values)  # 集中度 (Peakedness)
    }


def group_motifs_by_equivalence(structural_motifs):
    """按等价组分组 (Wyckoff > Species)"""
    eq_groups = defaultdict(list)
    for idx, motif in enumerate(structural_motifs):
        # 优先用 Wyckoff，其次用元素
        group = motif.get('central_atom', {}).get('wyckoff', None)
        if group is None:
            group = motif['central_atom']['species']
        eq_groups[group].append(idx)
    return eq_groups


def analyze_layer1_intra_motif(structural_motifs):

    eq_groups = group_motifs_by_equivalence(structural_motifs)
    features = []
    weights = []

    for group_name, indices in eq_groups.items():

        all_bond_lengths = []

        for i in indices:
            neighbors = structural_motifs[i]['neighbors']
            dists = [n['distance'] for n in neighbors]
            all_bond_lengths.extend(dists)

        if not all_bond_lengths: continue


        feats = calculate_moments(all_bond_lengths, 'L1_bond')

        # 额外补充：配位数的统计 (防止有些基元配位数不一致)
        cns = [len(structural_motifs[i]['neighbors']) for i in indices]
        feats['L1_cn_mean'] = np.mean(cns)

        features.append(feats)
        weights.append(len(indices))  # 权重为该组原子数量

    return compute_weighted_features_population(features, weights)


def analyze_layer2_intra_group(structural_motifs, structure):

    eq_groups = group_motifs_by_equivalence(structural_motifs)
    features = []
    weights = []

    for group_name, indices in eq_groups.items():
        if len(indices) < 2: continue

        frac_coords = [structural_motifs[i]['central_atom']['coords'] for i in indices]

        # 1. 计算子晶格内部的全距离矩阵
        dists_matrix = structure.lattice.get_all_distances(frac_coords, frac_coords)
        np.fill_diagonal(dists_matrix, np.inf)  # 排除自己

        # 2. 只取【最近邻】
        # 这代表了子晶格中原子排列的紧密程度
        nn_dists = np.min(dists_matrix, axis=1)

        # 3. 计算统计矩
        # prefix 命名为 'L2_lattice' (Layer 2)
        feats = calculate_moments(nn_dists, 'L2_lattice')

        features.append(feats)
        weights.append(len(indices))

    return compute_weighted_features_population(features, weights)

def analyze_layer3_inter_group(structural_motifs, structure):

    eq_groups = group_motifs_by_equivalence(structural_motifs)
    group_names = list(eq_groups.keys())
    features = []
    weights = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            indices_i = eq_groups[group_names[i]]
            indices_j = eq_groups[group_names[j]]

            coords_i = [structural_motifs[k]['central_atom']['coords'] for k in indices_i]
            coords_j = [structural_motifs[k]['central_atom']['coords'] for k in indices_j]

            # 计算两组之间的距离矩阵
            dists_matrix = structure.lattice.get_all_distances(coords_i, coords_j)
            if dists_matrix.size == 0: continue

            # 视角 A: 组 i 的每个原子，离组 j 最近的有多远？
            min_i_to_j = np.min(dists_matrix, axis=1)
            # 视角 B: 组 j 的每个原子，离组 i 最近的有多远？
            min_j_to_i = np.min(dists_matrix, axis=0)

            combined_nn = np.concatenate([min_i_to_j, min_j_to_i])

            # prefix 命名为 'L3_inter' (Layer 3)
            feats = calculate_moments(combined_nn, 'L3_inter')

            # 额外特征：绝对最小距离 (Steric Limit)
            feats['L3_global_min_dist'] = np.min(dists_matrix)

            features.append(feats)
            weights.append(1.0)

    return compute_weighted_features_population(features, weights)


def compute_three_layer_spatial_features(structural_motifs, structure):
    pass


    f1 = analyze_layer1_intra_motif(structural_motifs)

    f2 = analyze_layer2_intra_group(structural_motifs, structure)

    f3 = analyze_layer3_inter_group(structural_motifs, structure)

    # 合并所有特征
    final = {**f1, **f2, **f3}
    return final

def compute_weighted_features_population(feature_list, counts):
    if not feature_list: return {}
    all_keys = sorted(list(set().union(*(d.keys() for d in feature_list))))
    total_count = sum(counts)
    if total_count == 0: return {}
    weights = np.array(counts) / total_count

    weighted_features = {}
    X = np.zeros((len(feature_list), len(all_keys)))
    for i, f in enumerate(feature_list):
        for j, k in enumerate(all_keys):
            X[i, j] = f.get(k, np.nan)

    for j, k in enumerate(all_keys):
        col = X[:, j]
        mask = ~np.isnan(col)
        if np.any(mask):
            w = weights[mask]
            w /= w.sum()
            weighted_features[f'{k}_mean'] = np.sum(col[mask] * w)
            weighted_features[f'{k}_max'] = np.max(col[mask])
        else:
            weighted_features[f'{k}_mean'] = 0
            weighted_features[f'{k}_max'] = 0

    return weighted_features