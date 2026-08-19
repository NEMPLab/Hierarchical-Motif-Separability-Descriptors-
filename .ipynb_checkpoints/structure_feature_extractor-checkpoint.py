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
        unique_distances = len(set(np.round(distances, 2)))
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
        unique_angles = len(set(np.round(angles, 1)))
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

def steinhardt_QW(center, neighbor_coords, ls=[2,4,6,8]):
    vecs = neighbor_coords - center
    n = len(vecs)
    if n == 0:
        return {f"Q{l}":0 for l in ls} | {f"W{l}":0 for l in ls}
    def cart2sph(v):
        x,y,z = v
        r = np.linalg.norm(v)
        if r==0: return 0,0
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)
        return theta, phi
    thetas, phis = [], []
    for v in vecs:
        t,p = cart2sph(v)
        thetas.append(t)
        phis.append(p)
    result = {}
    for l in ls:
        Ylm = []
        for m in range(-l,l+1):
            Ylm.append(sph_harm(m,l,phis,thetas))
        Ylm = np.array(Ylm)
        Q_l = np.sqrt((4*np.pi/(2*l+1))*np.sum(np.abs(Ylm)**2)/n)
        result[f"Q{l}"] = float(np.real(Q_l))
        W_l_sum = 0
        for m1 in range(-l,l+1):
            for m2 in range(-l,l+1):
                m3 = -m1-m2
                if -l <= m3 <= l:
                    W_l_sum += np.sum(Ylm[m1+l]*Ylm[m2+l]*np.conj(Ylm[m3+l]))
        W_l = np.real(W_l_sum)/((np.sum(np.abs(Ylm)**2))**1.5 + 1e-10)
        result[f"W{l}"] = float(W_l)
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
        aggregated_features[f'{key}_weighted_mean'] = weighted_mean
        # 加权方差（可选，反映样本内部特征差异）
        weighted_var = np.sum(weights * (values - weighted_mean) ** 2)
        aggregated_features[f'{key}_weighted_var'] = weighted_var

    return aggregated_features


def is_cation(species):
    """判断元素是否是典型的阳离子"""
    typical_cations = {
        # 碱金属 (1族)
        'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',

        # 碱土金属 (2族)
        'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',

        # 过渡金属 (3-12族)
        'Sc', 'Y', 'La', 'Ac',  # 3族
        'Ti', 'Zr', 'Hf', 'Rf',  # 4族
        'V', 'Nb', 'Ta', 'Db',  # 5族
        'Cr', 'Mo', 'W', 'Sg',  # 6族
        'Mn', 'Tc', 'Re', 'Bh',  # 7族
        'Fe', 'Ru', 'Os', 'Hs',  # 8族
        'Co', 'Rh', 'Ir', 'Mt',  # 9族
        'Ni', 'Pd', 'Pt', 'Ds',  # 10族
        'Cu', 'Ag', 'Au', 'Rg',  # 11族
        'Zn', 'Cd', 'Hg', 'Cn',  # 12族

        # 镧系元素
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',

        # 锕系元素
        'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',

        # 主族金属 (13-16族中可能形成阳离子的元素)
        'Al', 'Ga', 'In', 'Tl', 'Nh',  # 13族
        'Sn', 'Pb', 'Fl',  # 14族
        'Bi', 'Mc',  # 15族
        'Po', 'Lv',  # 16族

        # 稀土元素
        'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
        'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',

        # 其他常见阳离子
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'
    }
    return species in typical_cations

def identify_anions_general(motifs):
    """通用阴离子识别，基于常见阴离子列表"""
    common_anions = {'O', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'N', 'P'}
    anions = set()
    for motif in motifs:
        for n in motif['neighbors']:
            if n['species'] in common_anions:
                anions.add(n['species'])
    return sorted(anions)

# ----------------------
# 阴离子桥接计数
# ----------------------
def count_shared_anions(motif_i, motif_j, anion_species, tolerance_ratio=0.15):
    shared_count, shared_details = 0, []
    anions_i = [{'species': n['species'], 'distance': float(n['distance']), 'id': id(n)}
                for n in motif_i['neighbors'] if n['species'] in anion_species]
    anions_j = [{'species': n['species'], 'distance': float(n['distance']), 'id': id(n)}
                for n in motif_j['neighbors'] if n['species'] in anion_species]
    matched_i, matched_j = set(), set()
    for ai in anions_i:
        best_match, min_distance_diff = None, None
        for aj in anions_j:
            if ai['species'] != aj['species']: continue
            distance_diff = abs(ai['distance'] - aj['distance'])
            avg_distance = (ai['distance'] + aj['distance']) / 2
            adaptive_tolerance = avg_distance * tolerance_ratio
            if distance_diff <= adaptive_tolerance:
                if best_match is None or distance_diff < min_distance_diff:
                    best_match, min_distance_diff = aj, distance_diff
        if best_match is not None:
            shared_count += 1
            shared_details.append({'species': ai['species'], 'distance_i': ai['distance'],
                                   'distance_j': best_match['distance'], 'distance_diff': min_distance_diff,
                                   'adaptive_tolerance': adaptive_tolerance, 'avg_distance': avg_distance,
                                   'id_i': ai['id'], 'id_j': best_match['id']})
            matched_i.add(ai['id'])
            matched_j.add(best_match['id'])
    return shared_count, shared_details, matched_i.union(matched_j)


from collections import defaultdict

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

def extract_graph_features(adjacency_matrix, structural_motifs):
    G = nx.from_numpy_array(adjacency_matrix)
    features = {}
    features['motif_network_nodes'] = G.number_of_nodes()
    features['motif_network_edges'] = G.number_of_edges()
    features['motif_network_density'] = nx.density(G)
    features['motif_network_components'] = nx.number_connected_components(G)
    diameters, avg_path_lengths = [], []
    for comp in nx.connected_components(G):
        subgraph = G.subgraph(comp)
        if len(subgraph) > 1:
            diameters.append(nx.diameter(subgraph))
            avg_path_lengths.append(nx.average_shortest_path_length(subgraph))
    features['motif_network_diameter'] = max(diameters) if diameters else 0
    features['motif_network_avg_path_length'] = np.mean(avg_path_lengths) if avg_path_lengths else 0
    degrees = [d for n, d in G.degree()]
    features['motif_avg_degree'] = np.mean(degrees) if degrees else 0
    features['motif_degree_std'] = np.std(degrees) if degrees else 0
    features['motif_max_degree'] = np.max(degrees) if degrees else 0
    features['motif_min_degree'] = np.min(degrees) if degrees else 0
    # 新增聚类系数
    features['motif_avg_clustering'] = nx.average_clustering(G, weight='weight')
    features['motif_clustering_std'] = np.std(list(nx.clustering(G, weight='weight').values()))
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        features['motif_avg_betweenness'] = np.mean(list(betweenness.values()))
        features['motif_max_betweenness'] = np.max(list(betweenness.values()))
    except:
        features['motif_avg_betweenness'] = 0
        features['motif_max_betweenness'] = 0
    return features


def calculate_cation_anion_ratio(structural_motifs, anion_species):
    """计算阳离子-阴离子比例"""
    cation_count = 0
    anion_count = 0

    for motif in structural_motifs:
        central_species = motif['central_atom']['species']
        if is_cation(central_species):
            cation_count += 1
        elif central_species in anion_species:
            anion_count += 1

    return cation_count / anion_count if anion_count > 0 else 0


from itertools import combinations
def find_elbow_point(cutoffs, connection_counts):
    """
    自动检测拐点（elbow point）
    逻辑：
      - 计算一阶导数（连接数增长率）
      - 计算二阶导数（增长率变化）
      - 拐点定义为二阶导数最大处（曲线由快变缓）
    """
    connection_counts = np.array(connection_counts, dtype=float)
    cutoffs = np.array(cutoffs, dtype=float)

    if len(cutoffs) < 3:
        return None  # 数据太少无法检测

    # 一阶导（增长率）
    first_diff = np.gradient(connection_counts, cutoffs)
    # 二阶导（增长率变化）
    second_diff = np.gradient(first_diff, cutoffs)

    # 找二阶导最大点（变化率最快）
    elbow_idx = np.argmax(second_diff)

    # 返回对应 cutoff
    if elbow_idx < len(cutoffs) - 1:
        return round(float(cutoffs[elbow_idx]), 3)
    return None

def detect_optimal_cutoff(structural_motifs, structure, cutoff_min=2.0, cutoff_max=8.0, step=0.1):
    """
    自动检测最优阳离子–阳离子截断距离（不画图）
    """
    # 提取阳离子坐标
    cation_coords = [
        structure.lattice.get_cartesian_coords(motif['central_atom']['coords'])
        for motif in structural_motifs
        if is_cation(motif['central_atom']['species'])
    ]

    # 所有阳离子对距离
    distances = []
    for i, j in combinations(range(len(cation_coords)), 2):
        d = structure.lattice.get_distance_and_image(cation_coords[i], cation_coords[j])[0]
        distances.append(d)
    distances = np.array(distances)

    # 扫描不同 cutoff
    cutoffs = np.arange(cutoff_min, cutoff_max + step, step)
    connection_counts = [np.sum(distances < c) for c in cutoffs]

    # 自动检测拐点
    best_cutoff = find_elbow_point(cutoffs, connection_counts)

    if best_cutoff:
        print(f"✅ 检测到拐点: 推荐截断距离 ≈ {best_cutoff:.2f} Å")
    else:
        print("⚠️ 未检测到明显拐点，请检查连接数分布。")

    return best_cutoff, cutoffs, connection_counts


def analyze_motif_connectivity(structural_motifs, structure,
                               distance_cutoff=None, tolerance_ratio=0.15,
                               cutoff_min=2.0, cutoff_max=8.0, step=0.1):
    """
    结构基元连接关系分析（阴离子桥接 + 阳离子直连 + 权重 + 熵统计）
    自动选择阳离子–阳离子截断距离。
    """
    # 1️⃣ 尝试自动检测截断距离
    if distance_cutoff is None:
        best_cutoff, _, _ = detect_optimal_cutoff(structural_motifs, structure,
                                                  cutoff_min=cutoff_min,
                                                  cutoff_max=cutoff_max,
                                                  step=step)
        if best_cutoff is not None:
            distance_cutoff = best_cutoff
        else:
            # 检测失败，使用默认值
            distance_cutoff = 3.5
            print(f"⚠️ 自动截断检测失败，使用默认距离: {distance_cutoff:.2f} Å")

    print(f"🔹 阳离子–阳离子截断半径: {distance_cutoff:.2f} Å")

    # 阴离子识别
    anion_species = identify_anions_general(structural_motifs)
    print(f"识别到的阴离子类型: {anion_species}")

    n_motifs = len(structural_motifs)
    adjacency_matrix = np.zeros((n_motifs, n_motifs))
    connection_details = {
        'anion_bridged': defaultdict(list),
        'direct_connection': [],
        'cation_anion_ratio': calculate_cation_anion_ratio(structural_motifs, anion_species)
    }

    total_pairs = 0
    connected_pairs = 0
    all_matched_anions = set()

    motif_weights = []
    motif_entropy = []

    # 遍历基元组合
    for i in range(n_motifs):
        motif_i = structural_motifs[i]
        neighbors_i = motif_i['neighbors']

        # 权重：阴离子数量
        weight_i = sum(1 for n in neighbors_i if n['species'] in anion_species)
        motif_weights.append(weight_i)

        # 熵计算
        species_counts = defaultdict(int)
        for n in neighbors_i:
            species_counts[n['species']] += 1

        probs = np.array(list(species_counts.values())) / sum(species_counts.values())
        entropy_i = -np.sum(probs * np.log2(probs))
        motif_entropy.append(entropy_i)

        for j in range(i + 1, n_motifs):
            total_pairs += 1
            motif_j = structural_motifs[j]

            # 阴离子桥接
            shared_count, shared_details, matched_ids = count_shared_anions(
                motif_i, motif_j, anion_species, tolerance_ratio
            )

            if shared_count > 0:
                adjacency_matrix[i, j] += shared_count
                adjacency_matrix[j, i] += shared_count
                connected_pairs += 1
                all_matched_anions.update(matched_ids)
                for detail in shared_details:
                    connection_details['anion_bridged'][detail['species']].append(1)

            # 阳离子 – 阳离子直连
            center_i = motif_i['central_atom']['coords']
            center_j = motif_j['central_atom']['coords']

            distance = structure.lattice.get_distance_and_image(center_i, center_j)[0]
            if distance is None or np.isnan(distance) or np.isinf(distance):
                continue

            if (is_cation(motif_i['central_atom']['species'])
                    and is_cation(motif_j['central_atom']['species'])
                    and distance < distance_cutoff):
                adjacency_matrix[i, j] += 1
                adjacency_matrix[j, i] += 1
                connection_details['direct_connection'].append({
                    'motif_pair': (i, j),
                    'distance': float(distance),
                    'type': 'cation-cation'
                })

    # 构建图并提取图特征
    G = nx.from_numpy_array(adjacency_matrix)
    connectivity_features = extract_graph_features(adjacency_matrix, structural_motifs)
  # connectivity_features['connection_details'] = dict(connection_details)
    connectivity_features['total_anion_bridged'] = len(all_matched_anions)

    # motif 权重统计
    if motif_weights:
        motif_weights = np.array(motif_weights, dtype=float)
        connectivity_features['weight_mean'] = np.nanmean(motif_weights)
        connectivity_features['weight_std'] = np.nanstd(motif_weights)
        connectivity_features['weight_max'] = np.nanmax(motif_weights)
        connectivity_features['weight_min'] = np.nanmin(motif_weights)
        connectivity_features['weight_median'] = np.nanmedian(motif_weights)

    # motif 熵统计
    if motif_entropy:
        motif_entropy = np.array(motif_entropy, dtype=float)
        connectivity_features['entropy_mean'] = np.nanmean(motif_entropy)
        connectivity_features['entropy_std'] = np.nanstd(motif_entropy)
        connectivity_features['entropy_max'] = np.nanmax(motif_entropy)
        connectivity_features['entropy_min'] = np.nanmin(motif_entropy)
        connectivity_features['entropy_median'] = np.nanmedian(motif_entropy)

    return connectivity_features


from scipy.stats import entropy, skew, kurtosis
def safe_entropy(values):
    hist, _ = np.histogram(values, bins='auto', density=True)
    hist = hist + 1e-12
    return entropy(hist)

def enhanced_shape_features(points):
    """全局空间增强特征：PCA、Convex Hull、NN统计、角度熵、偏心率"""
    pts = np.array(points)
    N = len(pts)
    feat = {}

    if N < 1:
        return feat

    centroid = np.mean(pts, axis=0)
    radial = np.linalg.norm(pts - centroid, axis=1)

    feat["radial_mean"] = np.mean(radial)
    feat["radial_std"]  = np.std(radial)
    feat["radial_cv"]   = feat["radial_std"]/(feat["radial_mean"]+1e-12)

    # PCA 各向异性
    if N >= 3:
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eig(cov)
        vals_sorted = np.sort(vals)[::-1]
        total = np.sum(vals_sorted)+1e-12

        feat["pca_axis_ratio"] = vals_sorted[0]/(vals_sorted[-1]+1e-12)
        feat["pca_1d_ratio"] = vals_sorted[0]/total
        feat["pca_2d_ratio"] = (vals_sorted[0]+vals_sorted[1])/total
        feat["pca_3d_ratio"] = total/total
    else:
        feat["pca_axis_ratio"] = np.nan
        feat["pca_1d_ratio"] = np.nan
        feat["pca_2d_ratio"] = np.nan
        feat["pca_3d_ratio"] = np.nan

    # Convex Hull
    if N >= 4:
        try:
            hull = ConvexHull(pts)
            feat["hull_volume"] = hull.volume
            feat["hull_area"] = hull.area
            feat["hull_density"] = N/(hull.volume+1e-12)
        except:
            feat["hull_volume"] = np.nan
            feat["hull_area"] = np.nan
            feat["hull_density"] = np.nan
    else:
        feat["hull_volume"] = np.nan
        feat["hull_area"] = np.nan
        feat["hull_density"] = np.nan

    # 最近邻统计 + 熵/偏度/峰度
    if N >= 3:
        kd = KDTree(pts)
        nn_d, _ = kd.query(pts, k=2)
        nn = nn_d[:,1]
        feat["nn_mean"] = np.mean(nn)
        feat["nn_std"]  = np.std(nn)
        feat["nn_cv"]   = feat["nn_std"]/(feat["nn_mean"]+1e-12)
        feat["nn_entropy"] = safe_entropy(nn)
        feat["nn_skew"]  = skew(nn)
        feat["nn_kurtosis"] = kurtosis(nn)
    else:
        feat["nn_mean"] = np.nan
        feat["nn_std"] = np.nan
        feat["nn_cv"] = np.nan
        feat["nn_entropy"] = np.nan
        feat["nn_skew"] = np.nan
        feat["nn_kurtosis"] = np.nan

    # 角度熵（方向均匀性）
    if N >= 3:
        vecs = pts - centroid
        angles = []
        for i in range(N):
            for j in range(i+1, N):
                v1, v2 = vecs[i], vecs[j]
                a = np.dot(v1, v2)
                b = np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12
                angles.append(np.arccos(np.clip(a/b, -1, 1)))
        angles = np.array(angles)
        if len(angles) > 0:
            feat["angle_mean"] = np.mean(angles)
            feat["angle_std"]  = np.std(angles)
            feat["angle_entropy"] = safe_entropy(angles)
    else:
        feat["angle_mean"] = np.nan
        feat["angle_std"] = np.nan
        feat["angle_entropy"] = np.nan

    # 偏心率（检测链状/通道状结构）
    if N >= 3:
        max_r = np.max(radial)
        min_r = np.min(radial) if np.min(radial) > 1e-6 else np.nan
        feat["eccentricity"] = max_r/(min_r+1e-12)
    else:
        feat["eccentricity"] = np.nan

    return feat

def group_motifs_by_equivalence(structural_motifs):
    """
    按等价原子组对基元进行分组
    每个基元需要包含 'equivalent_group_index' 字段
    """
    eq_groups = defaultdict(list)
    for idx, motif in enumerate(structural_motifs):
        # 使用你已有的字段
        group = motif.get('central_atom', {}).get('equivalent_group_index', f'unknown_{idx}')
        eq_groups[group].append(idx)
    return eq_groups

from scipy.spatial import distance, KDTree

def calculate_pairwise_distances(centers):
    """计算所有点对之间的欧几里得距离"""
    n = len(centers)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            distances.append(dist)
    return distances

def calculate_centroid(points):
    """计算点集的质心"""
    return np.mean(points, axis=0)


def compute_weighted_features(feature_list, method='hybrid'):
    """
    根据距离和方差计算加权特征
    method='hybrid'：距离均值 + 方差权重
    """
    if not feature_list:
        return {}

    all_keys = list(feature_list[0].keys())
    X = np.array([[f.get(k, 0.0) for k in all_keys] for f in feature_list])

    # 距离权重（均值）
    mean_vals = np.mean(X, axis=0)
    # 方差权重
    std_vals = np.std(X, axis=0) + 1e-8  # 防止除零
    if method == 'hybrid':
        weights = mean_vals / np.sum(mean_vals) + std_vals / np.sum(std_vals)
        weights /= np.sum(weights)
    else:
        weights = np.ones(len(all_keys)) / len(all_keys)

    weighted_features = {}
    for i, k in enumerate(all_keys):
        weighted_features[k] = np.sum(X[:, i] * weights[i])
    return weighted_features


def analyze_intragroup_arrangement(structural_motifs, structure, weight_method='hybrid'):
    """分析同一等价组内空间排布，单基元组跳过"""
    eq_groups = group_motifs_by_equivalence(structural_motifs)
    feature_list = []

    for group_name, indices in eq_groups.items():
        if len(indices) < 2:  # 单元素跳过
            continue
        centers = [structure.lattice.get_cartesian_coords(structural_motifs[i]['central_atom']['coords'])
                   for i in indices]

        feat = {}
        distances = calculate_pairwise_distances(centers)
        feat['space_avg_distance'] = np.mean(distances)
        feat['space_min_distance'] = np.min(distances)
        feat['space_max_distance'] = np.max(distances)
        feat['space_distance_std'] = np.std(distances) if len(distances) > 1 else 0

        # 质心与径向分布
        centroid = calculate_centroid(centers)
        radial_dists = [np.linalg.norm(np.array(c) - centroid) for c in centers]
        feat['space_radial_std'] = np.std(radial_dists)
        feat['space_radial_asymmetry'] = np.max(radial_dists) / (np.min(radial_dists) + 1e-8)

        # 最近邻
        if len(centers) > 1:
            kdtree = KDTree(centers)
            nn_dists, _ = kdtree.query(centers, k=min(2, len(centers)))
            if len(centers) > 1:
                nn_dists = nn_dists[:, 1]
                feat['space_nn_avg'] = np.mean(nn_dists)
                feat['space_nn_std'] = np.std(nn_dists)

        feature_list.append(feat)

    # 加权合并
    weighted_features = compute_weighted_features(feature_list, method=weight_method)
    return weighted_features


def analyze_intergroup_arrangement(structural_motifs, structure, weight_method='hybrid'):
    """分析不同等价组间空间排布"""
    eq_groups = group_motifs_by_equivalence(structural_motifs)
    group_names = list(eq_groups.keys())
    feature_list = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            indices_i = eq_groups[group_names[i]]
            indices_j = eq_groups[group_names[j]]
            centers_i = [structure.lattice.get_cartesian_coords(structural_motifs[idx]['central_atom']['coords']) for
                         idx in indices_i]
            centers_j = [structure.lattice.get_cartesian_coords(structural_motifs[idx]['central_atom']['coords']) for
                         idx in indices_j]

            feat = {}
            min_dist = float('inf')
            nn_dists = []
            directions = []
            for ci in centers_i:
                for cj in centers_j:
                    dist_vec = np.array(cj) - np.array(ci)
                    dist = np.linalg.norm(dist_vec)
                    min_dist = min(min_dist, dist)
                    directions.append(dist_vec)
            feat['space_inter_min_distance'] = min_dist
            if directions:
                dir_mags = [np.linalg.norm(v) for v in directions]
                feat['space_inter_direction_magnitude'] = np.mean(dir_mags)

            # 最近邻
            if centers_j and centers_i:
                kdtree_j = KDTree(centers_j)
                for ci in centers_i:
                    dists, _ = kdtree_j.query([ci], k=1)
                    nn_dists.append(dists[0])
                feat['space_inter_nn_avg'] = np.mean(nn_dists)
                feat['space_inter_nn_std'] = np.std(nn_dists) if len(nn_dists) > 1 else 0

            feature_list.append(feat)

    # 加权合并
    weighted_features = compute_weighted_features(feature_list, method=weight_method)
    return weighted_features


def comprehensive_weighted_spatial_analysis(representative_motifs,structural_motifs, structure, weight_method='hybrid'):
    """
    综合空间排布分析:
    1. 同一等价组内特征（跳过单元素组）
    2. 多等价组间特征
    3. 全局径向和角度对称性
    """
    # 1. 同组特征
    intra_features = analyze_intragroup_arrangement(structural_motifs, structure, weight_method)
    print("✅ 同一等价组空间排布特征:", intra_features)

    # 2. 多组特征
    inter_features = analyze_intergroup_arrangement(representative_motifs, structure, weight_method)
    #print("✅ 不同等价组间空间排布特征:", inter_features)

    # 3. 全局对称性
    all_centers = [structure.lattice.get_cartesian_coords(m['central_atom']['coords']) for m in structural_motifs]
    centroid = calculate_centroid(all_centers)
    radial_dists = [np.linalg.norm(np.array(c) - centroid) for c in all_centers]
    global_radial_std = np.std(radial_dists)

    vectors = [np.array(c) - centroid for c in all_centers]
    angles = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            norm = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            if norm > 0:
                dp = np.dot(vectors[i], vectors[j])
                angles.append(np.degrees(np.arccos(np.clip(dp / norm, -1, 1))))
    global_angle_std = np.std(angles) if angles else 0
    global_angle_range = (np.max(angles) - np.min(angles)) if angles else 0

    global_features = {'global_radial_std': global_radial_std,
                       'global_angle_std': global_angle_std,
                       'global_angle_range': global_angle_range}

    # 合并所有特征
    all_features = {}
    all_features.update(intra_features)
    all_features.update(inter_features)
    all_features.update(global_features)

    return all_features