import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN

def identify_asymmetric_unit(structure, symprec=0.01):
    """识别不对称单元中的不等价原子

    Args:
        structure: pymatgen Structure 对象
        symprec: 对称性分析精度

    Returns:
        tuple: (inequivalent_sites, equivalent_sites, space_group_info)
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)

    # 获取空间群信息
    space_group = analyzer.get_space_group_number()
    space_group_symbol = analyzer.get_space_group_symbol()

    # 获取对称化结构
    asymmetric_structure = analyzer.get_symmetrized_structure()

    # 获取不等价原子位点
    equivalent_sites = asymmetric_structure.equivalent_sites
    inequivalent_sites = [sites[0] for sites in equivalent_sites]

    return inequivalent_sites, equivalent_sites, (space_group, space_group_symbol),asymmetric_structure

def calculate_all_angles(center, neighbor_coords):
    """
    计算中心原子到所有邻居的两两夹角，返回 n x n 上三角矩阵
    """
    vectors = neighbor_coords - center  # shape: (n,3)
    norms = np.linalg.norm(vectors, axis=1)
    n = len(neighbor_coords)
    angles = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dot = np.dot(vectors[i], vectors[j])
            angles[i, j] = np.degrees(np.arccos(np.clip(dot / (norms[i]*norms[j]), -1.0, 1.0)))
    return angles

def classify_coordination(site, neighbors, angle_tolerance=15):
    """
    根据邻居几何手动分类配位环境，同时判断是否扭曲
    - CN=2~12
    - 返回字典：{'environment': str, 'distorted': bool}
    """
    n = len(neighbors)
    if n == 0:
        return {'environment': "isolated", 'distorted': False}

    center = np.array(site.coords)
    neighbor_coords = np.array([n['coords'] for n in neighbors])
    angles = calculate_all_angles(center, neighbor_coords)
    angles_flat = angles[np.triu_indices(n, k=1)]
    angles_flat = np.round(angles_flat, 2)

    # -----------------------------
    # 初始化
    # -----------------------------
    environment = "unknown"
    distorted = False

    # -----------------------------
    # 按 CN 分类
    # -----------------------------
    if n == 2:
        if abs(angles_flat[0]-180)<angle_tolerance:
            environment = "linear"
            distorted = False
        else:
            environment = "bent"
            distorted = True

    elif n == 3:
        if np.all(np.abs(angles_flat-120)<angle_tolerance):
            environment = "trigonal_planar"
            distorted = False
        else:
            environment = "trigonal_pyramidal"
            distorted = True

    elif n == 4:
        ninety_count = np.sum(np.abs(angles_flat - 90) < angle_tolerance)
        one_eighty_count = np.sum(np.abs(angles_flat - 180) < angle_tolerance)
        one_o_nine = np.sum(np.abs(angles_flat - 109.47) < angle_tolerance)

        if one_o_nine == 6:
            environment = "tetrahedral"
            distorted = np.any(np.abs(angles_flat - 109.47) > angle_tolerance)
        elif ninety_count >= 4 and one_eighty_count >= 2:
            environment = "square_planar"
            distorted = not (ninety_count >= 4 and one_eighty_count >= 2)
        else:
            environment = "4-coordinated"
            distorted = True

    elif n == 5:
        ninety_count = np.sum(np.abs(angles_flat - 90) < angle_tolerance)
        oneeighty_count = np.sum(np.abs(angles_flat - 180) < angle_tolerance)
        one_twenty_count = np.sum(np.abs(angles_flat - 120) < angle_tolerance)

        if one_twenty_count >= 3 and oneeighty_count >= 1 and ninety_count >= 3:
            environment = "trigonal_bipyramidal"
            distorted = not (one_twenty_count >= 3 and oneeighty_count >= 1 and ninety_count >= 3)
        else:
            environment = "square_pyramidal"
            distorted = True

    elif n == 6:
        ninety_count = np.sum(np.abs(angles_flat-90)<angle_tolerance)
        oneeighty_count = np.sum(np.abs(angles_flat-180)<angle_tolerance)
        if ninety_count>=8 and oneeighty_count>=3:
            environment = "octahedral"
            distorted = not (ninety_count>=8 and oneeighty_count>=3)
        else:
            environment = "6-coordinated"
            distorted = True

    elif n == 8:
        environment = "cubic"
        distorted = False
    elif n == 12:
        environment = "cuboctahedral"
        distorted = False
    else:
        environment = f"{n}-coordinated"
        distorted = False

    return {'environment': environment, 'distorted': distorted}


from pymatgen.analysis.local_env import CrystalNN
import numpy as np


def get_crystalnn_motifs(structure, symmetrized_structure, weight_threshold=0.8, angle_tolerance=15):
    """
    使用 CrystalNN 生成结构基元（每个原子都生成），标记等价组和代表原子。

    Args:
        structure: pymatgen Structure 对象
        symmetrized_structure: pymatgen SymmetrizedStructure 对象
        weight_threshold: CrystalNN 邻居权重阈值
        angle_tolerance: 角度分类扭曲阈值

    Returns:
        motifs: list of dict，每个 dict 为一个基元
    """
    cnn = CrystalNN()
    motifs = []
    global_indices = set()  # 检查是否覆盖所有原子

    for group_idx, group in enumerate(symmetrized_structure.equivalent_sites):
        wyckoff_symbol = symmetrized_structure.wyckoff_symbols[group_idx]
        representative_index = structure.sites.index(group[0])  # 每组的代表原子

        for site in group:
            site_index = structure.sites.index(site)
            global_indices.add(site_index)

            # 获取邻居信息
            nn_info = cnn.get_nn_info(structure, site_index)
            nn_info = [n for n in nn_info if n['weight'] >= weight_threshold]

            neighbors = []
            for n in nn_info:
                vector = n['site'].coords - site.coords
                neighbors.append({
                    'species': n['site'].species_string,
                    'coords': n['site'].coords,
                    'weight': n['weight'],
                    'distance': np.linalg.norm(vector),
                    'vector': vector,
                    'index': n['site_index']  # 全局索引
                })

            # 判断配位环境及扭曲
            coord_info = classify_coordination(site, neighbors, angle_tolerance)

            motif = {
                'central_atom': {
                    'species': site.species_string,
                    'coords': site.frac_coords,
                    'wyckoff': wyckoff_symbol,
                    'index': site_index,
                    'equivalent_group_index': group_idx,
                    'representative_index': representative_index
                },
                'coordination_number': len(neighbors),
                'coordination_environment': coord_info['environment'],
                'distorted': coord_info['distorted'],
                'neighbors': neighbors
            }
            motifs.append(motif)

    # 检查是否覆盖所有原子
    all_indices = set(range(len(structure.sites)))
    missing_indices = all_indices - global_indices
    if missing_indices:
        print(f"Warning: the following atoms were not included as centers: {missing_indices}")

    return motifs
