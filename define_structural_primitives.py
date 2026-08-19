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
    vectors = neighbor_coords - center
    norms = np.linalg.norm(vectors, axis=1)
    dot = vectors @ vectors.T
    cos = dot / np.outer(norms, norms)
    cos = np.clip(cos, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos))
    return angles


import numpy as np

def rms_deviation(values, ideal):
    """计算角度与理想值的 RMS 偏差"""
    values = np.array(values)
    ideal = np.array(ideal)
    return np.sqrt(np.mean((values - ideal)**2))


def classify_coordination(site, neighbors, angle_tolerance=15):
    """
    自动分类配位环境并判断是否扭曲
    返回:
        {
            'environment': str,
            'distorted': bool,
            'rms_angle_deviation': float
        }
    """

    n = len(neighbors)
    if n == 0:
        return {'environment': "isolated", 'distorted': False, 'rms_angle_deviation': 0}

    center = np.array(site.coords)
    neighbor_coords = np.array([nb["coords"] for nb in neighbors])
    angles = calculate_all_angles(center, neighbor_coords)
    angles_flat = np.round(angles[np.triu_indices(n, k=1)], 3)

    # -----------------------------
    # 理想角度模板库（可继续扩展）
    # -----------------------------
    ideal_geometries = {
        2:  {"linear": [180]},
        3:  {"trigonal_planar": [120, 120, 120]},
        4:  {
            "tetrahedral": [109.47] * 6,
            "square_planar": [90, 90, 90, 90, 180, 180]
        },
        5:  {
            "trigonal_bipyramidal": [90]*6 + [120]*3 + [180],
            "square_pyramidal": [90]*8 + [180]*2
        },
        6:  {
            "octahedral": [90]*12 + [180]*3
        },
        8:  {"cubic": [90]*12},
        12: {"cuboctahedral": [60]*12 + [90]*24}
    }

    # 配位数不在模板库 → 统称
    if n not in ideal_geometries:
        return {
            'environment': f"{n}-coordinated",
            'distorted': False,
            'rms_angle_deviation': 0
        }

    # -----------------------------
    # 自动匹配最佳环境
    # -----------------------------
    candidates = ideal_geometries[n]
    best_env = None
    best_rms = np.inf

    for env_name, ideal_angles in candidates.items():
        # 若角度数量不一致，按最接近填补
        target = np.array(ideal_angles)
        if len(target) != len(angles_flat):
            target = np.interp(
                np.linspace(0, 1, len(angles_flat)),
                np.linspace(0, 1, len(target)),
                target
            )

        rms = rms_deviation(angles_flat, target)
        if rms < best_rms:
            best_rms = rms
            best_env = env_name

    # -----------------------------
    # 扭曲判断：RMS 偏差大于阈值
    # -----------------------------
    distorted = best_rms > angle_tolerance

    return {
        'environment': best_env,
        'distorted': distorted,
        'rms_angle_deviation': float(best_rms)
    }


from pymatgen.analysis.local_env import CrystalNN
import numpy as np


import numpy as np

def get_crystalnn_motifs(structure, symmetrized_structure, inequivalent_sites, weight_threshold=0.8, angle_tolerance=15):
    """
    使用 CrystalNN 生成结构基元，并保存 RMS 角度偏差
    """
    cnn = CrystalNN()
    motifs = []

    for site in inequivalent_sites:
        site_index = structure.sites.index(site)
        # CrystalNN 获取邻居信息
        nn_info = cnn.get_nn_info(structure, site_index)
        nn_info = [n for n in nn_info if n['weight'] >= weight_threshold]

        # 构建邻居列表
        neighbors = []
        for n in nn_info:
            vector = n['site'].coords - site.coords
            neighbors.append({
                'species': n['site'].species_string,
                'coords': n['site'].coords,
                'weight': n['weight'],
                'distance': np.linalg.norm(vector),
                'vector': vector,
                'index': n['site_index']
            })

        # 获取 Wyckoff 符号
        wyckoff = None
        for i, group in enumerate(symmetrized_structure.equivalent_sites):
            if site in group:
                wyckoff = symmetrized_structure.wyckoff_symbols[i]
                break

        # 判断配位环境及扭曲
        coord_info = classify_coordination(site, neighbors, angle_tolerance)
        coordination_env = coord_info['environment']
        distorted = coord_info['distorted']
        # 获取 RMS 偏差
        rms_val = coord_info['rms_angle_deviation']

        # 构建基元字典
        motif = {
            'central_atom': {
                'species': site.species_string,
                'coords': site.frac_coords,
                'wyckoff': wyckoff,
                'index': site_index
            },
            'coordination_number': len(neighbors),
            'coordination_environment': coordination_env,
            'distorted': distorted,
            'rms_angle_deviation': rms_val,  # <--- 这里添加了 RMS 数据
            'neighbors': neighbors
        }
        motifs.append(motif)

    return motifs


import numpy as np
import math
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.bond_valence import BVAnalyzer


def clean_sp(s):
    """清洗元素符号字符串"""
    return "".join([c for c in str(s) if c.isalpha()])


def get_crystalnn_motifs_adaptive(structure, symmetrized_structure=None, weight_threshold=0.8, angle_tolerance=15):

    working_structure = structure.copy()
    print("-> [Step 0] 检查晶胞尺寸...")

    try:
        sga_check = SpacegroupAnalyzer(working_structure, symprec=0.01)
        primitive_structure = sga_check.get_primitive_standard_structure()

        if working_structure.num_sites == primitive_structure.num_sites:
            MIN_LATTICE_SIZE = 9.0
            scaling = [math.ceil(MIN_LATTICE_SIZE / l) for l in working_structure.lattice.abc]
            if any(s > 1 for s in scaling):
                print(f"   ℹ️ 执行物理扩胞: {scaling}")
                working_structure.make_supercell(scaling)
                symmetrized_structure = None
    except Exception as e:
        print(f"预处理检查跳过: {e}")

    # === 1. 氧化态策略：BVAnalyzer 优先 ===
    print("-> [Step 1] 计算氧化态...")
    try:
        bva = BVAnalyzer()
        working_structure = bva.get_oxi_state_decorated_structure(working_structure)
        print("  BVAnalyzer 计算成功。")
    except Exception as e:
        print(f"  BVAnalyzer 失败，尝试回退到 Guess 模式...")
        try:
            if not hasattr(working_structure[0].specie, "oxi_state"):
                working_structure.add_oxidation_state_by_guess()
            print(" Guess 模式成功。")
        except:
            print(" 氧化态猜测全部失败，将使用纯几何模式。")
            working_structure.remove_oxidation_states()

    # === 2. Phase 1 & 2: 构建图与对称化 ===
    cnn = CrystalNN(cation_anion=True, x_diff_weight=0.0)
    n_sites = len(working_structure)
    edge_weights = {}

    print("-> [Step 2] Phase 1: 拓扑扫描与内填...")
    raw_adjacency = {i: set() for i in range(n_sites)}

    for current_index in range(n_sites):
        site_with_oxi = working_structure[current_index]

        # A. 锚定
        try:
            nn_info = cnn.get_nn_info(working_structure, current_index)
        except:
            nn_info = []

        cnn_distances = []
        found_indices = set()
        valid_neighbors_ref = []

        for n in nn_info:
            if n['weight'] > 0:
                dist = working_structure[n['site_index']].distance(site_with_oxi)
                cnn_distances.append(dist)
                valid_neighbors_ref.append({'dist': dist, 'weight': n['weight']})

                if n['weight'] >= weight_threshold:
                    n_idx = n['site_index']
                    raw_adjacency[current_index].add(n_idx)
                    found_indices.add(n_idx)
                    edge_weights[(current_index, n_idx)] = n['weight']

        valid_neighbors_ref.sort(key=lambda x: x['dist'])

        # B. 边界
        if cnn_distances:
            cutoff_distance = max(cnn_distances) * 1.05
        else:
            cutoff_distance = 3.0

        # C. 内填救援
        candidates = working_structure.get_neighbors(site_with_oxi, cutoff_distance)
        center_neg = getattr(site_with_oxi.specie, "X", 0)
        center_oxi = getattr(site_with_oxi.specie, "oxi_state", 0)

        for dn in candidates:
            if dn.index == current_index or dn.index in found_indices: continue
            if dn.nn_distance < 0.1: continue

            # 化学过滤
            should_skip = False
            neighbor_neg = getattr(dn.specie, "X", 0)
            neighbor_oxi = getattr(dn.specie, "oxi_state", 0)

            if center_oxi != 0 and neighbor_oxi != 0:
                if center_oxi * neighbor_oxi > 0: should_skip = True
            elif abs(center_neg - neighbor_neg) < 0.5:
                should_skip = True

            if not should_skip:
                n_idx = dn.index
                raw_adjacency[current_index].add(n_idx)

                # 参考权重计算
                base_weight = 0.0
                if not valid_neighbors_ref:
                    base_weight = math.exp(1.0 - dn.nn_distance / 2.5)
                else:
                    closest_ref = min(valid_neighbors_ref, key=lambda x: abs(x['dist'] - dn.nn_distance))
                    base_weight = closest_ref['weight']

                penalty = 0.5
                if valid_neighbors_ref:
                    min_dist = valid_neighbors_ref[0]['dist']
                    if dn.nn_distance > min_dist:
                        penalty *= (min_dist / dn.nn_distance) ** 2

                edge_weights[(current_index, n_idx)] = float(f"{min(0.5, base_weight * penalty):.4f}")

    print("-> [Step 3] Phase 2: 全局对称化...")
    final_adjacency = {i: set() for i in range(n_sites)}

    for i in range(n_sites):
        for neighbor_idx in raw_adjacency[i]:
            final_adjacency[i].add(neighbor_idx)
            if i not in final_adjacency[neighbor_idx]:
                final_adjacency[neighbor_idx].add(i)
                # 权重镜像补全
                if (i, neighbor_idx) in edge_weights:
                    edge_weights[(neighbor_idx, i)] = edge_weights[(i, neighbor_idx)]
                elif (neighbor_idx, i) not in edge_weights:
                    edge_weights[(neighbor_idx, i)] = 0.05

    # =================================================================
    # Phase 3: 全原子特征提取 (含 Representative 映射)
    # =================================================================
    print("-> [Step 4] Phase 3: 生成全原子特征...")

    rep_mapping = {}  # index -> rep_index
    wyckoff_map = {}  # index -> '4a'
    group_id_map = {}  # index -> 0

    try:
        # 即使传入了 symmetrized_structure，如果Step0扩胞了，这里也得重算
        # 所以我们总是基于当前的 working_structure 算一次
        sga = SpacegroupAnalyzer(working_structure, symprec=0.01)
        symm_data = sga.get_symmetrized_structure()

        equiv_indices = symm_data.equivalent_indices
        wyckoff_symbols = symm_data.wyckoff_symbols

        for group_id, indices in enumerate(equiv_indices):
            # 约定：每个等价组的第一个原子是 Representative
            rep_index = indices[0]
            symbol = wyckoff_symbols[group_id]

            for idx in indices:
                rep_mapping[idx] = rep_index
                wyckoff_map[idx] = symbol
                group_id_map[idx] = group_id

    except Exception as e:
        print(f"⚠️ 对称性分析警告: {e}。将默认每个原子互不等价。")
        # 兜底：每个原子都是自己的代表
        for i in range(n_sites):
            rep_mapping[i] = i
            wyckoff_map[i] = 'a'
            group_id_map[i] = i

    motifs = []

    # 遍历所有原子
    for i in range(n_sites):
        center_site = working_structure[i]

        neighbor_indices = final_adjacency[i]
        final_neighbors = []
        for n_idx in neighbor_indices:
            n_site = working_structure[n_idx]
            dist = center_site.distance(n_site)
            vector = n_site.coords - center_site.coords
            w = edge_weights.get((i, n_idx), 0.0)

            final_neighbors.append({
                'species': clean_sp(n_site.specie),
                'coords': n_site.coords,
                'weight': w,
                'distance': dist,
                'vector': vector,
                'index': n_idx,
                'is_rescued': False
            })

        coord_info = classify_coordination(center_site, final_neighbors, angle_tolerance)

        motif = {
            'central_atom': {
                'species': clean_sp(center_site.specie),
                'coords': center_site.frac_coords,
                'index': i,
                # [关键] 填入映射好的 representative_index
                'representative_index': rep_mapping.get(i, i),
                'wyckoff': wyckoff_map.get(i, 'a'),
                'group_id': group_id_map.get(i, 0),
                'oxi_state': getattr(center_site.specie, "oxi_state", 0)
            },
            'coordination_number': len(final_neighbors),
            'coordination_environment': coord_info['environment'],
            'distorted': coord_info['distorted'],
            'rms_angle_deviation': coord_info['rms_angle_deviation'],
            'neighbors': sorted(final_neighbors, key=lambda x: x['distance'])
        }
        motifs.append(motif)

    clean_ret = working_structure.copy()
    clean_ret.remove_oxidation_states()

    return motifs, clean_ret