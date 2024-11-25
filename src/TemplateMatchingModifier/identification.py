#### Python Modifier Name ####
# Description of your Python-based modifier.

from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Bool, List
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from functools import cached_property
from ase.io.lammpsdata import read_lammps_data
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from multiprocessing import Pool


# Transformation matrices
_PREFACTOR = 1 / np.sqrt(2)
TO_DOUBLE_CELL = np.array([[_PREFACTOR, -_PREFACTOR, 0], [_PREFACTOR, _PREFACTOR, 0], [0, 0, 1]])
P_TC = TO_DOUBLE_CELL @ np.eye(3)
P_TB = TO_DOUBLE_CELL @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
P_TA = TO_DOUBLE_CELL @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

P_MC = np.eye(3)
P_MB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
P_MA = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

P_OC = np.eye(3)
P_OB = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
P_OA = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

# Symmetry operation matrices
MIRROR_XY = np.diag([-1, -1, 1])
MIRROR_XZ = np.diag([-1, 1, -1])
MIRROR_YZ = np.diag([1, -1, -1])
SWAP_XY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
SWAP_XZ = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
SWAP_YZ = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

# Define symmetry operations of each variant
T_SYM_OPS = [
    P_TC,
    P_TB,
    P_TA,
]
M_SYM_OPS = [
    P_MC, P_MC @ MIRROR_YZ, P_MC @ SWAP_XY, P_MC @ MIRROR_YZ @ SWAP_XY,
    P_MB, P_MB @ MIRROR_XZ, P_MB @ SWAP_XY, P_MB @ MIRROR_XZ @ SWAP_XY,
    P_MA, P_MA @ MIRROR_XY, P_MA @ SWAP_XZ, P_MA @ MIRROR_XY @ SWAP_XZ,
]
M2_SYM_OPS = [TO_DOUBLE_CELL @ x for x in M_SYM_OPS]
O_SYM_OPS = [
    P_OC, P_OC @ MIRROR_YZ, P_OC @ SWAP_XY, P_OC @ MIRROR_YZ @ SWAP_XY,
    P_OB, P_OB @ MIRROR_XZ, P_OB @ SWAP_XY, P_OB @ MIRROR_XZ @ SWAP_XY,
    P_OA, P_OA @ MIRROR_XY, P_OA @ SWAP_XZ, P_OA @ MIRROR_XY @ SWAP_XZ,
]


def norm_lat(v: np.ndarray, G: np.ndarray) -> float:
    if v.ndim == 2:
        return v.T @ G @ v # type: ignore
    else:
        return v @ G @ v.T # type: ignore


class TemplateMatchingModifier(ModifierInterface):
    xdir = List((1, 0, 0), label="X-axis")
    ydir = List((0, 1, 0), label="Y-axis")
    
    enable_tetra = Bool(True, label="Enable Tetragonal")
    enable_mono1 = Bool(True, label="Enable P21/c")
    enable_mono2 = Bool(False, label="Enable P21/m")
    enable_ortho = Bool(False, label="Enable Pbc21")
    
    cutoff = 2.8
    
    path_data = Path("data")
    path_tetra = path_data / "tetra.lmp"
    path_mono1 = path_data / "mono.lmp"
    path_mono2 = path_data / "P21m.lmp"
    path_ortho = path_data / "Pbc21.lmp"
       
    @cached_property
    def templates_and_ids(self):
        template_files = {
            "tetra": (self.path_tetra, T_SYM_OPS, self.enable_tetra),
            "mono1": (self.path_mono1, M_SYM_OPS, self.enable_mono1),
            "mono2": (self.path_mono2, M2_SYM_OPS, self.enable_mono2),
            "ortho": (self.path_ortho, O_SYM_OPS, self.enable_ortho),
        }

        sym_ops = []
        for file, ops, enabled in template_files.values():
            if enabled:
                atoms = read_lammps_data(file)
                sym_ops.append((ops, self.get_template_bonds(atoms)))

        templates, variant_ids = self.normalize_templates(sym_ops)

        return templates, variant_ids
    
    def get_template_bonds(self, atoms: Atoms):
        zr_idx = np.argwhere(atoms.get_atomic_numbers() != 8).flatten()

        i, D = neighbor_list("iD", atoms, self.cutoff)
        bonds_info = []
        for idx in zr_idx:
            bonds = D[i == idx]
            
            # Sort the bonds by distance
            lengths = np.linalg.norm(bonds, axis=1)
            idx = np.argsort(lengths)
            bonds = bonds[idx]        
            
            bonds_info.append(bonds)
            
        return bonds_info
    
    @staticmethod
    def normalize_templates(sym_ops_bond_envs):
        variants_bonds = []
        for ops, bonds in sym_ops_bond_envs:
            for op in ops:
                variants_bonds.append([x @ op for x in bonds])
            
        all_templates = []
        variant_ids = []
        for i, envs in enumerate(variants_bonds, start=1):
            for bonds_matrix in envs:
                template = bonds_matrix / np.linalg.norm(bonds_matrix, axis=1, keepdims=True)
                
                variant_ids.append(i)
                all_templates.append(template)

        return all_templates, variant_ids
    
    def box_to_crystal_coords(self):
        xdir, ydir = np.asarray(self.xdir), np.asarray(self.ydir)
        
        # Metric tensor
        S = read_lammps_data(self.path_tetra).cell.array
        G = S.T @ S

        # Get the z direction
        zdir = np.linalg.solve(G.T, np.cross(xdir, ydir))

        # Transformation matrix to map vectors in the box to orthogonal coordinates
        norm_xdir = np.sqrt(norm_lat(xdir, G))
        norm_ydir = np.sqrt(norm_lat(ydir, G))
        norm_zdir = np.sqrt(norm_lat(zdir, G))
        box2ortho = (
            G
            @ np.column_stack((xdir, ydir, zdir))
            @ np.linalg.inv(np.diag([norm_xdir, norm_ydir, norm_zdir]))
        )

        inv_Mb2o_T = np.linalg.inv(box2ortho.T)
        a_axis = np.array([1, 0, 0]) @ inv_Mb2o_T
        a_axis /= np.linalg.norm(a_axis)
        b_axis = np.array([0, 1, 0]) @ inv_Mb2o_T
        b_axis /= np.linalg.norm(b_axis)
        c_axis = np.array([0, 0, 1]) @ inv_Mb2o_T
        c_axis /= np.linalg.norm(c_axis)

        R_grain = np.vstack([a_axis, b_axis, c_axis])
        return R_grain
    
    def get_current_frame_info(self, data: DataCollection):
        positions = data.particles_.positions_[:]
        positions -= data.cell[:, 3]
        cell = data.cell[:, :3].T
        pbc = np.array(data.cell.pbc)
        types = np.array([t for t in data.particles.particle_types])
        
        return positions, cell, pbc, types
    
    def get_periodic_neighbors(self, data, target, pbc):
        tree_box = data.max(axis=0) + 3 * self.cutoff * ~pbc + 1e-6
        tree = cKDTree(data, boxsize=tree_box)
        nn = tree.query_ball_point(target, self.cutoff, p=2.0, workers=-1)
        return nn
        
    
    def process_atom(self, i):
        current_zr_bonds = self.zr_bonds_crystal_frame_split[i]
        normalized_bonds = current_zr_bonds / np.linalg.norm(current_zr_bonds, axis=1, keepdims=True)
        current_coord = normalized_bonds.shape[0]
        
        best_angle = np.inf
        best_rmsd = -1.0
        best_variant = 0

        sorted_bonds = np.zeros_like(normalized_bonds)
        variant_id, templates = self.templates_and_ids
        for variant, template in zip(variant_id, templates):
            expected_coord = template.shape[0]
            if current_coord != expected_coord:
                continue
            
            # Sort the rows to minimize the distance between the arrays
            cost_matrix = cdist(normalized_bonds, template)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            sorted_bonds[col_ind] = normalized_bonds[row_ind]

            rot, rmsd = R.align_vectors(sorted_bonds, template, return_sensitivity=False) # type: ignore
            angle = np.rad2deg(np.arccos((np.trace(rot.as_matrix()) - 1) / 2))

            if angle < best_angle and (rmsd < best_rmsd or best_rmsd == -1.0):
                best_angle = angle
                best_rmsd = rmsd
                best_variant = variant
                
        return best_angle, best_rmsd, best_variant

    def modify(self, data: DataCollection, frame: int, **kwargs):
        assert len(self.xdir) == 3 and len(self.ydir) == 3 # type: ignore

        pos, cell, pbc, type_ids = self.get_current_frame_info(data)
        
        # Remove unnecessary atoms
        mask = type_ids > 3
        type_ids = type_ids[~mask]
        pos = pos[~mask]
        
        # Separate anions and cations
        mask_anions = type_ids == 1
        o_sample_pos = pos[mask_anions]
        zr_sample_pos = pos[~mask_anions]
        
        # We need to translate the positions to make sure there is no negative coordinates
        # This can happend when the box is not orthogonal
        translated_pos_o = o_sample_pos - np.min(pos, axis=0)
        translated_pos_zr = zr_sample_pos - np.min(pos, axis=0)
        
        # Find the nearest neighbors
        zr_nn = self.get_periodic_neighbors(translated_pos_o, translated_pos_zr, pbc)
        
        # Get bonds
        zr_nn_flat = np.concatenate(zr_nn)
        zr_bond_indices = np.repeat(np.arange(len(zr_sample_pos)), np.fromiter(map(len, zr_nn), dtype=int))
        zr_bonds = zr_sample_pos[zr_bond_indices] - o_sample_pos[zr_nn_flat]
        
        # Rotate bonds to crystal frame
        R_grain = self.box_to_crystal_coords()
        zr_bonds_crystal_frame = zr_bonds @ np.linalg.inv(R_grain)
        
        # Process the atoms in parallel
        self.zr_bonds_crystal_frame_split = np.split(zr_bonds_crystal_frame, np.cumsum(np.bincount(zr_bond_indices)[:-1]))
        with Pool() as pool:
            results = list(pool.map(self.process_atom, range(len(zr_sample_pos))))
            
        # Prepare arrays to store the results
        zr_angle = np.full(len(zr_sample_pos), np.inf)
        zr_rmsd = np.full(len(zr_sample_pos), -1.0)
        zr_variant = np.empty(len(zr_sample_pos), dtype="uint8")

        # Unpack the results
        for j, (angle, rmsd, variant) in enumerate(results):
            zr_angle[j] = angle
            zr_rmsd[j] = rmsd
            zr_variant[j] = variant