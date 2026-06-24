# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from numpy.random import choice
from scipy import interpolate
from scipy import ndimage

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:

    BASE_TERRAIN_NAMES = [
        'slope', 'rough_slope', 'stairs_up', 'stairs_down',
        'discrete', 'stepping_stones', 'gap', 'pit',
    ]
    EXTRA_TERRAIN_NAMES = [
        'parkour', 'parkour_hurdle', 'parkour_flat', 'parkour_step', 'parkour_gap',
        'T_step_stl', 'Slope', 'BridgeA', 'BridgeB',
    ]
    STL_TERRAIN_CONFIG = {
        'T_step_stl': {
            'heightmap_path': 'legged_gym/legged_gym/terrain_assets/height_maps/T_step.npy',
            'display_name': 'T_step',
            'goals': [(100, 125), (125, 125), (140, 125), (170, 125), (170, 155), (170, 175), (170, 190), (170, 200)],
        },
        'Slope': {
            'heightmap_path': 'legged_gym/legged_gym/terrain_assets/height_maps/Slope.npy',
            'display_name': 'Slope',
            'goals': [(55, 124), (95, 124), (125, 124), (150, 124), (170, 124), (190, 124), (200, 124), (235, 124)],
        },
        'BridgeA': {
            'heightmap_path': 'legged_gym/legged_gym/terrain_assets/height_maps/BridgeA.npy',
            'display_name': 'BridgeA',
            'goals': [(65, 100), (95, 100), (125, 100), (150, 100), (180, 100), (210, 100), (230, 100), (235, 140)],
        },
        'BridgeB': {
            'heightmap_path': 'legged_gym/legged_gym/terrain_assets/height_maps/BridgeB.npy',
            'display_name': 'BridgeB',
            'goals': [(70, 106), (110, 106), (145, 106), (155, 106), (165, 106), (175, 106), (185, 106), (195, 106)],
        },
    }

    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.num_goals = int(getattr(cfg, 'num_goals', 0))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, max(self.num_goals, 0), 3), dtype=np.float32)
        self.terrain_idx_map = np.full((cfg.num_rows, cfg.num_cols), fill_value=int(getattr(cfg, 'terrain_idx_unknown', -1)), dtype=np.int64)
        self.terrain_type = self.terrain_idx_map
        self.terrain_names = self.BASE_TERRAIN_NAMES + self.EXTRA_TERRAIN_NAMES
        self.terrain_name_to_idx = {name: idx for idx, name in enumerate(self.terrain_names)}
        self.stl_env_length = float(getattr(cfg, 'stl_terrain_length', cfg.terrain_length))
        self.stl_env_width = float(getattr(cfg, 'stl_terrain_width', cfg.terrain_width))
        self._prepare_terrain_proportions()
        self.terrain_assert()
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    

    def _prepare_terrain_proportions(self):
        base_props = list(getattr(self.cfg, 'terrain_proportions', []))
        if len(base_props) < len(self.BASE_TERRAIN_NAMES):
            base_props = base_props + [0.] * (len(self.BASE_TERRAIN_NAMES) - len(base_props))
        extra_cfg = getattr(self.cfg, 'terrain_extra_proportions', {}) or {}
        extra_props = [float(extra_cfg.get(name, 0.)) for name in self.EXTRA_TERRAIN_NAMES]
        props = np.array(base_props[:len(self.BASE_TERRAIN_NAMES)] + extra_props, dtype=np.float64)
        if not np.isfinite(props).all() or np.any(props < 0):
            raise ValueError(f'Invalid terrain proportions: {props}')
        if props.sum() <= 0:
            props[0] = 1.0
        props = props / props.sum()
        self.terrain_proportions_full = props
        self.proportions = np.cumsum(props).tolist()

    def terrain_assert(self):
        if len(self.proportions) != len(self.terrain_names):
            raise AssertionError('terrain proportions and terrain names length mismatch')
        for name in self.EXTRA_TERRAIN_NAMES:
            if name not in self.terrain_name_to_idx:
                raise AssertionError(f'missing terrain idx for {name}')
        for name, cfg in self.STL_TERRAIN_CONFIG.items():
            path = self._resolve_project_path(cfg['heightmap_path'])
            proportion = self._terrain_proportion(name)
            if proportion > 0. and not os.path.exists(path):
                raise FileNotFoundError(f'{name} heightmap not found: {path}')
            goals = cfg.get('goals', None)
            if goals is not None:
                goals = np.asarray(goals)
                if goals.ndim != 2 or goals.shape[1] != 2:
                    raise AssertionError(f'{name} goals must have shape [N, 2]')

    def _resolve_project_path(self, path):
        if os.path.isabs(path):
            return path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        return os.path.join(project_root, path)

    def _terrain_idx(self, name):
        return int(self.terrain_name_to_idx[name])

    def _terrain_proportion(self, name):
        idx = self.terrain_name_to_idx[name]
        return float(self.terrain_proportions_full[idx])

    def _make_stl_heightmap_terrain(self, terrain_name):
        terrain = terrain_utils.SubTerrain(
            'terrain',
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        terrain.terrain_length = self.env_length
        terrain.terrain_width = self.env_width
        stl_config = self.STL_TERRAIN_CONFIG[terrain_name]
        stl_heightmap_terrain(
            terrain,
            terrain_name=stl_config['display_name'],
            stl_heightmap_path=self._resolve_project_path(stl_config['heightmap_path']),
            goal_positions=stl_config.get('goals', None),
        )
        return terrain

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        amplitude = 0.01 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        idx = self._terrain_idx('pit')

        if choice < self.proportions[0]:
            idx = self._terrain_idx('slope')
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            idx = self._terrain_idx('rough_slope')
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                idx = self._terrain_idx('stairs_up')
                step_height *= -1
            else:
                idx = self._terrain_idx('stairs_down')
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            idx = self._terrain_idx('discrete')
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 20, platform_size=3.)
        elif choice < self.proportions[5]:
            idx = self._terrain_idx('stepping_stones')
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            idx = self._terrain_idx('gap')
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            idx = self._terrain_idx('pit')
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[8]:
            idx = self._terrain_idx('parkour')
            self.make_parkour_terrain(terrain, choice, difficulty)
        elif choice < self.proportions[9]:
            idx = self._terrain_idx('parkour_hurdle')
            self.make_parkour_terrain(terrain, choice, difficulty, terrain_kind='parkour_hurdle')
        elif choice < self.proportions[10]:
            idx = self._terrain_idx('parkour_flat')
            self.make_parkour_terrain(terrain, choice, difficulty, terrain_kind='parkour_flat')
        elif choice < self.proportions[11]:
            idx = self._terrain_idx('parkour_step')
            self.make_parkour_terrain(terrain, choice, difficulty, terrain_kind='parkour_step')
        elif choice < self.proportions[12]:
            idx = self._terrain_idx('parkour_gap')
            self.make_parkour_terrain(terrain, choice, difficulty, terrain_kind='parkour_gap')
        elif choice < self.proportions[13]:
            idx = self._terrain_idx('T_step_stl')
            terrain = self._make_stl_heightmap_terrain('T_step_stl')
        elif choice < self.proportions[14]:
            idx = self._terrain_idx('Slope')
            terrain = self._make_stl_heightmap_terrain('Slope')
        elif choice < self.proportions[15]:
            idx = self._terrain_idx('BridgeA')
            terrain = self._make_stl_heightmap_terrain('BridgeA')
        else:
            idx = self._terrain_idx('BridgeB')
            terrain = self._make_stl_heightmap_terrain('BridgeB')

        terrain.idx = idx
        return terrain

    def make_parkour_terrain(self, terrain, choice, difficulty, terrain_kind=None):
        if terrain_kind == 'parkour_hurdle':
            num_inner = max(int(getattr(self.cfg, 'num_goals', self.num_goals)) - 2, 1)
            parkour_hurdle_terrain(terrain, num_goals=self.num_goals, num_hurdles=num_inner, difficulty=difficulty)
            return terrain
        if terrain_kind == 'parkour_flat':
            parkour_flat_terrain(terrain, num_goals=self.num_goals, difficulty=difficulty)
            return terrain
        if terrain_kind == 'parkour_step':
            num_inner = max(int(getattr(self.cfg, 'num_goals', self.num_goals)) - 2, 1)
            parkour_step_terrain(terrain, num_goals=self.num_goals, num_steps=num_inner, difficulty=difficulty)
            return terrain
        if terrain_kind == 'parkour_gap':
            num_inner = max(int(getattr(self.cfg, 'num_goals', self.num_goals)) - 2, 1)
            parkour_gap_terrain(terrain, num_goals=self.num_goals, num_gaps=num_inner, difficulty=difficulty)
            return terrain

        proportions = getattr(self.cfg, 'parkour_terrain_proportions', [0.2, 0.2, 0.2, 0.2, 0.2])
        if len(proportions) == 0:
            proportions = [1.0]
        proportions = np.asarray(proportions, dtype=np.float64)
        proportions = proportions / max(np.sum(proportions), 1e-8)
        cumulative = np.cumsum(proportions)
        num_inner = max(int(getattr(self.cfg, 'num_goals', self.num_goals)) - 2, 1)
        y_range = getattr(self.cfg, 'y_range', [-0.4, 0.4])
        if choice < cumulative[0]:
            parkour_terrain(terrain, num_stones=num_inner, x_range=[-0.1, 0.1 + 0.3 * difficulty], y_range=[0.2, 0.3 + 0.1 * difficulty], stone_len=[0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty], stone_width=1.0, incline_height=0.25 * difficulty, last_incline_height=0.25 * difficulty + 0.1 - 0.1 * difficulty, pad_height=0, pit_depth=[0.2, 1])
        elif len(cumulative) > 1 and choice < cumulative[1]:
            parkour_hurdle_terrain(terrain, num_goals=self.num_goals, num_hurdles=num_inner, difficulty=difficulty)
        elif len(cumulative) > 2 and choice < cumulative[2]:
            parkour_flat_terrain(terrain, num_goals=self.num_goals, difficulty=difficulty)
        elif len(cumulative) > 3 and choice < cumulative[3]:
            parkour_step_terrain(terrain, num_goals=self.num_goals, num_steps=num_inner, difficulty=difficulty)
        else:
            parkour_gap_terrain(terrain, num_goals=self.num_goals, num_gaps=num_inner, difficulty=difficulty)
        return terrain

    def _normalized_terrain_goals(self, terrain):
        if self.num_goals <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        fallback = straight_goal_line(terrain, self.num_goals)
        goals = getattr(terrain, 'goals', fallback)
        goals = np.asarray(goals, dtype=np.float32)
        if goals.ndim != 2 or goals.shape[0] == 0 or goals.shape[1] not in (2, 3) or not np.all(np.isfinite(goals)):
            goals = fallback
        elif goals.shape[1] == 2:
            goals = np.pad(goals, ((0, 0), (0, 1)), mode='constant')
        if goals.shape[0] < self.num_goals:
            goals = np.concatenate((goals, np.repeat(goals[-1:], self.num_goals - goals.shape[0], axis=0)), axis=0)
        return goals[:self.num_goals].astype(np.float32)

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        if not hasattr(terrain, 'idx'):
            terrain.idx = int(getattr(self.cfg, 'terrain_idx_unknown', -1))
        self.terrain_idx_map[row, col] = int(terrain.idx)
        if self.num_goals > 0:
            self.goals[i, j, :, :] = self._normalized_terrain_goals(terrain) + np.array([i * self.env_length, j * self.env_width, 0.], dtype=np.float32)

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def _meters_to_px(terrain, value):
    return max(1, int(round(value / terrain.horizontal_scale)))

def _meters_to_height(terrain, value):
    return int(round(value / terrain.vertical_scale))

def straight_goal_line(terrain, num_goals):
    num_goals = max(int(num_goals), 1)
    start_x = terrain.length * terrain.horizontal_scale * 0.5 + 0.5
    end_x = terrain.length * terrain.horizontal_scale - 0.8
    if end_x <= start_x:
        end_x = start_x + 0.1
    x = np.linspace(start_x, end_x, num_goals, dtype=np.float32)
    y = np.full(num_goals, terrain.width * terrain.horizontal_scale * 0.5, dtype=np.float32)
    z = np.zeros(num_goals, dtype=np.float32)
    return np.stack((x, y, z), axis=-1)

def _goal_pixels(terrain, goals):
    xs = np.clip(np.round(goals[:, 0] / terrain.horizontal_scale).astype(np.int64), 1, terrain.length - 2)
    ys = np.clip(np.round(goals[:, 1] / terrain.horizontal_scale).astype(np.int64), 1, terrain.width - 2)
    return xs, ys

def _pad_parkour_edges(terrain, pad_width=0.1, pad_height=0.5):
    pad_w = _meters_to_px(terrain, pad_width)
    pad_h = _meters_to_height(terrain, pad_height)
    terrain.height_field_raw[:, :pad_w] = pad_h
    terrain.height_field_raw[:, -pad_w:] = pad_h
    terrain.height_field_raw[:pad_w, :] = pad_h
    terrain.height_field_raw[-pad_w:, :] = pad_h

def parkour_flat_terrain(terrain, num_goals=8, difficulty=0.0):
    terrain.goals = straight_goal_line(terrain, num_goals)[:, :2]
    _pad_parkour_edges(terrain)

def parkour_hurdle_terrain(terrain, num_goals=8, num_hurdles=6, difficulty=0.0):
    goals = straight_goal_line(terrain, num_goals)
    xs, ys = _goal_pixels(terrain, goals)
    hurdle_len = _meters_to_px(terrain, 0.25 + 0.10 * difficulty)
    half_width = _meters_to_px(terrain, 0.45 + 0.25 * difficulty)
    height = _meters_to_height(terrain, 0.12 + 0.18 * difficulty)
    for x, y in zip(xs[1:-1], ys[1:-1]):
        terrain.height_field_raw[max(0, x - hurdle_len // 2):min(terrain.length, x + hurdle_len // 2 + 1), :] = height
        terrain.height_field_raw[max(0, x - hurdle_len // 2):min(terrain.length, x + hurdle_len // 2 + 1), max(0, y - half_width):min(terrain.width, y + half_width)] = 0
    terrain.goals = goals[:, :2]
    _pad_parkour_edges(terrain)

def parkour_gap_terrain(terrain, num_goals=8, num_gaps=6, difficulty=0.0):
    goals = straight_goal_line(terrain, num_goals)
    xs, ys = _goal_pixels(terrain, goals)
    gap = _meters_to_px(terrain, 0.18 + 0.28 * difficulty)
    half_width = _meters_to_px(terrain, 0.55 + 0.25 * difficulty)
    depth = -_meters_to_height(terrain, 0.45 + 0.45 * difficulty)
    for x, y in zip(xs[1:-1], ys[1:-1]):
        terrain.height_field_raw[max(0, x - gap // 2):min(terrain.length, x + gap // 2 + 1), :] = depth
        terrain.height_field_raw[max(0, x - gap // 2):min(terrain.length, x + gap // 2 + 1), max(0, y - half_width):min(terrain.width, y + half_width)] = 0
    terrain.goals = goals[:, :2]
    _pad_parkour_edges(terrain)

def parkour_step_terrain(terrain, num_goals=8, num_steps=6, difficulty=0.0):
    goals = straight_goal_line(terrain, num_goals)
    xs, _ = _goal_pixels(terrain, goals)
    step_h = _meters_to_height(terrain, 0.08 + 0.14 * difficulty)
    height = 0
    last_x = max(0, xs[0])
    for idx, x in enumerate(xs[1:]):
        if idx < max(1, len(xs) // 2):
            height += step_h
        else:
            height = max(0, height - step_h)
        terrain.height_field_raw[last_x:min(terrain.length, x), :] = height
        last_x = min(terrain.length - 1, x)
    terrain.height_field_raw[last_x:, :] = height
    terrain.goals = goals[:, :2]
    _pad_parkour_edges(terrain)

def parkour_stair_terrain(terrain, num_goals=8, num_steps=6, difficulty=0.0):
    goals = straight_goal_line(terrain, num_goals)
    xs, _ = _goal_pixels(terrain, goals)
    step_h = _meters_to_height(terrain, 0.06 + 0.12 * difficulty)
    for idx in range(len(xs) - 1):
        terrain.height_field_raw[xs[idx]:xs[idx + 1], :] = idx * step_h
    terrain.height_field_raw[xs[-1]:, :] = max(0, len(xs) - 1) * step_h
    terrain.goals = goals[:, :2]
    _pad_parkour_edges(terrain)


def parkour_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=8, x_range=[1.8, 1.9], y_range=[0., 0.1], z_range=[-0.2, 0.2], stone_len=[1.0, 1.0], stone_width=0.6, pad_width=0.1, pad_height=0.5, incline_height=0.1, last_incline_height=0.6, last_stone_len=1.6, pit_depth=[0.5, 1.]):
    goals = np.zeros((num_stones + 2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    mid_y = terrain.length // 2
    stone_len = np.random.uniform(*stone_len) if isinstance(stone_len, (list, tuple)) else stone_len
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height
    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)
    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)
    dis_x = platform_len - np.random.randint(max(dis_x_min, 1), max(dis_x_max, dis_x_min + 1)) + stone_len // 2
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    dis_z = 0
    for i in range(num_stones):
        dis_x += np.random.randint(max(dis_x_min, 1), max(dis_x_max, dis_x_min + 1))
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(max(dis_y_min, 0), max(dis_y_max, dis_y_min + 1))
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[max(0, dis_x-last_stone_len//2):min(terrain.width, dis_x+last_stone_len//2), max(0, dis_y-stone_width//2):min(terrain.length, dis_y+stone_width//2)] = heights.astype(int)[:max(0, min(terrain.width, dis_x+last_stone_len//2)-max(0, dis_x-last_stone_len//2)), :max(0, min(terrain.length, dis_y+stone_width//2)-max(0, dis_y-stone_width//2))] + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[max(0, dis_x-stone_len//2):min(terrain.width, dis_x+stone_len//2), max(0, dis_y-stone_width//2):min(terrain.length, dis_y+stone_width//2)] = heights.astype(int)[:max(0, min(terrain.width, dis_x+stone_len//2)-max(0, dis_x-stone_len//2)), :max(0, min(terrain.length, dis_y+stone_width//2)-max(0, dis_y-stone_width//2))] + dis_z
        goals[i + 1] = [min(dis_x, terrain.width - 1), np.clip(dis_y, 0, terrain.length - 1)]
        left_right_flag = 1 - left_right_flag
    final_dis_x = min(dis_x + 2 * np.random.randint(max(dis_x_min, 1), max(dis_x_max, dis_x_min + 1)), terrain.width - 1)
    final_platform_start = min(dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale), terrain.width)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    terrain.goals = goals * terrain.horizontal_scale
    _pad_parkour_edges(terrain, pad_width=pad_width, pad_height=pad_height)


def stl_heightmap_terrain(terrain, terrain_name='custom_stl', stl_heightmap_path='legged_gym/legged_gym/terrain_assets/heightmap.npy', goal_positions=None, pad_width=0.1, pad_height=0.0):
    scale_factor_height = 1.0
    scale_factor_width = 1.0
    try:
        heightmap = np.load(stl_heightmap_path)
        target_height = terrain.height_field_raw.shape[0]
        target_width = terrain.height_field_raw.shape[1]
        scale_factor_height = target_height / heightmap.shape[0] if heightmap.shape[0] > 0 else 1.0
        scale_factor_width = target_width / heightmap.shape[1] if heightmap.shape[1] > 0 else 1.0
        if heightmap.shape != (target_height, target_width):
            heightmap = ndimage.zoom(heightmap, (scale_factor_height, scale_factor_width), order=1)
        height_field_raw = (heightmap / terrain.vertical_scale).astype(np.int16)
        terrain_h, terrain_w = terrain.height_field_raw.shape
        height_field_raw = height_field_raw[:terrain_h, :terrain_w]
        if height_field_raw.shape[0] < terrain_h:
            height_field_raw = np.vstack([height_field_raw, np.zeros((terrain_h - height_field_raw.shape[0], height_field_raw.shape[1]), dtype=np.int16)])
        if height_field_raw.shape[1] < terrain_w:
            height_field_raw = np.hstack([height_field_raw, np.zeros((height_field_raw.shape[0], terrain_w - height_field_raw.shape[1]), dtype=np.int16)])
        terrain.height_field_raw[:, :] = height_field_raw
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f'Failed to load {terrain_name} heightmap {stl_heightmap_path}: {exc}') from exc
    _pad_parkour_edges(terrain, pad_width=pad_width, pad_height=pad_height)
    if goal_positions is not None:
        goals = np.asarray(goal_positions, dtype=np.float32)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        goals_scaled = goals.copy()
        goals_scaled[:, 0] *= scale_factor_height
        goals_scaled[:, 1] *= scale_factor_width
        terrain.goals = goals_scaled * terrain.horizontal_scale
    return terrain
