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

import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from .widowGo1_config import WidowGo1RoughCfg

class WidowGo1(LeggedRobot):
    cfg: WidowGo1RoughCfg

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12: 12 + self.num_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12 + self.num_dofs: 12 + 2 * self.num_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12 + 2 * self.num_dofs: 12 + 2 * self.num_dofs + self.num_actions] = 0. # previous actions
        curr_idx = 12 + 2 * self.num_dofs + self.num_actions
        if self.cfg.terrain.measure_heights:
            noise_vec[curr_idx: curr_idx + 187] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _parse_cfg(self, cfg):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.arm_reward_scales = class_to_dict(self.cfg.rewards.arm_scales)

        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.init_lin_vel_x_ranges = self.lin_vel_x_ranges = np.array(self.command_ranges['init_lin_vel_x'])
        self.init_ang_vel_yaw_ranges = self.ang_vel_yaw_ranges = np.array(self.command_ranges['init_ang_vel_yaw'])
        self.final_lin_vel_x_ranges = np.array(self.command_ranges['final_lin_vel_x'])
        self.final_ang_vel_yaw_ranges = np.array(self.command_ranges['final_ang_vel_yaw'])
        self.final_tracking_ang_vel_yaw_exp = self.command_ranges['final_tracking_ang_vel_yaw_exp']
        self.lin_vel_x_schedule = self.cfg.commands.lin_vel_x_schedule
        self.ang_vel_yaw_schedule = self.cfg.commands.ang_vel_yaw_schedule
        self.tracking_ang_vel_yaw_schedule = self.cfg.commands.tracking_ang_vel_yaw_schedule
        self.update_counter = 0
        
        
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
        self.init_goal_ee_l_ranges = self.goal_ee_l_ranges = np.array(self.goal_ee_ranges['init_pos_l'])
        self.init_goal_ee_p_ranges = self.goal_ee_p_ranges = np.array(self.goal_ee_ranges['init_pos_p'])
        self.init_goal_ee_y_ranges = self.goal_ee_y_ranges = np.array(self.goal_ee_ranges['init_pos_y'])
        self.final_goal_ee_l_ranges = np.array(self.goal_ee_ranges['final_pos_l'])
        self.final_goal_ee_p_ranges = np.array(self.goal_ee_ranges['final_pos_p'])
        self.final_goal_ee_y_ranges = np.array(self.goal_ee_ranges['final_pos_y'])
        # self.final_arm_action_scale = self.cfg.goal_ee.ranges.final_arm_action_scale
        self.final_tracking_ee_reward = self.cfg.goal_ee.ranges.final_tracking_ee_reward
        self.goal_ee_l_schedule = self.cfg.goal_ee.l_schedule
        self.goal_ee_p_schedule = self.cfg.goal_ee.p_schedule
        self.goal_ee_y_schedule = self.cfg.goal_ee.y_schedule
        # self.arm_action_scale_schedule = self.cfg.goal_ee.arm_action_scale_schedule
        self.tracking_ee_reward_schedule = self.cfg.goal_ee.tracking_ee_reward_schedule

        self.goal_ee_delta_orn_ranges = torch.tensor(self.goal_ee_ranges['final_delta_orn'])
        

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.action_delay = self.cfg.env.action_delay

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            # else:
            #     self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.arm_reward_scales.keys()):
            scale = self.arm_reward_scales[key]
            if scale==0:
                self.arm_reward_scales.pop(key) 
            # else:
            #     self.arm_reward_scales[key] *= self.dt
        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}

        self.metric_names = ['leg_energy_abs_sum', 'tracking_lin_vel_x_l1', 'tracking_ang_vel_yaw_exp', 'tracking_ee_cart', 'tracking_ee_sphere', 'tracking_ee_orn', 'leg_action_l2', 'torque', 'energy_square', 'foot_contacts_z']
        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) \
            for name in self.metric_names}


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.rew_buf /= 100

        self.arm_rew_buf[:] = 0.
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew = self.arm_reward_functions[i]() * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.arm_reward_scales:
            rew = self._reward_termination() * self.arm_reward_scales["termination"]
            self.arm_rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.arm_rew_buf /= 100
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots

        half_col_size = self.cfg.terrain.tot_cols * self.cfg.terrain.horizontal_scale / 2
        half_row_size = self.cfg.terrain.tot_rows * self.cfg.terrain.horizontal_scale / 2
        x_bounds = [- 2.5 * half_col_size / 5, - 2 * half_col_size / 5]
        y_bounds = [- half_row_size + 10, half_row_size - 10]
        print('origin x_bounds', x_bounds)
        print('origin y_bounds', y_bounds)
        self.env_origins[:, 0] = torch_rand_float(x_bounds[0], x_bounds[1], (self.num_envs, 1), device=self.device)[:, 0]
        self.env_origins[:, 1] = torch_rand_float(y_bounds[0], y_bounds[1], (self.num_envs, 1), device=self.device)[:, 0]
        self.env_origins[:, 2] = 0.

        self.box_env_origins_x = self.cfg.box.box_env_origins_x
        self.box_env_origins_delta_y = (torch_rand_sign((self.num_envs, 1), self.device) * \
            torch_rand_float(self.cfg.box.box_env_origins_y_range[0], self.cfg.box.box_env_origins_y_range[1], (self.num_envs, 1), device=self.device))[:, 0]
        self.box_env_origins_z = self.cfg.box.box_env_origins_z

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain_Perlin(self.cfg.terrain)
        self._create_trimesh()
        self._create_envs()
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = self.cfg.terrain.transform_x
        tm_params.transform.p.y = self.cfg.terrain.transform_y
        tm_params.transform.p.z = self.cfg.terrain.transform_z
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        # widowGo1
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-2]
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        # self.num_bodies = len(self.body_names)
        # self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            penalized_contact_names.extend(body_names)
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            termination_contact_names.extend(body_names)

        self.sensor_indices = []
        for name in feet_names:
            foot_idx = self.body_names_to_idx[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)
        
        self.gripper_idx = self.body_names_to_idx["wx250s/ee_gripper_link"]

        # box
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, self.cfg.box.box_size, self.cfg.box.box_size, self.cfg.box.box_size, asset_options)

        print('------------------------------------------------------')
        print('num_actions: {}'.format(self.num_actions))
        print('num_torques: {}'.format(self.num_torques))
        print('num_dofs: {}'.format(self.num_dofs))
        print('num_bodies: {}'.format(self.num_bodies))
        print('dof_names: ', self.dof_names)
        print('dof_names_to_idx: {}'.format(sorted(list(self.dof_names_to_idx.items()), key=lambda x: x[1])))
        print('body_names: {}'.format(self.body_names))
        print('body_names_to_idx: {}'.format(sorted(list(self.body_names_to_idx.items()), key=lambda x: x[1])))
        print('penalized_contact_names: {}'.format(penalized_contact_names))
        print('termination_contact_names: {}'.format(termination_contact_names))
        print('feet_names: {}'.format(feet_names))
        print(f"EE Gripper index: {self.gripper_idx}")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        box_start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.box_actor_handles = []
        box_body_indices = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # widowGo1 
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-self.cfg.terrain.origin_perturb_range, self.cfg.terrain.origin_perturb_range, (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_dog_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot_dog", i, self.cfg.asset.self_collisions, 0)
            self.actor_handles.append(robot_dog_handle)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_dog_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_dog_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_dog_handle, body_props, recomputeInertia=True)
            
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

            # box
            box_pos = pos.clone()
            box_pos[0] = self.box_env_origins_x
            box_pos[1] += self.box_env_origins_delta_y[i]
            box_pos[2] = self.box_env_origins_z
            box_start_pose.p = gymapi.Vec3(*box_pos)
            box_handle = self.gym.create_actor(env_handle, box_asset, box_start_pose, "box", i, self.cfg.asset.self_collisions, 0)
            self.box_actor_handles.append(box_handle)

            box_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            box_body_props, _ = self._box_process_rigid_body_props(box_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, box_body_props, recomputeInertia=True)

            box_body_idx = self.gym.get_actor_rigid_body_index(env_handle, box_handle, 0, gymapi.DOMAIN_SIM)
            box_body_indices.append(box_body_idx)
        
        assert(np.all(np.array(self.actor_handles) == 0))
        assert(np.all(np.array(self.box_actor_handles) == 1))
        assert(np.all(np.array(box_body_indices) % (self.num_bodies + 1) == self.num_bodies))
        self.robot_actor_indices = torch.arange(0, 2 * self.num_envs, 2, device=self.device)
        self.box_actor_indices = torch.arange(1, 2 * self.num_envs, 2, device=self.device)

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat([
                    torch_rand_float(self.cfg.domain_rand.leg_motor_strength_range[0], self.cfg.domain_rand.leg_motor_strength_range[1], (self.num_envs, 12), device=self.device),
                    torch_rand_float(self.cfg.domain_rand.arm_motor_strength_range[0], self.cfg.domain_rand.arm_motor_strength_range[1], (self.num_envs, 6), device=self.device)
                ], dim=1)
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)
        
        # if self.cfg.domain_rand.randomize_arm_ema:
        #     self.arm_ema = torch_rand_float(self.cfg.domain_rand.arm_ema_range[0], self.cfg.domain_rand.arm_ema_range[1], (self.num_envs, 6), device=self.device)
        # else:
        #     self.arm_ema = torch.zeros(self.num_envs, 6, device=self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        print('penalized_contact_indices: {}'.format(self.penalized_contact_indices))
        print('termination_contact_indices: {}'.format(self.termination_contact_indices))
        print('feet_indices: {}'.format(self.feet_indices))

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)

        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass])
        return props, mass_params
    
    def _box_process_rigid_body_props(self, props, env_id):
        if self.cfg.box.randomize_base_mass:
            rng_mass = self.cfg.box.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        return props, rand_mass

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 1000
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs, 1, 1)) 
        
        return props

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)
        # self.old_arm_actions = torch.zeros((self.num_envs, 6), device=self.device)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "robot_dog")
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13) # 2 actors
        self.root_states = self._root_states[:, 0, :]
        self.box_root_state = self._root_states[:, 1, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_pos_wrapped = self.dof_pos.clone()
        self.dof_pos_wo_gripper = self.dof_pos[:, :-2]
        self.dof_pos_wo_gripper_wrapped = self.dof_pos_wo_gripper.clone()
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-2]
        self.base_quat = self.root_states[:, 3:7]
        # self.yaw_ema = euler_from_quat(self.base_quat)[2]
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies + 1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = self._contact_forces[:, :-1, :]
        self.box_contact_force = self._contact_forces[:, -1, :]

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_state = self._rigid_body_state[:, :-1, :]
        self.box_rigid_body_state = self._rigid_body_state[:, -1, :]

        self.mm_whole = gymtorch.wrap_tensor(mass_matrix_tensor)
        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)

        # ee info
        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -8:-2]
        self.mm = self.mm_whole[:, -8:-2, -8:-2]
        self.arm_osc_kp = torch.tensor(self.cfg.arm.osc_kp, device=self.device, dtype=torch.float)
        self.arm_osc_kd = torch.tensor(self.cfg.arm.osc_kd, device=self.device, dtype=torch.float)

        # box info & target_ee info
        self.box_pos = self.box_root_state[:, 0:3]
        self.down_dir = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float).view(3, 1)
        self.ee_orn_des = torch.tensor([ 0, 0.7071068, 0, 0.7071068 ], device=self.device).repeat((self.num_envs, 1))
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        ## generate target ee
        # self.update_target_ee_base()
        # self.target_ee_base_clipped = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # self.target_ee_base_clipped[:, 0] = 0.15
        # self.target_ee_base_clipped[:, 2] = 0.15
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_delta_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        if self.cfg.goal_ee.command_mode == 'cart':
            self.curr_ee_goal = self.curr_ee_goal_cart
        else:
            self.curr_ee_goal = self.curr_ee_goal_sphere
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.arm_base_overhead = torch.tensor([0., 0., 0.165], device=self.device)
        self.z_invariant_offset = torch.tensor([0.53], device=self.device).repeat(self.num_envs, 1)

        print('------------------------------------------------------')
        print(f'root_states shape: {self.root_states.shape}')
        print(f'dof_state shape: {self.dof_state.shape}')
        print(f'force_sensor_tensor shape: {self.force_sensor_tensor.shape}')
        print(f'contact_forces shape: {self.contact_forces.shape}')
        print(f'rigid_body_state shape: {self.rigid_body_state.shape}')
        print(f'mm_whole shape: {self.mm_whole.shape}')
        print(f'jacobian_whole shape: {self.jacobian_whole.shape}')
        print(f'box_root_state shape: {self.box_root_state.shape}')
        print(f'box_contact_force shape: {self.box_contact_force.shape}')
        print(f'box_rigid_body_state shape: {self.box_rigid_body_state.shape}')
        print('------------------------------------------------------')
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.extras["episode"] = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        # self.target_ee = torch.zeros(self.num_envs, self.cfg.target_ee.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # ee x, ee y, ee z
        self.gripper_torques_zero = torch.zeros(self.num_envs, 2, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        
        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    raise Exception(f"PD gain of joint {name} were not defined, setting them to zero")
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-2]

        link_mass = torch.zeros(1, 9, dtype=torch.float, device=self.device)
        for j, prop in enumerate(self.gym.get_actor_rigid_body_properties(self.envs[0], 0)[-9:]):
            link_mass[0, j] = prop.mass
        self.link_mass = link_mass.repeat((self.num_envs, 1))
        g = torch.zeros(self.num_envs, 9, 6, 1, dtype=torch.float, device=self.device)
        g[:, :, 2, :] = 9.81
        self.g_force = self.link_mass.unsqueeze(-1).unsqueeze(-1) * g

        self._get_init_start_ee_sphere()
    
    
    def _get_curriculum_value(self, schedule, init_range, final_range, counter):
        return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range
    
    def update_command_curriculum(self):
        self.update_counter += 1

        self.lin_vel_x_ranges = self._get_curriculum_value(self.lin_vel_x_schedule, self.init_lin_vel_x_ranges, self.final_lin_vel_x_ranges, self.update_counter)
        self.ang_vel_yaw_ranges = self._get_curriculum_value(self.ang_vel_yaw_schedule, self.init_ang_vel_yaw_ranges, self.final_ang_vel_yaw_ranges, self.update_counter)
        self.reward_scales['tracking_ang_vel_yaw_exp'] = self._get_curriculum_value(self.tracking_ang_vel_yaw_schedule, 0, self.final_tracking_ang_vel_yaw_exp, self.update_counter)

        self.goal_ee_l_ranges = self._get_curriculum_value(self.goal_ee_l_schedule, self.init_goal_ee_l_ranges, self.final_goal_ee_l_ranges, self.update_counter)
        self.goal_ee_p_ranges = self._get_curriculum_value(self.goal_ee_p_schedule, self.init_goal_ee_p_ranges, self.final_goal_ee_p_ranges, self.update_counter)
        self.goal_ee_y_ranges = self._get_curriculum_value(self.goal_ee_y_schedule, self.init_goal_ee_y_ranges, self.final_goal_ee_y_ranges, self.update_counter)
        # self.action_scale[-6:] = self._get_curriculum_value(self.arm_action_scale_schedule, 0, self.final_arm_action_scale, self.update_counter)
        if 'tracking_ee_sphere' in self.arm_reward_scales:
            self.arm_reward_scales['tracking_ee_sphere'] = self._get_curriculum_value(self.tracking_ee_reward_schedule, 0, self.final_tracking_ee_reward, self.update_counter)
        else:
            self.arm_reward_scales['tracking_ee_cart'] = self._get_curriculum_value(self.tracking_ee_reward_schedule, 0, self.final_tracking_ee_reward, self.update_counter)


    def reset_idx(self, env_ids, start=False):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        ## fetch latest simulation results for ee generator
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        if start:
            command_env_ids = env_ids
        else:
            command_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        self._resample_commands(command_env_ids)
        # self._resample_target_ee(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.goal_timer[env_ids] = 0.
        # self.old_arm_actions[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        for key in self.episode_metric_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_metric_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_metric_sums[key][env_ids] = 0.

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-self.cfg.terrain.origin_perturb_range, self.cfg.terrain.origin_perturb_range, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    
        self.box_root_state[env_ids, 0] = self.box_env_origins_x
        self.box_root_state[env_ids, 1] = self.root_states[env_ids, 1] + self.box_env_origins_delta_y[env_ids]
        self.box_root_state[env_ids, 2] = self.box_env_origins_z
        
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-self.cfg.terrain.init_vel_perturb_range, self.cfg.terrain.init_vel_perturb_range, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        
        # print('')

        # print('-----------------')
        # print('delta_y: ', self.box_env_origins_delta_y)
        # print(env_ids)
        # print('-----------------')
        # print('(before) dog root state: ')
        # print(self.root_states[:, :3])
        # print('(before) box root state: ')
        # print(self.box_root_state[:, :3])

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # print('-----------------')
        # print('dog root state: ')
        # print(self.root_states[:, :3])
        # print('box root state: ')
        # print(self.box_root_state[:, :3])

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self._root_states.clone()),
        #                                              gymtorch.unwrap_tensor(self.robot_actor_indices), self.num_envs)

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.box_root_state.clone()),
        #                                              gymtorch.unwrap_tensor(self.box_actor_indices), self.num_envs)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            self.root_states[:, 7:9] * 2.5,
            self.root_states[:, 7:9]
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.lin_vel_x_ranges[0], self.lin_vel_x_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(self.ang_vel_yaw_ranges[0], self.ang_vel_yaw_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :] *= (torch.logical_or(self.commands[env_ids, 0] > self.cfg.commands.lin_vel_x_clip, \
            torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)).unsqueeze(1)

    # def _resample_target_ee(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     target_ee_len = torch_rand_float(self.target_ee_ranges['pos_l'][0], self.target_ee_ranges['pos_l'][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     target_ee_pitch = torch_rand_float(self.target_ee_ranges['pos_p'][0], self.target_ee_ranges['pos_p'][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     target_ee_yaw = torch_rand_float(self.target_ee_ranges['pos_y'][0], self.target_ee_ranges['pos_y'][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     pitch_sin = torch.sin(target_ee_pitch)
    #     pitch_cos = torch.cos(target_ee_pitch)
    #     yaw_sin = torch.sin(target_ee_yaw)
    #     yaw_cos = torch.cos(target_ee_yaw)

    #     proj_len = target_ee_len * pitch_cos
    #     self.target_ee[env_ids, 0] = proj_len * yaw_cos
    #     self.target_ee[env_ids, 1] = proj_len * yaw_sin
    #     self.target_ee[env_ids, 2] = target_ee_len * pitch_sin

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # update box obs
        # self.update_target_ee_base()

        # update ee goal
        self.update_curr_ee_goal()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids, start=False)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.extras["episode"]['coeff_lin_vel_x_upper_bound'] = self.lin_vel_x_ranges[1]
        self.extras["episode"]['coeff_lin_vel_x_lower_bound'] = self.lin_vel_x_ranges[0]
        self.extras["episode"]['coeff_ang_vel_yaw_upper_bound'] = self.ang_vel_yaw_ranges[1]
        self.extras["episode"]['coeff_ang_vel_yaw_lower_bound'] = self.ang_vel_yaw_ranges[0] 
        self.extras["episode"]['coeff_tracking_ang_vel_yaw_exp'] = self.reward_scales['tracking_ang_vel_yaw_exp']

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_debug_vis()
            self._draw_ee_goal()
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        command_env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # target_ee_env_ids = (self.episode_length_buf % int(self.cfg.target_ee.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

        self._resample_commands(command_env_ids)
        # self._resample_target_ee(target_ee_env_ids)
        # if self.cfg.commands.heading_command:
        #     forward = quat_apply(self.base_quat, self.forward_vec)
        #     heading = torch.atan2(forward[:, 1], forward[:, 0])
        #     self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        r, p, _ = euler_from_quat(self.base_quat) 
        z = self.root_states[:, 2]

        r_threshold_buff = ((r > 0.2) & (self.curr_ee_goal[:, 2] >= 0)) | ((r < -0.2) & (self.curr_ee_goal[:, 2] <= 0))
        p_threshold_buff = ((p > 0.2) & (self.curr_ee_goal[:, 1] >= 0)) | ((p < -0.2) & (self.curr_ee_goal[:, 1] <= 0))
        z_threshold_buff = z < self.cfg.termination.z_threshold
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        # print(self.base_quat)
        # print(torch.stack([self.reset_buf, r_threshold_buff, p_threshold_buff, z_threshold_buff], dim=-1)[0])
        # print('r: ', r[0].item())
        # print('p: ', p[0].item())
        # print('z: ', z[0].item())
        # print('-----------------------------------------------------')
        # time.sleep(0.5)

        # self.reset_triggers = torch.stack([termination_contact_buf, r_threshold_buff, p_threshold_buff, z_threshold_buff, self.time_out_buf], dim=-1).nonzero(as_tuple=False)
        # if len(self.reset_triggers) > 0:
        #     print('reset_triggers: ', self.reset_triggers)

        self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | self.time_out_buf
    

    def compute_observations(self):
        """ Computes observations
        """
        self.dof_pos_wrapped[:] = self.dof_pos
        self.dof_pos_wrapped[:, -8] = torch_wrap_to_pi_minuspi(self.dof_pos_wrapped[:, -8])
        # self.dof_pos_wrapped[:, 12:] = 0
        # self.dof_vel[:, 12:] = 0
        obs_buf = torch.cat((       self.get_body_orientation(),  # dim 2
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    self.ig2raisim((self.dof_pos_wrapped - self.default_dof_pos) * self.obs_scales.dof_pos),  # dim 20
                                    self.ig2raisim(self.dof_vel * self.obs_scales.dof_vel),  # dim 20
                                    self.ig2raisim_wo_gripper(self.action_history_buf[:, -1]),  # dim 18
                                    self.ig2raisim_feet(self.get_foot_contacts()),  # dim 4
                                    self.commands[:, :3] * self.commands_scale,  # dim 3
                                    self.curr_ee_goal,  # dim 3
                                    self.ee_goal_delta_orn_euler  # dim 3
                                    ),dim=-1)
        # g_torques = self.get_g_torques()
        # arm_mm = self.get_arm_mm().reshape(self.num_envs, -1)
        # ee_jac = self.get_ee_jac().reshape(self.num_envs, -1)
        if self.cfg.domain_rand.observe_priv:
            priv_buf = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength - 1
            ), dim=-1)
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )        

    def ig2raisim_feet(self, vec):
        # raisim_order = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        # ig_order = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        if self.cfg.env.reorder_dofs:
            return vec[:, [1, 0, 3, 2]]
        return vec

    def ig2raisim(self, vec):
        if self.cfg.env.reorder_dofs:
            # Need to reorder DOFS to match what the a1 hardware gives -(FR (hip, thigh, calf), FL, RR, RL)
            if not hasattr(self, "ig_2_raisim_reordering_idx"):
                self.ig_2_raisim_reordering_idx = []
                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                                    'widow_waist', 'widow_shoulder', 'widow_elbow', 
                                    'widow_forearm_roll', 'widow_wrist_angle', 'widow_wrist_rotate', 
                                    'widow_left_finger', 'widow_right_finger']

                for name in dof_order_a1_robot:
                    self.ig_2_raisim_reordering_idx.append(self.dof_names.index(name))

            vec = vec[:, self.ig_2_raisim_reordering_idx]

        return vec

    def raisim2ig(self, vec):
        # print(self.cfg.env.reorder_dofs)
        if self.cfg.env.reorder_dofs:
            if not hasattr(self, "raisim2ig_reordering_idx"):
                self.raisim2ig_reordering_idx = []

                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                    'widow_waist', 'widow_shoulder', 'widow_elbow', 
                                    'widow_forearm_roll', 'widow_wrist_angle', 'widow_wrist_rotate', 
                                    'widow_left_finger', 'widow_right_finger']

                for name in self.dof_names:
                    self.raisim2ig_reordering_idx.append(dof_order_a1_robot.index(name))

            vec = vec[:, self.raisim2ig_reordering_idx]
        
        return vec
    
    def ig2raisim_wo_gripper(self, vec):
        if self.cfg.env.reorder_dofs:
            # Need to reorder DOFS to match what the a1 hardware gives -(FR (hip, thigh, calf), FL, RR, RL)
            if not hasattr(self, "ig_2_raisim_wo_gripper_reordering_idx"):
                self.ig_2_raisim_wo_gripper_reordering_idx = []
                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                                    'widow_waist', 'widow_shoulder', 'widow_elbow', 
                                    'widow_forearm_roll', 'widow_wrist_angle', 'widow_wrist_rotate']

                for name in dof_order_a1_robot:
                    self.ig_2_raisim_wo_gripper_reordering_idx.append(self.dof_names.index(name))

            vec = vec[:, self.ig_2_raisim_wo_gripper_reordering_idx]

        return vec
    
    def raisim2ig_wo_gripper(self, vec):
        # print(self.cfg.env.reorder_dofs)
        if self.cfg.env.reorder_dofs:
            if not hasattr(self, "raisim2ig_wo_gripper_reordering_idx"):
                self.raisim2ig_wo_gripper_reordering_idx = []

                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                    'widow_waist', 'widow_shoulder', 'widow_elbow', 
                                    'widow_forearm_roll', 'widow_wrist_angle', 'widow_wrist_rotate']

                for name in self.dof_wo_gripper_names:
                    self.raisim2ig_wo_gripper_reordering_idx.append(dof_order_a1_robot.index(name))

            vec = vec[:, self.raisim2ig_wo_gripper_reordering_idx]
        
        return vec

    def get_foot_contacts(self):
        # contacts = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 0.1
        # self.contact_filts = torch.logical_or(contacts, self.last_contacts) 
        # self.last_contacts = contacts
        # return self.contact_filts 
        self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 1.5
        # print('force: ', self.force_sensor_tensor.norm(dim=-1))

        return self.foot_contacts_from_sensor

    
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        transformed_target_ee = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        upper_arm_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_state[:, self.gripper_idx, :3]

        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)
        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(transformed_target_ee[i, 0], transformed_target_ee[i, 1], transformed_target_ee[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

    def _draw_ee_goal(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze()
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)[:, :, None]
        # curr_ee_goal_cart_world = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) + self.root_states[:, :3]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
            # pose_curr = gymapi.Transform(gymapi.Vec3(curr_ee_goal_cart_world[i, 0], curr_ee_goal_cart_world[i, 1], curr_ee_goal_cart_world[i, 2]), r=None)
            # gymutil.draw_lines(sphere_geom_yellow, self.gym, self.viewer, self.envs[i], pose_curr)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.raisim2ig_wo_gripper(actions)
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
            actions = self.action_history_buf[:, -self.action_delay - 1] # delay for 1/50=20ms

        # self.old_arm_actions = self.arm_ema * actions[:, -6:] + (1 - self.arm_ema) * self.old_arm_actions[:, -6:]
        
        # self.actions = torch.cat([actions[:, :-6], self.old_arm_actions], dim=1) 
        self.actions = actions.clone()

        for t in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            # print(self.torques[0, :12].reshape(4,3).abs().sum(dim=-1))
            if self.cfg.control.torque_supervision and t == 0:
                self.torques_t0 = self.torques
                self.extras['target_arm_torques'] = self.arm_ee_control_torques
                self.extras['current_arm_dof_pos'] = self.dof_pos[:, -8: -2]
                self.extras['current_arm_dof_vel'] = self.dof_vel[:, -8: -2]
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if self.cfg.control.torque_supervision:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_mass_matrix_tensors(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras
    
    def get_g_torques(self):
        self.gym.refresh_jacobian_tensors(self.sim)

        g_torque = (torch.transpose(self.jacobian_whole[:, -9:, :, -8:], 2, 3) @ self.g_force).squeeze(-1)           # new shape is (n_envs, n_bodies, n_links)
        g_torque = torch.sum(g_torque, dim=1, keepdim=False)[:, :6]

        return g_torque
    
    def get_arm_mm(self):
        self.gym.refresh_mass_matrix_tensors(self.sim)
        return self.mm
    
    def get_ee_jac(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        return self.ee_j_eef
    
    def get_arm_ee_control_torques(self):
        # Solve for control (Operational Space Control)
        m_inv = torch.pinverse(self.mm)

        m_eef = torch.pinverse(self.ee_j_eef @ m_inv @ torch.transpose(self.ee_j_eef, 1, 2))
        ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        orn_err = orientation_error(self.ee_orn_des, ee_orn_normalized)

        pos_err = (torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) - self.ee_pos)
        # pos_err = quat_rotate_inverse(self.base_quat, pos_err)

        dpose = torch.cat([pos_err, orn_err], -1)

        g_torque = (torch.transpose(self.jacobian_whole[:, -9:, :, -8:], 2, 3) @ self.g_force).squeeze(-1)           # new shape is (n_envs, n_bodies, n_links)
        g_torque = torch.sum(g_torque, dim=1, keepdim=False)[:, :6]

        u = (torch.transpose(self.ee_j_eef, 1, 2) @ m_eef @ (self.arm_osc_kp * dpose - self.arm_osc_kd * self.ee_vel)[:, :6].unsqueeze(-1)).squeeze(-1)

        u += g_torque

        # damping = 0.05
        # j_eef_T = torch.transpose(j_eef, 1, 2)
        # lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        # u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).squeeze(-1)

        return u 
    
    # def update_target_ee_base(self):
    #     to_box = self.box_pos - self.ee_pos
    #     box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    #     box_dir = to_box / box_dist
    #     box_dot = box_dir @ self.down_dir

    #     above_box = ((box_dot >= 0.7) & (box_dist < self.grasp_offset * 3)).squeeze(-1)
    #     target_ee = self.box_pos.clone()
    #     target_ee[:, 2] = torch.where(above_box, self.box_pos[:, 2], self.box_pos[:, 2] + self.grasp_offset * 2)
    #     target_ee_base = quat_rotate_inverse(self.base_quat, target_ee)

    #     self.target_ee_base_clipped = torch.where(
    #         (self.box_pos - self.root_states[:, 0:3]).norm(dim=-1, keepdim=True) < self.cfg.box.box_pos_obs_range,
    #         target_ee_base, 
    #         self.init_target_ee_base
    #     )


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        if self.cfg.control.adaptive_arm_gains:
            actions, delta_arm_gains = actions[:, :12+6], actions[:, 12+6:]
        actions_scaled = actions * self.motor_strength * self.action_scale

        self.dof_pos_wo_gripper_wrapped[:] = self.dof_pos_wo_gripper
        self.dof_pos_wo_gripper_wrapped[:, -8] = torch_wrap_to_pi_minuspi(self.dof_pos_wo_gripper_wrapped[:, -8])

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper_wrapped) - self.d_gains * self.dof_vel_wo_gripper

        if self.cfg.control.adaptive_arm_gains:
            leg_torques = default_torques[:, :12]
            adaptive_arm_p_gains = self.p_gains[12:] + delta_arm_gains
            adaptive_arm_d_gains = 2 * (adaptive_arm_p_gains ** 0.5)
            arm_torques = adaptive_arm_p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper_wrapped)[:, 12:] - adaptive_arm_d_gains * self.dof_vel_wo_gripper[:, 12:]
            torques = torch.cat([leg_torques, arm_torques, self.gripper_torques_zero], dim=-1)
        else:
            torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        
        if self.cfg.control.torque_supervision:
            self.arm_ee_control_torques = self.get_arm_ee_control_torques()
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _get_init_start_ee_sphere(self):
        init_start_ee_cart = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        init_start_ee_cart[:, 0] = 0.15
        init_start_ee_cart[:, 2] = 0.15
        self.init_start_ee_sphere = cart2sphere(init_start_ee_cart)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_l_ranges[0], self.goal_ee_l_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_p_ranges[0], self.goal_ee_p_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_y_ranges[0], self.goal_ee_y_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_delta_orn_ranges[0, 0], self.goal_ee_delta_orn_ranges[0, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_delta_orn_ranges[1, 0], self.goal_ee_delta_orn_ranges[1, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_delta_orn_ranges[2, 0], self.goal_ee_delta_orn_ranges[2, 1], (len(env_ids), 1), device=self.device)
        self.ee_goal_delta_orn_euler[env_ids] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
        self.ee_goal_orn_euler[env_ids] = torch_wrap_to_pi_minuspi(self.ee_goal_delta_orn_euler[env_ids] + self.base_yaw_euler[env_ids])
        # self.ee_goal_orn_euler[env_ids] = torch.tensor([ 0, np.pi / 2, 0], dtype=torch.float, device=self.device)

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            self._resample_ee_goal_orn_once(env_ids)
            # if is_init:
            #     self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[env_ids].clone()
            #     self._resample_ee_goal_sphere_once(env_ids, self.cfg.goal_ee.init_ranges)
            # else:
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self.collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

        # self.ee_goal_cart[env_ids, :] = sphere2cart(self.ee_goal_sphere[env_ids, :])
        # self.goal_timer[env_ids] = 0.0

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))

        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.curr_ee_goal_sphere) * self.sphere_error_scale, dim=1)

        self.episode_metric_sums['tracking_ee_sphere'] += ee_pos_error
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_cart(self):
        target_ee = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        ee_pos_error = torch.sum(torch.abs(self.ee_pos - target_ee), dim=1)

        self.episode_metric_sums['tracking_ee_cart'] += ee_pos_error
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)
    
    def _reward_tracking_ee_orn(self):
        # ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        # ee_goal_orn_normalized = self.ee_goal_orn_quat / torch.norm(self.ee_goal_orn_quat, dim=-1).unsqueeze(-1)
        # orn_err = orientation_error(ee_goal_orn_normalized, ee_orn_normalized)
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)

        # self.episode_metric_sums['tracking_ee_orn'] += orn_err

        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        self.episode_metric_sums['leg_action_l2'] += action_l2
        return action_l2

    def _reward_tracking_ee_orn_ry(self):
        # ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        # ee_goal_orn_normalized = self.ee_goal_orn_quat / torch.norm(self.ee_goal_orn_quat, dim=-1).unsqueeze(-1)
        # orn_err = orientation_error(ee_goal_orn_normalized, ee_orn_normalized)
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler) * self.orn_error_scale)[:, [0, 2]]), dim=1)

        self.episode_metric_sums['tracking_ee_orn'] += orn_err

        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)


    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.torques[:, :12] * self.dof_vel[:, :12]), dim = 1)
        self.episode_metric_sums['leg_energy_abs_sum'] += energy
        return energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.torques[:, :12] * self.dof_vel[:, :12], dim = 1))
        return energy
    
    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.actions[:, :12] ** 2, dim=1)
        self.episode_metric_sums['leg_action_l2'] += action_l2
        return action_l2
    
    def _reward_leg_energy(self):
        energy = torch.sum(self.torques[:, :12] * self.dof_vel[:, :12], dim = 1)
        return energy
    
    def _reward_arm_energy_abs_sum(self):
        return torch.sum(torch.abs(self.torques[:, 12:-2] * self.dof_vel[:, 12:-2]), dim = 1)
    
    # def _reward_arm_orientation(self):
    #     # ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
    #     ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
    #     return - orientation_error(self.ee_orn_des, ee_orn_normalized).norm(dim=-1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_x_l1(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.episode_metric_sums['tracking_lin_vel_x_l1'] += error
        return - error + torch.abs(self.commands[:, 0])

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.episode_metric_sums['tracking_lin_vel_x_l1'] += error
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return - error + torch.abs(self.commands[:, 2])
    
    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.episode_metric_sums['tracking_ang_vel_yaw_exp'] += error
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y_l2(self):
        return (self.commands[:, 1] - self.base_lin_vel[:, 1]) ** 2
    
    def _reward_tracking_lin_vel_z_l2(self):
        return (self.commands[:, 2] - self.base_lin_vel[:, 2]) ** 2
    
    def _reward_survive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        self.episode_metric_sums['foot_contacts_z'] += foot_contacts_z
        return foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.torques), dim=1)
        self.episode_metric_sums['torque'] += torque
        return torque
    
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.torques[:, :12] * self.dof_vel[:, :12]), dim=1)
        self.episode_metric_sums['energy_square'] += energy
        return energy
    