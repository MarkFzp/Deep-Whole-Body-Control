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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import torch

RESUME = True

class WidowGo1RoughCfg( LeggedRobotCfg ):
    # class target_ee:
    #     num_commands = 3
    #     resampling_time = 1.5 # time before command are changed[s]
    #     # heading_command = True # if true: compute ang vel command from heading error
    #     class ranges:
    #         pos_l = [0.4, 0.6] # min max [m/s]
    #         pos_p = [0, np.pi / 2]   # min max [m/s]
    #         pos_y = [0, 2 * np.pi]    # min max [rad/s]

    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'sphere'

        l_schedule = [0, 1]
        p_schedule = [0, 1]
        y_schedule = [0, 1]
        # arm_action_scale_schedule = [0, 1]
        tracking_ee_reward_schedule = [0, 1]
        
        class ranges:
            final_pos_l = [0.2, 0.7] # min max [m/s]
            final_pos_p = [- 2 * np.pi / 5, 1 * np.pi / 5]   # min max [m/s]
            final_pos_y = [- 3 * np.pi / 5, 3 * np.pi / 5]    # min max [rad/s]
            init_pos_l = [0.6, 0.6]
            init_pos_p = [1 * np.pi / 4, 1 * np.pi / 4]
            init_pos_y = [-1 * np.pi / 6, 1 * np.pi / 6]

            final_delta_orn = [[-0, 0], [-0, 0], [-0, 0]]

            # final_arm_action_scale = 1.7
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [2 / np.pi, 2 / np.pi, 2 / np.pi]
            
        class init_ranges:
            pos_l = [0.3, 0.5] # min max [m/s]
            pos_p = [np.pi / 4, 3 * np.pi / 4]   # min max [m/s]
            pos_y = [0, 0]    # min max [rad/s]
    
    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3. # time before command are changed[s]

        lin_vel_x_schedule = [0, 1]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.6
        lin_vel_x_clip = 0.3

        class ranges:
            final_lin_vel_x = [0, 0.9] # min max [m/s]
            # lin_vel_y = [0, 0]   # min max [m/s]
            final_ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            init_lin_vel_x = [0, 0]
            init_ang_vel_yaw = [0, 0]
            
            final_tracking_ang_vel_yaw_exp = 0.15


    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel =  1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 5000
        num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = 2  # -1 for no delay
        # num_dofs = 19
        num_proprio = 2 + 3 + 20 + 20 + 18 + 4 + 3 + 3 + 3 
        num_priv = 5 + 1 + 18
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

        reorder_dofs = True

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.1,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.1 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.1,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'widow_waist': 0,
            'widow_shoulder': 0,
            'widow_elbow': 0,
            'widow_wrist_angle': 0,
            'widow_forearm_roll': 0,
            'widow_wrist_rotate': 0,
            'widow_left_finger': 0,
            'widow_right_finger': 0
        }

    class control:
        # PD Drive parameters:
        # Kp = [ 5.1876, 5.1876, 3.4584, 0.1729, 1.7292, 0.1729]
        # Kd = [ 0.4323, 0.4323, 0.0865, 0,      0.0864, 0]
        stiffness = {'joint': 50, 'widow': 5}  # [N*m/rad]
        damping = {'joint': 1, 'widow': 0.5}     # [N*m*s/rad]
        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/widowGo1/urdf/widowGo1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "trunk"]
        terminate_after_contacts_on = [] # ["wx250", "base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
        box_pos_obs_range = 1.0
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [-0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.5]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]

        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        # randomize_arm_friction = True
        # arm_friction_range = [0.0, 0.2]
        # randomize_arm_ema = True
        # arm_ema_range = [0.05, 0.25]

        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 0.5

        cube_y_range = [0.2, 0.4]
        
    
    class noise( LeggedRobotCfg.noise ):
        add_noise = False
  
    class rewards:
        class scales:
            termination = -0
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0.
            ang_vel_xy = -0.
            orientation = -0.
            torques = 0
            energy_square = -6e-5
            dof_vel = 0
            dof_acc = -0
            base_height = 0
            feet_air_time =  0
            collision = 0
            feet_stumble = -0 
            action_rate = -0
            stand_still = 0
            survive = 0.2
            leg_energy = -0
            leg_energy_abs_sum = -0
            tracking_lin_vel_x_l1 = 0.5
            tracking_lin_vel_x_exp = 0.
            tracking_ang_vel_yaw_l1 = 0  #check final_tracking_ang_vel_yaw_l1
            tracking_ang_vel_yaw_exp = 0.15  #check final_tracking_ang_vel_yaw_exp
            tracking_lin_vel_y_l2 = 0
            tracking_lin_vel_z_l2 = -0.0
            leg_action_l2 = -0.0
            hip_action_l2 = -0.01
            foot_contacts_z = -1e-4
        class arm_scales:
            termination = -0.0
            tracking_ee_sphere = 0.55
            tracking_ee_cart = 0.0
            arm_orientation = -0.
            arm_energy_abs_sum = -0.0040
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = 0.
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 1 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    
    class termination:
        r_threshold = 0.78
        p_threshold = 0.60
        z_threshold = 0.325

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        add_slopes = True
        slope_incline = 0.2
        horizontal_scale = 0.025 # [m]
        vertical_scale = 1 / 100000 # [m]
        border_size = 0 # [m]
        tot_cols = 600
        tot_rows = 10000
        zScale = 0.15
        transform_x = - tot_cols * horizontal_scale / 2
        transform_y = - tot_rows * horizontal_scale / 2
        transform_z = 0.0

        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # trimesh only:
        slope_treshold = 100000000 # slopes above this threshold will be corrected to vertical surfaces

        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1


class WidowGo1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = WidowGo1RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 #1.e-3 #5.e-4
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule=[1.0, 0, 3000] if not RESUME else [1.0, 0, 1]
        torque_supervision = WidowGo1RoughCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule=[0.0, 1000, 1000]
        adaptive_arm_gains = WidowGo1RoughCfg.control.adaptive_arm_gains

        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 40
        max_iterations = 40000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'rough_widowGo1'
        run_name = ''
        # load and resume
        resume = RESUME
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

