

import os, sys
import torch
sys.path.append("../../../rsl_rl")
# from rsl_rl.modules.actor_critic import get_activation
import sys
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
            def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                num_leg_actions, num_arm_actions, adaptive_arm_gains, adaptive_arm_gains_scale,
                num_priv, num_hist, num_prop, priv_encoder_dims):
                super().__init__()
                self.adaptive_arm_gains = adaptive_arm_gains
                self.adaptive_arm_gains_scale = adaptive_arm_gains_scale
                self.num_arm_actions = num_arm_actions

                # Policy
                if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        # if l == len(priv_encoder_dims) - 1:
                        #     priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_actions))
                        #     # priv_encoder_layers.append(nn.Tanh())
                        # else:
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
                else:
                    self.priv_encoder = nn.Identity()
                    priv_encoder_output_dim = num_priv

                self.num_priv = num_priv
                self.num_hist = num_hist
                self.num_prop = num_prop

                # Priv Encoder
                # encoder_dim = 8
                # self.priv_encoder =  nn.Sequential(*[
                #                         nn.Linear(num_priv, 256), activation,
                #                         nn.Linear(256, 128), activation,
                #                         nn.Linear(128, encoder_dim), 
                #                         # nn.Tanh()
                #                         nn.LeakyReLU()
                #                     ])
                
                self.history_encoder = StateHistoryEncoder(activation, mlp_input_dim_a, num_hist, priv_encoder_output_dim)

                # Policy
                if len(actor_hidden_dims) > 0:
                    actor_layers = []
                    actor_layers.append(nn.Linear(mlp_input_dim_a + priv_encoder_output_dim, actor_hidden_dims[0]))
                    actor_layers.append(activation)
                    for l in range(len(actor_hidden_dims) - 1):
                        # if l == len(actor_hidden_dims) - 1:
                        #     actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                        #     # actor_layers.append(nn.Tanh())
                        # else:
                        actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                        actor_layers.append(activation)
                    self.actor_backbone = nn.Sequential(*actor_layers)
                    actor_backbone_output_dim = actor_hidden_dims[-1]
                else:
                    self.actor_backbone = nn.Identity()
                    actor_backbone_output_dim = mlp_input_dim_a + priv_encoder_output_dim

                actor_leg_layers = []
                actor_leg_layers.append(nn.Linear(actor_backbone_output_dim, leg_control_head_hidden_dims[0]))
                actor_leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], num_leg_actions))
                        actor_leg_layers.append(nn.Tanh())
                    else:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                        actor_leg_layers.append(activation)
                self.actor_leg_control_head = nn.Sequential(*actor_leg_layers)

                actor_arm_layers = []
                actor_arm_layers.append(nn.Linear(actor_backbone_output_dim, arm_control_head_hidden_dims[0]))
                actor_arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], num_arm_actions))
                        actor_arm_layers.append(nn.Tanh())
                    else:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                        actor_arm_layers.append(activation)
                self.actor_arm_control_head = nn.Sequential(*actor_arm_layers)
            
            def forward(self, obs_prop_and_latent):
                backbone_input = obs_prop_and_latent
                backbone_output = self.actor_backbone(backbone_input)
                leg_output = self.actor_leg_control_head(backbone_output)
                arm_output = self.actor_arm_control_head(backbone_output)
                if self.adaptive_arm_gains:
                    gains = self.adaptive_arm_gains_scale * (arm_output[:, self.num_arm_actions // 2:])
                    arm_output = torch.cat([arm_output[:, :self.num_arm_actions // 2], gains], dim=-1)
                return torch.cat([leg_output, arm_output], dim=-1)
            
            def infer_priv_latent(self, obs):
                priv = obs[:, self.num_prop: self.num_prop + self.num_priv]
                return self.priv_encoder(priv)
            
            def infer_hist_latent(self, obs):
                print(obs.shape)
                hist = obs[:, -self.num_hist*self.num_prop:]
                return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
     if checkpoint==-1:
         models = [file for file in os.listdir(root) if model_name_include in file]
         models.sort(key=lambda m: '{0:0>15}'.format(m))
         model = models[-1]
     else:
         model = "model_{}_actor.pt".format(checkpoint) 

     load_path = os.path.join(root, model)
     return load_path

def play(load_run, checkpoint):
    num_proprio = 2 + 3 + 20 + 20 + 18 + 4 + 3 + 3 + 3 
    priv_encoder_dims=[64, 20]
    actor = Actor(mlp_input_dim_a = num_proprio, 
                   actor_hidden_dims=[128], 
                   activation=nn.ELU(), 
                   leg_control_head_hidden_dims = [128, 128],
                   arm_control_head_hidden_dims = [128, 128],
                   num_leg_actions=12, 
                   num_arm_actions=6, 
                   adaptive_arm_gains=False, 
                   adaptive_arm_gains_scale=10.,
                   num_priv=5 + 1 + 18, 
                   num_hist=10, 
                   num_prop=num_proprio, 
                   priv_encoder_dims=priv_encoder_dims)

    load_path = get_load_path(root=os.path.join(load_run, 'exported'), checkpoint=checkpoint)
    print("Loading model from: ".format(load_path))
    actor.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    actor.cpu()
    # policy = policy.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    exptid = sys.argv[1].split("_")[0]
    # Save the traced actor
    actor.eval()
    test_input = torch.zeros(1, priv_encoder_dims[-1] + actor.num_prop)
    test = actor(test_input)
    traced_policy = torch.jit.trace(actor, test_input)
    save_path = os.path.join(load_run, "traced", exptid + "_" + str(checkpoint) + "_actor_jit.pt")
    traced_policy.save(save_path)
    print("Saved traced actor at ", save_path)

    test_hist_encoder_input = torch.zeros(1, actor.num_hist*actor.num_prop)
    actor.history_encoder(test_hist_encoder_input)
    traced_encoder = torch.jit.trace(actor.history_encoder, test_hist_encoder_input)
    save_path = os.path.join(load_run, "traced", exptid + "_" + str(checkpoint) + "_hist_encoder_jit.pt")
    traced_encoder.save(save_path)
    print("Saved traced history encoder at ", save_path)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    load_run = "../../logs/rough_widowGo1/" + sys.argv[1]
    checkpoint = int(sys.argv[2])
    play(load_run, checkpoint)