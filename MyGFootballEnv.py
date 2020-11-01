from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

import gfootball
import gfootball.env as football_env

data_dic = {
    'ball_owned_team':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=2, name='ball_owned_team'),
    'steps_left':array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=0.0, name='steps_left'),
    'ball_owned_player':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=11, name='ball_owned_player'),
    'game_mode':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=6, name='game_mode'),
    'designated':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=-1, maximum=11, name='designated'),
    'active':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=-1, maximum=11, name='active'),
    'left_team_active':array_spec.BoundedArraySpec(shape=(11,), dtype=np.int32, minimum=0, maximum=1, name='left_team_active'),
    'left_team':array_spec.BoundedArraySpec(shape=(11,2), dtype=np.float32, minimum=-1.5, maximum=1.5, name='left_team'),
    'right_team_active':array_spec.BoundedArraySpec(shape=(11,), dtype=np.int32, minimum=0, maximum=1, name='right_team_active'),
    'ball_direction':array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, name='ball_direction'),
    'ball': array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, name='ball'),
    'left_team_tired_factor':array_spec.BoundedArraySpec(shape=(11,), dtype=np.float32, minimum=0.0, name='left_team_tired_factor'),
    'left_team_direction':array_spec.BoundedArraySpec(shape=(11,2), dtype=np.float32, minimum=-1.5, maximum=1.5, name='left_team_direction'),
    'score':array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=0.0, name='score'),
    'left_team_roles':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=10, name='left_team_roles'),
    'right_team_tired_factor':array_spec.BoundedArraySpec(shape=(11,), dtype=np.float32, minimum=0.0, name='right_team_tired_factor'),
    'right_team':array_spec.BoundedArraySpec(shape=(11,2), dtype=np.float32, minimum=-1.5, maximum=1.5, name='right_team'),
    'right_team_direction':array_spec.BoundedArraySpec(shape=(11,2), dtype=np.float32, minimum=-1.5, maximum=1.5, name='right_team_direction'),
    'right_team_yellow_card':array_spec.BoundedArraySpec(shape=(11,), dtype=np.int32, minimum=0, maximum=1, name='right_team_yellow_card'),
    'right_team_roles':array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=10, name='right_team_roles'),
    'ball_rotation':array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, name='ball_rotation'),
    'left_team_yellow_card':array_spec.BoundedArraySpec(shape=(11,), dtype=np.int32, minimum=0, maximum=1, name='left_team_yellow_card'),
    'sticky_actions':array_spec.BoundedArraySpec(shape=(10,), dtype=np.int32, minimum=0, maximum=1, name='sticky_actions'),
    }

class MyFootballEnv(py_environment.PyEnvironment):

    @staticmethod
    def convert_observation_to_tf(obs, data_cfg, propertie_list = []):
        
        if(len(propertie_list) == 0):
            propertie_list = ['right_team_direction', 'right_team_tired_factor', 
                               'left_team_roles', 'left_team_direction', 'ball_direction', 
                               'ball_owned_player', 'right_team_yellow_card', 'ball', 
                               'right_team', 'steps_left', 'ball_rotation', 
                               'ball_owned_team', 'game_mode', 'left_team_yellow_card', 
                               'left_team', 'right_team_roles', 'right_team_active', 
                               'left_team_active', 'left_team_tired_factor', 'score', 
                               'designated', 'active', 'sticky_actions']
        rp = {}

        for name in propertie_list:
            
            if(('ball_owned_player' == name) and (obs[0][name] == -1)):
                obs[0][name] = 11
            if(('ball_owned_team' == name) and (obs[0][name] == -1)):
                obs[0][name] = 2
            
            if('right_team_roles' == name):
                if((obs[0]['ball_owned_team'] == 1) and ((obs[0]['ball_owned_player'] != -1) or (obs[0]['ball_owned_player'] != 11))):
                    obs[0][name] = obs[0][name][obs[0]['ball_owned_player']]
                else:
                    obs[0][name] = 10

            if('left_team_roles' == name):
                if((obs[0]['ball_owned_team'] == 0) and ((obs[0]['ball_owned_player'] != -1) or (obs[0]['ball_owned_player'] != 11))):
                    obs[0][name] = obs[0][name][obs[0]['ball_owned_player']]
                else:
                    obs[0][name] = 10


            if(data_cfg[name].shape == (1,)):
                rp[name] = np.array([np.squeeze(obs[0][name])]).astype(data_cfg[name].dtype)
            else:
                a = np.zeros(data_cfg[name].shape, dtype=data_cfg[name].dtype)
                a[:len(obs[0][name])] = obs[0][name]
                obs[0][name] = a.copy()
                rp[name] = np.array(obs[0][name]).astype(data_cfg[name].dtype)

        return rp

    @staticmethod
    def construct_obs_spec(data_cfg, cfg):
        rp = {}
        
        for name in cfg:
            rp[name] = data_cfg[name]
        
        return rp
        
    
    def __init__(self, propertie_list = [], scenario='11_vs_11_kaggle'):
        super().__init__()
        self._data_dic = data_dic
        
        if(len(propertie_list) == 0):
            propertie_list = ['right_team_direction', 'right_team_tired_factor', 
                               'left_team_roles', 'left_team_direction', 'ball_direction', 
                               'ball_owned_player', 'right_team_yellow_card', 'ball', 
                               'right_team', 'steps_left', 'ball_rotation', 
                               'ball_owned_team', 'game_mode', 'left_team_yellow_card', 
                               'left_team', 'right_team_roles', 'right_team_active', 
                               'left_team_active', 'left_team_tired_factor', 'score', 
                               'designated', 'active', 'sticky_actions']

        self.propertie_list = propertie_list
        
        self.env = football_env.create_environment(
              env_name=scenario,  
              stacked=False,
              representation='raw',  
              rewards='scoring, checkpoints',
              write_goal_dumps=False,
              write_full_episode_dumps=False,
              render=False,
              write_video=False,
              dump_frequency=1,
              logdir='./',
              extra_players=None,
              number_of_left_players_agent_controls=1,
              number_of_right_players_agent_controls=0
            )
        
        
        self._state = self.convert_observation_to_tf(self.env.reset(), self._data_dic, self.propertie_list)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=18, name='action')
        
        
        #representation of the enviroment: price + open position state
        self._observation_spec = self.construct_obs_spec(self._data_dic, self.propertie_list)

        #used for idndication of the end of episode        
        self._episode_ended = False

        pass
       
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._state = self.convert_observation_to_tf(self.env.reset(), self._data_dic, self.propertie_list)
        return ts.restart(self._state)


    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        self._state, reward, self._episode_ended, info = self.env.step(action)
        self._state = self.convert_observation_to_tf(self._state, self._data_dic, self.propertie_list)

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

