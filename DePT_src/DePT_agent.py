import numpy as np 
import os 
import pickle  
from agent import Agent
import random 
import time
from keras.utils import to_categorical
import copy

import torch
import torch.nn as nn

from .model import *
from .dataprep import getID2Pos
from glob import glob




class DePTAgent(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None,
        best_round=None, bar_round=None,intersection_id="0"):
        super(DePTAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        self.assigned_total_cnt = dic_agent_conf['RUN_COUNTS']
        self.n_generators = dic_agent_conf['NUM_GENERATORS']

        self.cnt_round = cnt_round
        self.N_nodes=dic_traffic_env_conf['NUM_INTERSECTIONS']
        l1, l2 = dic_path['PATH_TO_DATA'].split('/')[-1].split('_')
        N_nodes = int(l1)*int(l2)
        assert N_nodes==self.N_nodes
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.args_cacot = getArgs(); assert self.num_actions==self.args_cacot['n_actions']
        self.batch_size = self.args_cacot['batch_size']
        self.ddqn_freq = self.args_cacot['ddqn_freq']
        self.multiGPU = self.args_cacot['multiGPU']
        self.action_epsilon = max(0.2, 0.8 * 0.95**cnt_round)
        roadnet_file = glob(os.path.join(dic_path['PATH_TO_DATA'],'roadnet*'))
        assert len(roadnet_file)==1
        ID2Pos = getID2Pos(roadnet_file[0], dic_traffic_env_conf)
        self.q_network = DelayedPropagationTransformer(ID2Pos, N_nodes).to(DEVICE)
        if self.multiGPU: 
            self.q_network = nn.DataParallel(self.q_network, device_ids=self.multiGPU).module
        if cnt_round > 0:
            load_model(self.q_network, self.args_cacot['cacot_fname'])
        if self.args_cacot['which_teacher']=='cacot':
            self.q_network_bar = DelayedPropagationTransformer(ID2Pos, N_nodes).to(DEVICE)
            if self.multiGPU: 
                self.q_network_bar = nn.DataParallel(self.q_network_bar, device_ids=self.multiGPU).module
            if cnt_round > 0:
                if cnt_round % self.ddqn_freq==0:
                    load_model(self.q_network_bar, self.args_cacot['cacot_fname'])
                else:
                    load_model(self.q_network_bar, self.args_cacot['cacot_bar_fname'])
        elif self.args_cacot['which_teacher']=='colight':
            self.q_network_bar = load_colight(self.args_cacot['colight_fname'])
        else:
            raise NotImplementedError
        if self.args_cacot['use_pretrained_cone']:
            assert N_nodes in [36, 28*7]
            which_roadmap = '6x6' if N_nodes==36 else 'newyork'
            load_cone(self.q_network.cone, which_roadmap, self.args_cacot['n_head'])

    def prepare_Xs_Y(self, memory, dic_exp_conf):
        ind_end = len(memory)
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        all_samples = self.assigned_total_cnt*self.n_generators
        sampleRate = 0.8
        some_int = max(1, int(all_samples*sampleRate//self.batch_size))
        sample_size = int(some_int*self.batch_size)
        N = len(memory_after_forget)
        _state = []
        _next_state = []
        _action=[]
        _reward=[]
        sample_slice_idx = np.random.choice(np.arange(N,dtype=int)[self.args_cacot['Tmax']:], sample_size)
        for i in range(N):  
            _state.append([])
            _next_state.append([])
            for j in range(self.N_nodes): # self.num_agents=36
                state, action, next_state, reward, _ = memory_after_forget[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
        for _i in range(sample_size):  
            _action.append([])
            _reward.append([])
            i = sample_slice_idx[_i]
            for j in range(self.N_nodes):
                state, action, next_state, reward, _ = memory_after_forget[i][j]
                _action[_i].append(action)
                _reward[_i].append(reward)
        _features,_adjs,q_values,_=self.action_att_predict(_state, 
            is_batch=True, 
            batch_idx=sample_slice_idx,
            bar=False)
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            is_batch=True, 
            batch_idx=sample_slice_idx,
            bar=True)
        assert sample_size/self.args_cacot['batch_size']%1==0
        for i in range(sample_size):  
            for j in range(self.N_nodes):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        return 


    def train_network(self):
        n_epochs = self.args_cacot['n_epochs']
        epoch_len = len(self.Y)//self.batch_size
        lossfun = nn.MSELoss()
        optimizer = torch.optim.SGD(self.q_network.parameters(), lr=self.args_cacot['lr'], momentum=0.9)
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.args_cacot['lr'])
        X = self.Xs[0]
        Y = self.Y
        allloss = []
        for ie in tqdm(range(n_epochs)):
            print(f'training in progress: {ie} / {n_epochs},')
            X, Y = shuffle_Xy(X, Y)
            for ib in range(epoch_len):
                x = X[ib*self.batch_size:(ib+1)*self.batch_size]
                ytrue = torch.tensor(Y[ib*self.batch_size:(ib+1)*self.batch_size],device=DEVICE,dtype=torch.float32)
                ypred = self.q_network(x)

                loss = lossfun(ytrue,ypred)
                loss.backward()
                optimizer.step()
                allloss.append(loss.item())
        return 

    def save_network(self):
        save_model(self.q_network, self.args_cacot['cacot_fname'])
        if self.args_cacot['which_teacher']=='cacot':
            save_model(self.q_network_bar, self.args_cacot['cacot_bar_fname'])

    def choose_action(self, count, state_ns, is_optimal=False):
        act, attention = self.action_att_predict([state_ns], is_batch=False, interacting_cnt=count,bar=False, is_optimal=is_optimal)
        return act, attention

    def action_att_predict(self,state_bns,
        is_batch, 
        batch_idx=None,
        interacting_cnt=None,
        bar=False,
        is_optimal=False):
        _batch_size=len(state_bns)
        features_bnf = []
        total_adjs = []
        for i in range(_batch_size): 
            adj=[]
            feature_nf = self.convert_state_to_input_1sample(state_bns[i])
            for j in range(self.N_nodes):
                adj.append(state_bns[i][j]['adjacency_matrix'])

            features_bnf.append(feature_nf)
            total_adjs.append(adj)
        features_bnf = np.array(features_bnf)


        if is_batch:
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs)[batch_idx])
            features_out_bTnf = []
            action_out = []
            attention_out = []
            N_process = len(batch_idx)//self.batch_size
            for ib in range(N_process):
                features_bTnf = time_expand_reverse(features_bnf, batch_idx, self.args_cacot['Tmax'], _range=[ib*self.args_cacot['batch_size'] , (ib+1)*self.args_cacot['batch_size'] ])
                features_out_bTnf.extend(features_bTnf)
                if bar:
                    if self.args_cacot['which_teacher']=='cacot':
                        all_output= self.q_network_bar.predict([features_bTnf,total_adjs])
                    elif self.args_cacot['which_teacher']=='colight':
                        all_output= self.q_network_bar.predict([to_colight_input(features_bTnf),total_adjs])
                    else:
                        raise NotImplementedError
                else:
                    all_output = self.q_network.predict([features_bTnf,total_adjs])

                action, attention = all_output
                if type(action) is not np.ndarray:
                    action = action.cpu().data.numpy().tolist()
                action_out.extend(action)
                # attention_out.extend(attention)

            return np.array(features_out_bTnf),total_adjs,np.array(action_out),attention_out

        else:

            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))

            assert _batch_size==1 and (interacting_cnt is not None)

            if interacting_cnt==0:
                self.bufferInteract_Tnf = [feature_nf for _ in range(self.args_cacot['Tmax'])]

            del self.bufferInteract_Tnf[-1]

            self.bufferInteract_Tnf.insert(0, feature_nf)



            actor_is_cacot = 1

            if actor_is_cacot:
                action_bna, attention = self.q_network.predict([[self.bufferInteract_Tnf], None])
                action_bna = action_bna.cpu().data.numpy()

            else:
                action_bna, attention = self.q_network_bar.predict([to_colight_input([self.bufferInteract_Tnf]),total_adjs]);is_optimal=True

            action_bn = action_bna.argmax(axis=-1)
            bestaction_n = action_bn[0]    # [n_nodes,]

            if is_optimal:
                # print('..........using optimal action')
                return bestaction_n, attention
            else:
                act = []
                for inode in range(len(bestaction_n)):
                    _act = np.random.randint(self.num_actions) if np.random.rand()<self.action_epsilon else bestaction_n[inode]
                    act.append(_act)


                return np.array(act), attention









    def convert_state_to_input_1sample(self,state_ns):
        
        # return: [N_nodes, nodeID + timeFineGrained + lane_num_vehicle (12 dims) + cur_phase (int)]
        # where is lane_num_vehicle: [2:-1] = [2:14]
        # where is actionIdx: [-1]
        feature=[]
        for j in range(self.N_nodes):
            ##WPH: Dummy time ####
            ###WPH: Initialized with nodeID + timeFineGrained
            observation = [j,]

            observation.append(state_ns[j]['cur_time'])
            observation.extend(state_ns[j]['lane_num_vehicle']) # len=12
            observation.append(state_ns[j]['cur_phase'][0])

            assert len(state_ns[j]["cur_phase"])==1
            feature.append(observation)

        feature_nf = np.asarray(feature)

        return feature_nf


    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        l = to_categorical(adjacency_index_new,num_classes=self.N_nodes)
        return l


    def load_network(self, xxx, xxx2=None):
        load_model(self.q_network, self.args_cacot['cacot_fname'])


def to_colight_input(features_bTnf):
    phase_dic_colight = {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
    }

    curSnapshot = np.asarray(features_bTnf)[:,0,...] # [batch, Nodes, 2+12+1]
    feats = curSnapshot[...,2:-1]       # [batch, Nodes, 12]
    actionIdx = curSnapshot[...,-1]     # [batch, Nodes]
    batch, Nodes = actionIdx.shape
    action_emb = np.zeros((batch, Nodes, 8))
    for ib in range(len(actionIdx)):
        for ino in range(len(actionIdx[0])):
            action_emb[ib,ino,:] = np.asarray(phase_dic_colight[actionIdx[ib,ino]])
    feat4colight = np.concatenate([action_emb,feats], axis=-1)
    return feat4colight



def load_colight(colight_model_name):
    from keras.engine.topology import Layer
    from keras import backend as K
    class RepeatVector3D(Layer):
        def __init__(self,times,**kwargs):
            super(RepeatVector3D, self).__init__(**kwargs)
            self.times = times

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.times, input_shape[1],input_shape[2])

        def call(self, inputs):
            #[batch,agent,dim]->[batch,1,agent,dim]
            #[batch,1,agent,dim]->[batch,agent,agent,dim]

            return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])


        def get_config(self):
            config = {'times': self.times}
            base_config = super(RepeatVector3D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    from keras.models import load_model as keras_load_model
    colight = keras_load_model(colight_model_name,
        custom_objects={'RepeatVector3D':RepeatVector3D})
    print('\n\n----- ›››››››  LOAD colight teacher success!\n')

    return colight



def time_expand_reverse(bx, idx_list, memLen, _range=None):
    # bx is either torch tensor or np array, or list
    # return: [items, memLen, x] where items are assigned by idx_list, and idx_list is optionally focued on starting/end point in _range

    # bx = copy.deepcopy(bx)
    if _range is not None:
        start, end = int(_range[0]), int(_range[1])
    else:
        start, end = 0, len(idx_list)
    assert min(idx_list)>=memLen
    res = []
    for i in range(start,end):
        idx = idx_list[i]
        mem = bx[idx-memLen:idx][::-1]
        res.append(mem)

    return res



