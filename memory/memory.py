import numpy as np
import random
import _pickle as pickle
import os
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (np.array(obses_t),
                np.array(actions),
                np.array(rewards),
                np.array(obses_tp1),
                np.array(dones))

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
class ReplayBuffer_decom(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, rewards_decom):
        data = (obs_t, action, reward, obs_tp1, done, rewards_decom)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, rewards_decoms = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, rewards_decom = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            rewards_decoms.append(rewards_decom)
        return (np.array(obses_t),
                np.array(actions),
                np.array(rewards),
                np.array(obses_tp1),
                np.array(dones),
                np.array(rewards_decoms))

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    

    def save(self, path, name=""):
        """Save the memory in case of crash
        Parameters
        ----------
        path: str
            The network path inside the yaml file of the model where the model is being saved
        name: str
            The name you wish to call the saved file default is nothing
        """
        info = {
            "storage": self._storage,
            "maxsize": self._maxsize,
            "next_idx": self._next_idx
        }
        with open(path + "/memory.info", "wb") as file:
#             pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
            p = pickle.Pickler(file) 
            p.fast = True 
            p.dump(info)
            p.memo.clear()
            
    def load(self, path):
        """ Load the parameters of a saved off memory file
        Parameters
        ----------
        path: str
            The path of where the saved off file exists
        """
        restore_path = path + "/memory.info"
        if os.path.exists(restore_path):
            with open(restore_path, "rb") as file:
#                 info = pickle.load(file)
                p = pickle.Unpickler(file) 
#                 p.fast = True 
                info = p.load()
                p.memo.clear()

            self._storage = info["storage"]
            self._maxsize = info["maxsize"]
            self._next_idx = info["next_idx"]