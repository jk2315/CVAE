
import datetime
import numpy as np
from esig import tosig
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

from utils.leadlag import leadlag
from cvae import CVAE

class cGBM:
    def __init__(self, mu = 0.02, sigma = 0.3, S0 = [1,2,3], n_points = 20, n_data = 220, sig_order = 4):

        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.n_points = n_points
        self.n_data = n_data
        self.order = sig_order

        self._load_data()

        self._build_dataset()
        self.generator = CVAE(n_latent=8, alpha=0.003)

    def _load_data(self):
        try:
            dt = 1. / np.sqrt(self.n_points)
            
            timeline = np.linspace(0, 1, self.n_points)
            self.windows = []
            for s in self.S0:
                for i in range(self.n_data):
                    bm = dt * np.r_[0., np.random.randn(self.n_points - 1).cumsum()]
                    path = s * np.exp((self.mu - self.sigma ** 2 / 2.) * timeline + self.sigma * bm)
                    self.windows.append(leadlag(path))
        except:
            raise RuntimeError(f"Could not produce data for GBM for {self.n_data}.")

    def _logsig(self, path):
        return tosig.stream2logsig(path, self.order)

    def _build_dataset(self):
        if self.order:
            self.orig_logsig = np.array([self._logsig(path) for path in tqdm(self.windows, desc="Computing log-signatures")])
        else:
            self.orig_logsig = np.array([np.diff(np.log(path[::2, 1])) for path in self.windows])

            self.orig_logsig = np.array([p for p in self.orig_logsig if len(p) >= 4])
            steps = min(map(len, self.orig_logsig))
            self.orig_logsig = np.array([val[:steps] for val in self.orig_logsig])

        self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig = self.scaler.fit_transform(self.orig_logsig)

        self.logsigs = logsig
        self.conditions = np.concatenate([np.full((int(len(logsig)/len(self.S0)),np.shape(logsig)[1]),j) for j in range(len(self.S0))])
                

    def train(self, n_epochs=10000):
        self.generator.train(self.logsigs, self.conditions, n_epochs=n_epochs)

    def generate(self, logsig, n_samples=None, normalised=False):
        generated = self.generator.generate(logsig, n_samples=n_samples)

        if normalised:
            return generated

        if n_samples is None:
            return self.scaler.inverse_transform(generated.reshape(1, -1))[0]

        return self.scaler.inverse_transform(generated)
