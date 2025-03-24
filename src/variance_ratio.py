import numpy as np

class VarianceRatio(object):
    def __init__(self):
        self.div_class = None
        self.sigma_b = None
        self.sigma_w = None
        self.j_sigma = None
        
    def fit(self, fs, label):
        assert len(fs)==len(label), 'サンプルサイズとラベルサイズを統一してください．'
        self._calc_j_sigma(fs, label)
        return self.j_sigma

    def _calc_j_sigma(self, fs, label):
        self._divide_class(fs, label)
        self._calc_sigma_b(fs)
        self._calc_sigma_w(fs)
        self.j_sigma = self.sigma_b / self.sigma_w
        
    def _divide_class(self, fs, label):
        dev_class = []
        unique_labels = list(set(label))
        for ul in unique_labels:
            dev_class.append(fs[np.where(label == int(ul))[0]])
        self.div_class = dev_class

    def _calc_sigma_b(self, fs):
        class_vals = []
        for _, c in enumerate(self.div_class):
            diff = np.mean(c,axis=0) - np.mean(fs,axis=0)
            inner_prod = np.linalg.norm(diff, ord=2, axis=0)
            assert np.isscalar(inner_prod), 'ノルムの計算をする軸方向が間違っています'
            # サンプルの多いクラスに平均が引っ張られるので，重みとしてサンプルサイズを乗じる
            class_vals.append(len(c) * inner_prod)
        self.sigma_b = np.sum(class_vals) / len(fs)
        
    def _calc_sigma_w(self, fs):
        class_vals = []
        for _, c in enumerate(self.div_class):
            # 各特徴量における個々のサンプル値とそれら特徴量の平均値との差分をとる
            diff = c - np.mean(c,axis=0)
            # 差分ベクトルの大きさを取ることで，サンプルごとのばらつきを計算する
            inner_prod = np.linalg.norm(diff, ord=2, axis=1)
            class_vals.append(np.sum(inner_prod))
        # それらを全クラスに対して行い総和をとることで，全体のクラス内分散を算出する
        self.sigma_w = (np.sum(class_vals) / len(fs)) if np.sum(class_vals)!=0 else np.spacing(1)