import numpy as np

class VarianceRatio(object):
    def __init_(self):
        self.div_class = None
        self.sigma_w = None
        self.sigma_b = None
        self.j_sigma = None
        
    def run(self, fs, label, c_num=2):
        self._calc_j_sigma(fs, label, c_num)
        return self.j_sigma
        
    def _devide_class(self, fs, label, c_num):
        dev_class = []
        for i in range(c_num): # c_num:クラス数
            dev_class.append(fs[np.where(label == i)[0]])
        self.div_class = dev_class
        
    def _calc_sigma_w(self, fs):
        class_vals = []
        for _, c in enumerate(self.div_class):
            diff = c - np.mean(c,axis=0)
            inner_prod = [np.dot(vals.T, vals) for _, vals in enumerate(diff)]
            class_vals.append(np.sum(inner_prod))
        self.sigma_w = np.sum(class_vals) / len(fs)
        
    def _calc_sigma_b(self, fs):
        class_vals = []
        for _, c in enumerate(self.div_class):
            diff = np.mean(c,axis=0) - np.mean(fs,axis=0)
            inner_prod = np.dot(diff.T, diff)
            class_vals.append(len(c) * inner_prod)
        self.sigma_b = np.sum(class_vals) / len(fs)
        
    def _calc_j_sigma(self, fs, label, c_num):
        self._devide_class(fs, label, c_num)
        self._calc_sigma_w(fs)
        self._calc_sigma_b(fs)
        self.j_sigma = self.sigma_b / self.sigma_w
        