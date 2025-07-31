import numpy as np
import torch
import scipy.interpolate


class Preprocess62to2s:
    """
    一步完成：
        1. 通道重排 / 取反
        2. Clarke 6→2s
        3. (可选) 只保留 αβ
    """
    def __init__(self,
                 reorder=[0,1,2,3,4,5],

                 keep_alpha_beta=True):
        self.reorder = np.asarray(reorder, dtype=int)

        self.keep_ab = keep_alpha_beta

        # Clarke 矩阵
        self.T = (1/3.) * np.array([
            [1, -0.5, -0.5,  np.sqrt(3)/2, -np.sqrt(3)/2, 0],
            [0,  np.sqrt(3)/2, -np.sqrt(3)/2,  0.5,  0.5, -1],
            [1, -0.5, -0.5, -np.sqrt(3)/2,  np.sqrt(3)/2, 0],
            [0, -np.sqrt(3)/2,  np.sqrt(3)/2,  0.5,  0.5, -1],
            [1,  1,  1,  0, 0, 0],
            [0,  0,  0,  1, 1, 1],
        ], dtype=np.float32)

    def __call__(self, x):
        # ---------- 通道修正 ----------
        if x.shape[0] == 6:        # (6, L)
            x = x[self.reorder, :]

        else:                      # (L, 6)
            x = x[:, self.reorder]


        # ---------- Clarke ----------
        x = self.T @ x if x.shape[0] == 6 else x @ self.T.T

        # ---------- 只保留 αβ ----------
        if self.keep_ab:
            x = x[0:4, :] if x.shape[0] == 6 else x[:, 0:4]

        return x
