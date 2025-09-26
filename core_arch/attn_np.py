import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

print(X.shape)

Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

print(Wv.shape)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print(Q.shape)

scale = 1.0 / np.sqrt(Q.shape[-1])
attn_sc = (Q @ K.transpose(0, 2, 1)) * scale
print(attn_sc.shape)

mask = np.triu(np.ones((1, 3, 3), dtype=bool), k=1)
print(mask, mask.shape)
attn_sc = np.where(mask, -1e9, attn_sc)
print(attn_sc, attn_sc.shape)

wei = np.exp(attn_sc - attn_sc.max(axis=-1, keepdims=True))
wei = wei / wei.sum(axis=-1, keepdims=True)
print(wei.shape)

out = wei @ V
print(out.shape)

