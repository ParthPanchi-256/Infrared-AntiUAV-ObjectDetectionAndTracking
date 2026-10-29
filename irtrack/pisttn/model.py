"""PISTTN network (Zhou & Hu, TGRS 2026), built layer-by-layer from Table VII with spconv.

Interpretation notes where Table VII / the text are ambiguous:
  * Rows with padding 0 and stride 1 are submanifold convs (SubMConv3d): they keep the
    sparsity pattern, and a non-padded dense 3x3x3 conv would shrink T=10 to nothing.
    Strided rows (padding 1) are SparseConv3d; USBlocks invert them via indice keys.
  * "DSBlock6 x2: 32 -> 64, stride (1,2,2)" is a typo in the table: it follows DSBlock5
    (output 64) and a 4th downsample contradicts Table VI (11 DSBlocks -> 1/8 resolution).
    It is implemented as 64 -> 64, stride 1.
  * Query head_l (128 in) and head_h (16 in) have different widths in Table VII, so they
    cannot share weights as the text says; two heads are used, as the table specifies.
  * Eq. (8) adds DSBlock(F_concat) back to F_concat, but their widths differ (e.g. 128 vs 64).
    The residual is taken from the up-path input F_up (same width as the DSBlock output).
  * Detector output (6 channels) = (dx, dy, log w, log h, vx, vy): center offset, size and
    per-frame motion of the target the voxel belongs to. Queries act as the classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as sp

T_DET = 4          # profile detector temporal length
SEG_STARTS = (0, 3, 6)  # 3 segments of 4 frames with 1-frame overlap cover T=10


def ds_block(cin, cout, stride=1, key=None, subm=True):
    if subm:
        conv = sp.SubMConv3d(cin, cout, 3, padding=1, bias=False, indice_key=key)
    else:
        conv = sp.SparseConv3d(cin, cout, 3, stride=stride, padding=1, bias=False, indice_key=key)
    return sp.SparseSequential(conv, nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


def us_block(cin, cout, key):
    return sp.SparseSequential(sp.SparseInverseConv3d(cin, cout, 3, indice_key=key, bias=False),
                               nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    """conv-BN-ReLU, residual add with block input, then conv-ReLU (Sec. II-C)."""

    def __init__(self, c, key):
        super().__init__()
        self.c1 = sp.SubMConv3d(c, c, 3, padding=1, bias=False, indice_key=key)
        self.bn = nn.BatchNorm1d(c)
        self.c2 = sp.SubMConv3d(c, c, 3, padding=1, indice_key=key)

    def forward(self, x):
        y = self.c1(x)
        y = y.replace_feature(F.relu(self.bn(y.features)) + x.features)
        y = self.c2(y)
        return y.replace_feature(F.relu(y.features))


class QueryHead(nn.Module):
    """Three linear layers applied per active voxel (voxels act as the batch dimension)."""

    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 2))

    def forward(self, feats):
        return self.net(feats)


def _cat(a, b):
    return a.replace_feature(torch.cat([a.features, b.features], 1))


class PISTTN(nn.Module):
    def __init__(self, with_detector=True):
        super().__init__()
        # ---- STE encoder: 11 DSBlocks ----
        self.dsb1 = ds_block(1, 16, subm=False, key="enc_in")   # padding 1: dilates the input support
        self.dsb2 = ds_block(16, 16, key="s0")
        self.dsb3 = ds_block(16, 32, (1, 2, 2), key="down1", subm=False)
        self.dsb4 = sp.SparseSequential(ds_block(32, 32, key="s1"), ds_block(32, 32, key="s1"))
        self.dsb5 = ds_block(32, 64, (1, 2, 2), key="down2", subm=False)
        self.dsb6 = sp.SparseSequential(ds_block(64, 64, key="s2"), ds_block(64, 64, key="s2"))
        self.dsb7 = ds_block(64, 64, (1, 2, 2), key="down3", subm=False)
        self.dsb8 = sp.SparseSequential(ds_block(64, 64, key="s3"), ds_block(64, 64, key="s3"))
        # ---- STQ decoder ----
        self.res1, self.res2 = ResBlock(16, "s0"), ResBlock(32, "s1")
        self.res3, self.res4 = ResBlock(64, "s2"), ResBlock(64, "s3")
        self.dsb9, self.usb1 = ds_block(128, 64, key="s3"), us_block(64, 64, "down3")
        self.dsb10, self.usb2 = ds_block(128, 64, key="s2"), us_block(64, 32, "down2")
        self.dsb11, self.usb3 = ds_block(64, 32, key="s1"), us_block(32, 16, "down1")
        self.dsb12, self.dsb13 = ds_block(32, 16, key="s0"), ds_block(16, 16, key="s0")
        self.head_l, self.head_h = QueryHead(128), QueryHead(16)
        # ---- Profile detector: two SPCBLs ----
        self.with_detector = with_detector
        if with_detector:
            self.det = sp.SparseSequential(
                sp.SubMConv3d(2, 16, 3, padding=1, bias=False, indice_key="det"),
                nn.BatchNorm1d(16), nn.LeakyReLU(0.1, inplace=True),
                sp.SubMConv3d(16, 6, 1, bias=False, indice_key="det1"),
                nn.BatchNorm1d(6), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x: sp.SparseConvTensor):
        e0 = self.dsb2(self.dsb1(x))
        e1 = self.dsb4(self.dsb3(e0))
        e2 = self.dsb6(self.dsb5(e1))
        e3 = self.dsb8(self.dsb7(e2))

        hl = _cat(self.res4(e3), e3)                                   # H_l, 128 ch at 1/8
        u = self.dsb9(hl); u = self.usb1(u.replace_feature(u.features + e3.features))
        c = self.dsb10(_cat(self.res3(e2), u)); u = self.usb2(c.replace_feature(c.features + u.features))
        c = self.dsb11(_cat(self.res2(e1), u)); u = self.usb3(c.replace_feature(c.features + u.features))
        c = self.dsb12(_cat(self.res1(e0), u))
        hh = self.dsb13(c.replace_feature(c.features + u.features))  # H_h, 16 ch, full res

        ql = self.head_l(hl.features)                                  # (N_l, 2) Eq. 5
        qh = self.head_h(hh.features)                                  # (N_h, 2)
        q = qh + self._upsample_ql(hl, ql, hh)                         # Eq. 6 + Eq. 9
        out = dict(hl=hl, ql=ql, hh=hh, q=q)
        if self.with_detector:
            out["det"] = self._detect(hh, q)
        return out

    @staticmethod
    def _upsample_ql(hl, ql, hh):
        """Bilinearly upsample the low-res query map (Eq. 6) and sample it at H_h's voxels."""
        B = hl.batch_size
        T, h, w = hl.spatial_shape
        dense = ql.new_full((B, T, 2, h, w), 0.0)
        dense[:, :, 0] = 4.0  # inactive low-res cells = confident background (logit gap 4)
        ci = hl.indices.long()
        dense[ci[:, 0], ci[:, 1], :, ci[:, 2], ci[:, 3]] = ql
        H, W = hh.spatial_shape[1:]
        up = F.interpolate(dense.flatten(0, 1), size=(H, W), mode="bilinear", align_corners=False)
        up = up.view(B, T, 2, H, W)
        hi = hh.indices.long()
        return up[hi[:, 0], hi[:, 1], :, hi[:, 2], hi[:, 3]]

    def _detect(self, hh, q):
        """Split the high-res query tensor into overlapping 4-frame segments and run the detector.

        Returns (seg_index, raw 6-ch output) per voxel; seg_index maps back into hh's voxel list.
        """
        idx = hh.indices
        T = hh.spatial_shape[0]
        parts_i, parts_f, parts_src = [], [], []
        for s, t0 in enumerate(SEG_STARTS):
            m = (idx[:, 1] >= t0) & (idx[:, 1] < t0 + T_DET)
            src = m.nonzero().squeeze(1)
            si = idx[src].clone()
            si[:, 0] = si[:, 0] * len(SEG_STARTS) + s      # each segment is its own batch item
            si[:, 1] -= t0
            parts_i.append(si); parts_f.append(q[src]); parts_src.append(torch.full_like(src, s))
            parts_src[-1] = torch.stack([src, parts_src[-1]], 1)
        seg = sp.SparseConvTensor(torch.cat(parts_f), torch.cat(parts_i).int(),
                                  [T_DET, *hh.spatial_shape[1:]], hh.batch_size * len(SEG_STARTS))
        out = self.det(seg)
        return dict(src=torch.cat(parts_src), raw=out.features)
