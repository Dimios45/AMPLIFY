"""
EgocentricDataset
=================

Dataset loader for egocentric video data in the HaWoR-processed format
(hand detection bounding boxes + MANO params + SLAM trajectory).

Expected layout on disk::

    root_dir/
      {split}/                             # e.g. gcs_scored, high_quality
        {video_id}/
          {video_id}.mp4                   # 1920×1080 @ 30 fps
          {video_id}/                      # processed artefacts
            tracks_0_{N}/
              model_tracks.npy            # hand detection tracks (dict of per-frame bboxes)
            cam_space/
              0/
                {start}_{end}.json        # MANO params for frames [start, end]
            SLAM/
              hawor_slam_w_scale_0_{N}.npz
            head_pose_vo.npy              # (N_frames, 7) camera VO trajectory
            est_focal.txt                 # focal length in pixels

Track synthesis (no CoTracker required)
----------------------------------------
`model_tracks.npy` gives per-frame hand bounding boxes.  We synthesise
400 query-point tracks by:

1. Placing a uniform 20×20 grid over the *input* image.
2. For each grid point, checking whether it falls inside a detected hand
   bbox at `start_t`.  Points inside a bbox are tagged as "hand points".
3. For each subsequent frame `t` in the window, hand points are shifted by
   the displacement of their associated hand's centroid.  Points outside
   all bboxes stay static (background).
4. Visibility is 1.0 for frames where the owning hand is detected, 0.5
   otherwise (hand briefly occluded).

This is a coarse approximation — use ``use_cotracker_tracks=True`` and
run ``scripts/preprocess_egocentric.py`` first for proper optical-flow
tracks when training the motion tokenizer.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from amplify.loaders.base_dataset import BaseDataset
from amplify.utils.data_utils import interpolate_traj, normalize_traj


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_model_tracks(path: str) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load ``model_tracks.npy`` and return a compact per-frame lookup.

    Returns
    -------
    Dict[track_id -> Dict[frame_idx -> bbox (4,)]]
        bbox = [x1, y1, x2, y2] in pixel coordinates.
        handedness stored separately as track_id -> float (0=left, 1=right).
    Also attaches ``._handedness`` as a side-channel via a plain dict attribute.
    """
    raw: dict = np.load(path, allow_pickle=True).item()

    # frame_dets[frame] = list of (bbox_xyxy, handedness_float)
    frame_dets: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for track_id, entries in raw.items():
        for e in entries:
            frame = int(e['frame'])
            box_raw = e['det_box']          # (1, 5) or (1, 4)
            box = box_raw[0, :4].astype(np.float32)  # x1 y1 x2 y2
            hand = float(e['det_handedness'][0])      # 0=left, 1=right
            frame_dets.setdefault(frame, []).append((box, hand))

    return frame_dets


def _box_centroid(box: np.ndarray) -> np.ndarray:
    """Return (cx, cy) of an [x1, y1, x2, y2] box."""
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _read_frames(mp4_path: str, start: int, end: int) -> np.ndarray:
    """
    Read frames [start, end) from an mp4 using OpenCV.

    Returns
    -------
    np.ndarray  shape (T, H, W, 3) uint8 RGB
    """
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            # Pad with black if video is shorter than expected
            frames.append(np.zeros_like(frames[-1]) if frames else np.zeros((1080, 1920, 3), np.uint8))
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def _video_n_frames(mp4_path: str) -> int:
    cap = cv2.VideoCapture(mp4_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _find_tracks_dir(data_dir: str) -> Optional[str]:
    """Return the ``tracks_0_*`` directory inside *data_dir*, if present."""
    candidates = glob.glob(os.path.join(data_dir, 'tracks_0_*'))
    return candidates[0] if candidates else None


def _find_slam_file(data_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(data_dir, 'SLAM', 'hawor_slam_w_scale_*.npz'))
    return candidates[0] if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Main dataset class
# ──────────────────────────────────────────────────────────────────────────────

class EgocentricDataset(BaseDataset):
    """
    Egocentric video dataset using HaWoR hand-detection artefacts.

    Parameters
    ----------
    root_dir : str
        Parent directory that contains one sub-directory per split
        (e.g. ``gcs_scored``, ``high_quality``).
    dataset_names : list[str]
        Which split sub-directories to include, e.g. ``['gcs_scored']``.
    window_stride : int
        Stride (in frames) between consecutive training windows.
        Default 8 → 50 % overlap between windows.  Use ``true_horizon``
        for non-overlapping windows.
    use_cotracker_tracks : bool
        If True, load pre-computed CoTracker track files instead of
        synthesising bbox-based pseudo-tracks.  The track files must live at
        ``<data_dir>/cotracker_tracks.npy`` (shape ``(N_frames, 400, 2)``).
    min_hand_frames : int
        Minimum number of frames in a window that must contain at least one
        hand detection for the window to be included in the index.
        Set to 0 to include all windows.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_400_reinit_16',
        cond_cameraviews: List[str] = ('egocentric',),
        keys_to_load: List[str] = ('images', 'tracks'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 16,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        aug_cfg: Dict = None,
        window_stride: int = 8,
        use_cotracker_tracks: bool = False,
        min_hand_frames: int = 1,
        video_subset: float = 1.0,
    ):
        self.window_stride = window_stride
        self.use_cotracker_tracks = use_cotracker_tracks
        self.min_hand_frames = min_hand_frames
        self.video_subset = video_subset

        # Call super — this triggers create_index_map
        super().__init__(
            root_dir=root_dir,
            dataset_names=dataset_names,
            track_method=track_method,
            cond_cameraviews=list(cond_cameraviews),
            keys_to_load=list(keys_to_load),
            img_shape=img_shape,
            true_horizon=true_horizon,
            track_pred_horizon=track_pred_horizon,
            interp_method=interp_method,
            num_tracks=num_tracks,
            use_cached_index_map=use_cached_index_map,
            aug_cfg=aug_cfg,
        )

        # Uniform grid query points in *input* image space (1920×1080).
        # These are the initial positions before any motion is applied.
        self._grid_xy = self._make_grid_xy(1920, 1080, num_tracks)  # (N, 2) x,y

    # ── index map ─────────────────────────────────────────────────────────────

    def get_cache_file(self) -> str:
        splits = '_'.join(sorted(self.dataset_names))
        return os.path.expanduser(
            f'~/.cache/amplify/index_maps/egocentric/'
            f'{splits}_stride{self.window_stride}_mhf{self.min_hand_frames}_sub{self.video_subset:.2f}.json'
        )

    def create_index_map(self) -> List[Dict]:
        index: List[Dict] = []

        for split in self.dataset_names:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.isdir(split_dir):
                print(f"[warn] split directory not found: {split_dir}")
                continue

            all_video_ids = sorted(os.listdir(split_dir))
            n_total = len(all_video_ids)

            # Apply video_subset: positive = first N%, negative = last N%
            subset_size = max(1, int(n_total * abs(self.video_subset)))
            if self.video_subset >= 0:
                video_ids = all_video_ids[:subset_size]
            else:
                video_ids = all_video_ids[n_total - subset_size:]

            print(f"  {split}: using {len(video_ids)}/{n_total} videos "
                  f"(video_subset={self.video_subset:.2f})")

            for vid_id in tqdm(video_ids, desc=f'indexing {split}'):
                entry = self._make_video_entries(split_dir, vid_id, split)
                index.extend(entry)

        return index

    def _make_video_entries(self, split_dir: str, vid_id: str, split: str) -> List[Dict]:
        vid_dir = os.path.join(split_dir, vid_id)
        mp4_path = os.path.join(vid_dir, f'{vid_id}.mp4')
        data_dir = os.path.join(vid_dir, vid_id)

        if not os.path.isfile(mp4_path):
            return []

        # Create data_dir on the fly if it doesn't exist yet
        # (needed for CoTracker-only splits with no HaWoR artefacts)
        os.makedirs(data_dir, exist_ok=True)

        # --- model_tracks path (may not exist for CoTracker-only datasets) ---
        tracks_dir = _find_tracks_dir(data_dir)
        if tracks_dir is not None:
            model_tracks_path = os.path.join(tracks_dir, 'model_tracks.npy')
        else:
            model_tracks_path = None

        # Auto-detect: if cotracker_tracks.npy exists, we can always proceed.
        cotracker_path = os.path.join(data_dir, 'cotracker_tracks.npy')
        has_cotracker = os.path.isfile(cotracker_path)

        # When using bbox/MANO tracks we need model_tracks.npy.
        # When CoTracker tracks are available (either forced or auto-detected), we don't.
        if not has_cotracker and not self.use_cotracker_tracks and model_tracks_path is None:
            return []
        if not has_cotracker and not self.use_cotracker_tracks and not os.path.isfile(model_tracks_path):
            return []

        n_frames = _video_n_frames(mp4_path)
        if n_frames < self.true_horizon:
            return []

        # Load frame detections only if available (needed for hand-frame filtering)
        if model_tracks_path and os.path.isfile(model_tracks_path):
            frame_dets = _parse_model_tracks(model_tracks_path)
        else:
            frame_dets = {}   # no detections — skip min_hand_frames filter

        entries = []
        for start_t in range(0, n_frames - self.true_horizon + 1, self.window_stride):
            end_t = start_t + self.true_horizon

            if self.min_hand_frames > 0 and frame_dets:
                n_hand = sum(1 for f in range(start_t, end_t) if f in frame_dets)
                if n_hand < self.min_hand_frames:
                    continue

            entries.append({
                'split': split,
                'video_id': vid_id,
                'mp4_path': mp4_path,
                'data_dir': data_dir,
                'model_tracks_path': model_tracks_path,
                'start_t': start_t,
                'end_t': end_t,
                'rollout_len': n_frames,
            })

        return entries

    # ── per-key loaders ───────────────────────────────────────────────────────

    def load_images(self, idx_dict: Dict) -> Dict:
        """
        Returns the single conditioning frame at ``start_t``.

        Returns
        -------
        dict
            ``'images'``: np.float32 ``(V=1, H, W, C=3)`` in [0, 255].
        """
        start_t = idx_dict['start_t']
        frames = _read_frames(idx_dict['mp4_path'], start_t, start_t + 1)  # (1, H, W, 3)
        return {'images': frames.astype(np.float32)}

    def load_actions(self, idx_dict: Dict) -> Dict:
        """No robot action labels — return zeros."""
        T = idx_dict['end_t'] - idx_dict['start_t']
        return {'actions': np.zeros((T, 7), dtype=np.float32)}

    def load_proprioception(self, idx_dict: Dict) -> Dict:
        """No proprioception — return zeros."""
        return {'proprioception': np.zeros(9, dtype=np.float32)}

    def load_tracks(self, idx_dict: Dict) -> Dict:
        """
        Load tracks for the window ``[start_t, end_t)``.

        Priority:
        1. CoTracker tracks (if ``use_cotracker_tracks=True``)
        2. MANO 2D joints  (if ``mano_joints_2d.npy`` exists in data_dir)
        3. Bbox-based pseudo-tracks (fallback)

        Returns
        -------
        dict
            ``'tracks'``: np.float32 ``(V=1, T, N, 2)`` in pixel (x, y).
            ``'vis'``:    np.float32 ``(V=1, T, N, 1)``.
        """
        # Auto-detect CoTracker tracks even if use_cotracker_tracks=False
        _ct_path = os.path.join(idx_dict['data_dir'], 'cotracker_tracks.npy')
        if self.use_cotracker_tracks or os.path.isfile(_ct_path):
            return self._load_cotracker_tracks(idx_dict)

        mano_path = os.path.join(idx_dict['data_dir'], 'mano_joints_2d.npy')
        if os.path.isfile(mano_path):
            return self._load_mano_tracks(idx_dict, mano_path)

        start_t = idx_dict['start_t']
        end_t = idx_dict['end_t']

        frame_dets = _get_cached_frame_dets(idx_dict['model_tracks_path'])
        tracks, vis = self._synthesise_tracks(frame_dets, start_t, end_t)
        # tracks: (T, N, 2), vis: (T, N)
        return {
            'tracks': tracks[np.newaxis],                    # (1, T, N, 2)
            'vis':    vis[np.newaxis, ..., np.newaxis],      # (1, T, N, 1)
        }

    def _load_mano_tracks(self, idx_dict: Dict, mano_path: str) -> Dict:
        """
        Load MANO 2D joint tracks from ``mano_joints_2d.npy``.

        The file has shape ``(N_frames, 2_hands, 16_joints, 2_coords)``
        with NaN where no hand was detected.

        We flatten hands × joints → N tracks and build visibility from NaN mask.
        If ``self.num_tracks > 32`` (hands × joints), the remaining slots are
        padded with static background grid points at 0.5 visibility.
        """
        s, e = idx_dict['start_t'], idx_dict['end_t']
        T = e - s

        all_joints = _get_cached_mano_joints(mano_path)   # (N_frames, 2, 16, 2)
        window = all_joints[s:e]                           # (T, 2, 16, 2)

        # Flatten hands × joints → (T, 32, 2)
        T_actual = window.shape[0]
        joints_flat = window.reshape(T_actual, -1, 2)      # (T, 32, 2)
        n_mano = joints_flat.shape[1]                      # 32

        # Visibility: 1.0 where not NaN, 0.0 where NaN
        vis_flat = (~np.isnan(joints_flat[..., 0])).astype(np.float32)  # (T, 32)

        # Replace NaN with last valid position (forward fill per joint)
        for j in range(n_mano):
            last = np.array([960.0, 540.0], dtype=np.float32)  # image centre
            for t in range(T_actual):
                if not np.isnan(joints_flat[t, j, 0]):
                    last = joints_flat[t, j].copy()
                else:
                    joints_flat[t, j] = last

        # Pad to self.num_tracks with static background grid if needed
        N = self.num_tracks
        if n_mano < N:
            n_pad = N - n_mano
            # Sample n_pad points from the full image grid
            grid = self._grid_xy[:n_pad]              # (n_pad, 2) pixel x,y
            pad_tracks = np.tile(grid, (T_actual, 1, 1))   # (T, n_pad, 2)
            pad_vis = np.full((T_actual, n_pad), 0.5, dtype=np.float32)
            tracks_full = np.concatenate([joints_flat, pad_tracks], axis=1)  # (T, N, 2)
            vis_full    = np.concatenate([vis_flat,    pad_vis],    axis=1)  # (T, N)
        else:
            tracks_full = joints_flat[:, :N]
            vis_full    = vis_flat[:, :N]

        return {
            'tracks': tracks_full[np.newaxis].astype(np.float32),       # (1, T, N, 2)
            'vis':    vis_full[np.newaxis, ..., np.newaxis].astype(np.float32),  # (1, T, N, 1)
        }

    def load_text(self, idx_dict: Dict) -> Dict:
        """Constant task description — used by forward dynamics as weak conditioning."""
        return {'text': 'factory worker performing precision manipulation task'}

    # ── track synthesis ───────────────────────────────────────────────────────

    def _synthesise_tracks(
        self,
        frame_dets: Dict[int, List[Tuple[np.ndarray, float]]],
        start_t: int,
        end_t: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (T, N, 2) tracks and (T, N) visibility from bbox detections.

        Strategy
        --------
        * Start with a static uniform 20×20 grid of 400 points.
        * At ``start_t``, for each detected hand, find the grid points that
          fall inside the bbox and tag them as belonging to that hand.
        * For each subsequent frame, shift tagged points by the delta of their
          hand's centroid.  Points with no detection in a frame get their last
          known centroid extrapolated.
        * Visibility = 1.0 when the owning hand is detected, 0.5 otherwise.
        """
        N = self.num_tracks
        T = end_t - start_t

        # Initial grid positions at frame start_t (float x, y in pixel space)
        pts = self._grid_xy.copy()  # (N, 2)

        # Determine hand-centroid track for each frame ─ index [0]=left, [1]=right
        # centroids[t] = {hand_id: centroid_xy}  where hand_id ∈ {0, 1}
        centroids_per_frame: List[Dict[int, np.ndarray]] = []
        last_centroids: Dict[int, np.ndarray] = {}
        for t in range(T):
            frame_idx = start_t + t
            dets = frame_dets.get(frame_idx, [])
            cur: Dict[int, np.ndarray] = {}
            if dets:
                # Aggregate: per handedness take the largest bbox
                for box, hand in dets:
                    hid = int(round(hand))   # 0=left, 1=right
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if hid not in cur or area > (
                        (cur[hid][0] - box[0]) ** 2 + (cur[hid][1] - box[1]) ** 2
                    ):
                        cur[hid] = _box_centroid(box)
            # Fill missing hands from last known position
            for hid, c in last_centroids.items():
                if hid not in cur:
                    cur[hid] = c
            last_centroids = dict(cur)
            centroids_per_frame.append(cur)

        # Assign each grid point to a hand (based on which bbox it falls inside
        # at start_t) or to background (-1).
        assignment = np.full(N, -1, dtype=int)
        anchor_centroid = np.full((N, 2), np.nan, dtype=np.float32)

        if frame_dets.get(start_t):
            dets_t0 = frame_dets[start_t]
            for i, (px, py) in enumerate(pts):
                for box, hand in dets_t0:
                    if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                        hid = int(round(hand))
                        assignment[i] = hid
                        anchor_centroid[i] = _box_centroid(box)
                        break

        # Build tracks by applying centroid deltas
        tracks = np.zeros((T, N, 2), dtype=np.float32)
        vis = np.full((T, N), 0.5, dtype=np.float32)

        init_centroids = centroids_per_frame[0]

        for t in range(T):
            cur_centroids = centroids_per_frame[t]
            positions = pts.copy()  # start from initial grid

            for i in range(N):
                hid = assignment[i]
                if hid == -1:
                    # Background point — stays static
                    vis[t, i] = 0.5
                else:
                    if hid in cur_centroids and hid in init_centroids:
                        delta = cur_centroids[hid] - init_centroids[hid]
                        positions[i] = pts[i] + delta
                        vis[t, i] = 1.0
                    else:
                        # Hand not detected at this frame
                        vis[t, i] = 0.3

            tracks[t] = positions  # (N, 2) pixel x,y

        return tracks, vis

    def _load_cotracker_tracks(self, idx_dict: Dict) -> Dict:
        """
        Load pre-computed CoTracker tracks from ``<data_dir>/cotracker_tracks.npy``.
        Expected shape: ``(N_frames, N, 2)`` in pixel (x, y).
        Also loads ``cotracker_vis.npy`` if present, else assumes all visible.
        """
        path = os.path.join(idx_dict['data_dir'], 'cotracker_tracks.npy')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'CoTracker tracks not found at {path}. '
                'Run scripts/preprocess_cotracker.py first, or set '
                'use_cotracker_tracks=False.'
            )
        all_tracks = _get_cached_cotracker_tracks(path)  # (N_frames, N, 2)
        s, e = idx_dict['start_t'], idx_dict['end_t']
        tracks = all_tracks[s:e].copy()   # (T, N, 2)

        vis_path = os.path.join(idx_dict['data_dir'], 'cotracker_vis.npy')
        if os.path.isfile(vis_path):
            all_vis = _get_cached_cotracker_vis(vis_path)  # (N_frames, N) bool
            vis = all_vis[s:e].astype(np.float32)          # (T, N)
        else:
            vis = np.ones((e - s, tracks.shape[1]), dtype=np.float32)

        # Replace NaN positions (from short-video padding) with image centre
        nan_mask = np.isnan(tracks[..., 0])
        tracks[nan_mask] = np.array([960.0, 540.0], dtype=np.float32)
        vis[nan_mask] = 0.0

        return {
            'tracks': tracks[np.newaxis],
            'vis': vis[np.newaxis, ..., np.newaxis],
        }

    # ── processing ────────────────────────────────────────────────────────────

    def process_data(self, data: Dict) -> Dict:
        """
        Normalise and reshape to AMPLIFY canonical format.

        images: uint8/float → float32 [0,1], resized to img_shape, shape (V, H, W, C)
        tracks → traj: pixel (x,y) → normalized (row,col) ∈ [-1,1], shape (V, Ht, N, 2)
        vis:    shape (V, Ht, N, 1)
        """
        H_in, W_in = 1080, 1920
        H_out, W_out = self.img_shape

        # ── images ──
        if 'images' in data:
            imgs = data['images']               # (V, H_in, W_in, C)
            imgs = imgs / 255.0 if imgs.max() > 1.0 else imgs
            # Resize each view
            resized = []
            for v in range(imgs.shape[0]):
                frame = cv2.resize(
                    imgs[v].astype(np.float32),
                    (W_out, H_out),
                    interpolation=cv2.INTER_AREA,
                )
                resized.append(frame)
            data['images'] = np.stack(resized, axis=0).astype(np.float32)  # (V, H_out, W_out, C)

        # ── tracks → traj ──
        if 'tracks' in data:
            tracks = data.pop('tracks')    # (V, T, N, 2) pixel x,y

            # Scale pixel coords to the resized image space, then convert
            # x,y → row,col and normalise to [-1, 1]
            #   col = x * (W_out / W_in)  → col_norm = col / (W_out/2) - 1
            #   row = y * (H_out / H_in)  → row_norm = row / (H_out/2) - 1
            x = tracks[..., 0] * (W_out / W_in)
            y = tracks[..., 1] * (H_out / H_in)
            col_norm = x / (W_out * 0.5) - 1.0
            row_norm = y / (H_out * 0.5) - 1.0
            # traj stores (row, col) to match AMPLIFY convention
            traj = np.stack([row_norm, col_norm], axis=-1).astype(np.float32)
            traj = np.clip(traj, -1.0, 1.0)

            # Interpolate to track_pred_horizon if different from true_horizon
            T = traj.shape[1]
            if T != self.track_pred_horizon:
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, T)
                t_new = np.linspace(0, 1, self.track_pred_horizon)
                V, _, N, D = traj.shape
                traj_interp = np.zeros((V, self.track_pred_horizon, N, D), dtype=np.float32)
                for v in range(V):
                    for n in range(N):
                        for d in range(D):
                            f = interp1d(t_old, traj[v, :, n, d])
                            traj_interp[v, :, n, d] = f(t_new)
                traj = traj_interp

            data['traj'] = traj

        # ── vis ──
        if 'vis' in data:
            vis = data['vis']   # (V, T, N, 1)
            if vis.shape[1] != self.track_pred_horizon:
                # Simple nearest-neighbour resize along time axis
                idx = np.round(
                    np.linspace(0, vis.shape[1] - 1, self.track_pred_horizon)
                ).astype(int)
                data['vis'] = vis[:, idx]

        # ── actions: pad to true_horizon ──
        if 'actions' in data:
            acts = data['actions']
            if acts.shape[0] < self.true_horizon:
                pad = np.zeros((self.true_horizon - acts.shape[0], acts.shape[1]), dtype=np.float32)
                data['actions'] = np.concatenate([acts, pad], axis=0)

        return data

    # ── utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_grid_xy(W: int, H: int, N: int) -> np.ndarray:
        """
        Uniform grid of *N* points in pixel (x, y) space for a W×H image.
        Uses a square grid with side ≈ sqrt(N); actual count may differ slightly.
        """
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(W * 0.05, W * 0.95, side)
        ys = np.linspace(H * 0.05, H * 0.95, side)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        # Trim or pad to exactly N
        if len(pts) >= N:
            pts = pts[:N]
        else:
            pad = np.tile(pts, (N // len(pts) + 1, 1))[:N]
            pts = pad
        return pts  # (N, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level LRU cache so the same model_tracks.npy is not reloaded
# for every window within a video.
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _get_cached_frame_dets(path: str):
    return _parse_model_tracks(path)


@lru_cache(maxsize=64)
def _get_cached_mano_joints(path: str) -> np.ndarray:
    """Load ``mano_joints_2d.npy`` once and keep in memory."""
    return np.load(path)  # (N_frames, 2, 16, 2) float32 with NaNs


@lru_cache(maxsize=64)
def _get_cached_cotracker_tracks(path: str) -> np.ndarray:
    """Load ``cotracker_tracks.npy`` once and keep in memory."""
    return np.load(path)  # (N_frames, N, 2) float32


@lru_cache(maxsize=64)
def _get_cached_cotracker_vis(path: str) -> np.ndarray:
    """Load ``cotracker_vis.npy`` once and keep in memory."""
    return np.load(path)  # (N_frames, N) bool
