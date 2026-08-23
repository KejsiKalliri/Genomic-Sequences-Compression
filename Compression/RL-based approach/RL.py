import pickle
import numpy as np
from typing import Any
from pathlib import Path
from C_core import compress_block, decompress_block
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------- Features extraction ------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def build_state(
    current_window_feats: list[float],
    next_window_feats: list[float] | None,
    next2_window_feats: list[float] | None,
    next3_window_feats: list[float] | None,
    current_position_norm: float,
    remaining_fraction: float,
) -> np.ndarray:

    current = np.array(current_window_feats, dtype=np.float32)

    next1 = (
        np.zeros_like(current)
        if next_window_feats is None
        else np.array(next_window_feats, dtype=np.float32)
    )

    next2 = (
        np.zeros_like(current)
        if next2_window_feats is None
        else np.array(next2_window_feats, dtype=np.float32)
    )

    next3 = (
        np.zeros_like(current)
        if next3_window_feats is None
        else np.array(next3_window_feats, dtype=np.float32)
    )

    # Structural differences between consecutive future windows
    diff_next1 = next1 - current
    diff_next2 = next2 - next1
    diff_next3 = next3 - next2

    # Dinucleotide similarities
    cur_di = current[6:22]
    next1_di = next1[6:22]
    next2_di = next2[6:22]
    next3_di = next3[6:22]

    sim_current_next1 = cosine_similarity(cur_di, next1_di)
    sim_next1_next2 = cosine_similarity(next1_di, next2_di)
    sim_next2_next3 = cosine_similarity(next2_di, next3_di)

    scalar_features = np.array([
        sim_current_next1,
        sim_next1_next2,
        sim_next2_next3,
        current_position_norm,
        remaining_fraction,
    ], dtype=np.float32)

    state = np.concatenate(
        [
            current,
            next1,
            diff_next1,
            diff_next2,
            diff_next3,
            scalar_features,
        ],
        axis=0,
    )

    return state.astype(np.float32)

# ---------------------------- Cache ------------------------------------

def load_feature_cache(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------- Compression Reward ------------------------

def compress_segment_real(segment_seq: str, level: int = 9) -> tuple[int, int]:
    raw = segment_seq.encode("ascii")
    comp = compress_block(raw, level=level)
    return len(raw), len(comp)

def segment_reward_real(segment_seq: str, level: int = 9, segment_overhead: int = 12,  cut_penalty: float = 0.75) -> float:

    raw_len, comp_len = compress_segment_real(segment_seq, level=level)
    effective_comp = comp_len + segment_overhead

    if raw_len == 0:
        return -1.0

    compression_gain = (raw_len - effective_comp) / raw_len
    return float(compression_gain - cut_penalty)

# --------------------------------- REPLAY BUFFER ------------------------------

class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 1e-4,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.eps = eps

        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        item = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = priorities ** self.alpha
        probs_sum = probs.sum()

        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        idxs = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in idxs]

        self.beta = min(1.0, self.beta + self.beta_increment)

        weights = (len(self.buffer) * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        s, a, r, ns, d = zip(*batch)

        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns).astype(np.float32),
            np.array(d, dtype=np.float32),
            idxs,
            weights,
        )

    def update_priorities(self, idxs, td_errors):
        td_errors = np.abs(td_errors) + self.eps

        for idx, err in zip(idxs, td_errors):
            self.priorities[idx] = float(err)

    def __len__(self):
        return len(self.buffer)

# ------------------------------------ DQN AGENT ----------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class DQNAgentTorch:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 1,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.98,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_every: int = 250,
        device: str | None = None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=30,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=1e-6,
        )

        self.current_lr = lr
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.train_steps = 0


    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if (not deterministic) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(0, self.action_dim))

        with torch.no_grad():   # exploitation
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(state_t)
            return int(torch.argmax(q, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)


    def step_scheduler(self, avg_loss: float, num_updates: int) -> None:
        if num_updates <= 0:
            return

        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(avg_loss)
        new_lr = self.optimizer.param_groups[0]["lr"]
        self.current_lr = new_lr

        if new_lr != old_lr:
            print(f"[lr] learning rate changed: {old_lr:.6e} -> {new_lr:.6e}")

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, idxs, weights = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)

            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        td_errors = targets - q_values

        loss = (weights_t * torch.nn.functional.smooth_l1_loss(q_values, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.buffer.update_priorities(idxs, td_errors.detach().abs().cpu().numpy(),)

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "action_dim": self.action_dim,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: str | None = None,
    ) -> "DQNAgentTorch":

        agent = cls(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, device=device)
        ckpt = torch.load(path, map_location=agent.device)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.epsilon = ckpt.get("epsilon", 0.05)
        return agent

# -----------------------------Segmentation-------------------------------
class SegmentLengthEnv:
    def __init__(
        self,
        sequence: str,
        window_features: list[list[float]],
        segment_lengths: list[int],
        window_size: int = 1000,
        level: int = 9,
    ):

        self.sequence = sequence
        self.window_features = window_features
        self.segment_lengths = segment_lengths
        self.window_size = window_size
        self.level = level

        self.windows = [sequence[i:i + window_size] for i in range(0, len(sequence), window_size)]

        if not self.window_features:
            raise ValueError("No window features provided")

        self.action_dim = len(segment_lengths)

        sample_state = self._get_state(0)
        self.state_dim = len(sample_state)

        self.reset()

    def reset(self) -> np.ndarray:
        self.pos_bp = 0
        self.done = False
        self.segments: list[tuple[int, int]] = []
        return self._get_state(self.pos_bp)

    def _get_window_idx(self, pos_bp: int) -> int:
        idx = pos_bp // self.window_size
        return min(idx, len(self.window_features) - 1)


    def _get_state(self, pos_bp: int) -> np.ndarray:
        idx = self._get_window_idx(pos_bp)

        current_feats = self.window_features[idx]

        next1_feats = (
            self.window_features[idx + 1]
            if idx + 1 < len(self.window_features)
            else None
        )

        next2_feats = (
            self.window_features[idx + 2]
            if idx + 2 < len(self.window_features)
            else None
        )

        next3_feats = (
            self.window_features[idx + 3]
            if idx + 3 < len(self.window_features)
            else None
        )

        current_position_norm = pos_bp / max(1, len(self.sequence))
        remaining_fraction = max(0.0, (len(self.sequence) - pos_bp) / max(1, len(self.sequence)))

        return build_state(
            current_window_feats=current_feats,
            next_window_feats=next1_feats,
            next2_window_feats=next2_feats,
            next3_window_feats=next3_feats,
            current_position_norm=current_position_norm,
            remaining_fraction=remaining_fraction,
        )

    def step(self, action: int, compute_reward: bool = True) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already done")

        seg_len = self.segment_lengths[action]  # action 0,1,2,3 -> 200k,400k,600k,800k

        start_bp = self.pos_bp
        end_bp = min(start_bp + seg_len, len(self.sequence))
        end_bp = (end_bp // self.window_size) * self.window_size
        if end_bp <= start_bp:
            end_bp = min(start_bp + self.window_size, len(self.sequence))

        if compute_reward:
            segment_seq = self.sequence[start_bp:end_bp]
            reward = segment_reward_real(segment_seq, level=self.level)
        else:
            reward = 0.0

        self.segments.append((start_bp, end_bp))
        self.pos_bp = end_bp

        if self.pos_bp >= len(self.sequence):
            self.done = True
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            return next_state, reward, True, {"segments": self.segments[:]}

        next_state = self._get_state(self.pos_bp)
        return next_state, reward, False, {}


def rollout_segmentation(
    sequence: str,
    window_features: list[list[float]],
    agent: DQNAgentTorch,
    segment_lengths: list[int],
    window_size: int = 1000,
    level: int = 9,
) -> list[tuple[int, int]]:

    env = SegmentLengthEnv(
        sequence=sequence,
        window_features=window_features,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
    )

    state = env.reset()
    done = False

    action_counts = {seg_len: 0 for seg_len in segment_lengths}

    while not done:
        action = agent.select_action(state, deterministic=True)

        chosen_seg_len = segment_lengths[action]
        action_counts[chosen_seg_len] += 1

        state, _reward, done, info = env.step(action, compute_reward = False)  # Training is done. So it's not a problem to use compute_reward = false

    return info.get("segments", [])


