from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any
import numpy as np

import json
from collections import Counter
import matplotlib.pyplot as plt
from RL import SegmentLengthEnv, DQNAgentTorch
from feature_cache import build_window_feature_cache

DEFAULT_SEGMENT_LENGTHS = [200_000, 400_000, 600_000, 800_000]
CHUNK_THRESHOLD_BP = 9_000_000
TRAIN_CHUNK_BP = 5_000_000


# ---------------------------- FASTA folder reader ----------------------------

def collect_fasta_files_from_folder(folder: str | Path) -> list[str]:
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    fasta_extensions = {".fa", ".fasta", ".fna", ".fas"}

    fasta_files = [
        str(p)
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in fasta_extensions
    ]

    fasta_files.sort()

    if not fasta_files:
        raise ValueError(f"No FASTA files found in folder: {folder}")

    print(f"[folder] found {len(fasta_files)} FASTA files:")
    for f in fasta_files:
        print(f"  - {f}")

    return fasta_files


# ---------------------------- FASTA multi-sequence reader ----------------------------

def read_fasta_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    current_header: str | None = None
    current_seq_parts: list[str] = []
    current_line_width = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    sequence = "".join(current_seq_parts).upper()
                    if sequence:
                        records.append(
                            {
                                "source": str(path),
                                "header": current_header,
                                "sequence": sequence,
                                "line_width": current_line_width if current_line_width > 0 else 60,
                                "length": len(sequence),
                            }
                        )

                current_header = line
                current_seq_parts = []
                current_line_width = 0

            else:
                if current_line_width == 0:
                    current_line_width = len(line)
                current_seq_parts.append(line)

    if current_header is not None:
        sequence = "".join(current_seq_parts).upper()
        if sequence:
            records.append(
                {
                    "source": str(path),
                    "header": current_header,
                    "sequence": sequence,
                    "line_width": current_line_width if current_line_width > 0 else 60,
                    "length": len(sequence),
                }
            )

    return records


def load_training_records(fasta_paths: list[str]) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []

    for fasta in fasta_paths:
        records = read_fasta_records(fasta)
        print(f"[load] {fasta}: {len(records)} records")

        for r in records:
            if r["length"] > 0:
                all_records.append(r)

    if not all_records:
        raise ValueError("No valid FASTA records found")

    total_bp = sum(r["length"] for r in all_records)
    print(f"[load] total records: {len(all_records)}")
    print(f"[load] total bp: {total_bp:,}")

    return all_records


# ---------------------------- Chunk sampling ----------------------------

def sample_sequence_chunk(sequence: str, window_size: int, train_chunk_bp: int,) -> tuple[str, int]:

    total_len = len(sequence)

    if total_len <= train_chunk_bp:
        return sequence, 0

    max_start = total_len - train_chunk_bp

    start_bp = random.randint(0, max_start)
    start_bp = (start_bp // window_size) * window_size

    end_bp = min(start_bp + train_chunk_bp, total_len)
    end_bp = (end_bp // window_size) * window_size

    if end_bp <= start_bp:
        end_bp = min(start_bp + window_size, total_len)

    return sequence[start_bp:end_bp], start_bp


def choose_record_weighted_by_length(records: list[dict[str, Any]]) -> dict[str, Any]:

    if random.random() < 0.5:
        # uniform sampling
        return random.choice(records)
    else:
        # weighted sampling
        lengths = [max(1, r["length"]) for r in records]
        return random.choices(records, weights=lengths, k=1)[0]



# ---------------------------- Pretraining ----------------------------

def build_warm_env(
    records: list[dict[str, Any]],
    segment_lengths: list[int],
    window_size: int,
    level: int,
    train_chunk_bp: int,
) -> SegmentLengthEnv:

    record = choose_record_weighted_by_length(records)
    chunk_seq, _ = sample_sequence_chunk(   # is returned either the whole sequence or only a chunk of it
        sequence=record["sequence"],
        window_size=window_size,
        train_chunk_bp=train_chunk_bp,
    )

    chunk_features = build_window_feature_cache(chunk_seq, window_size=window_size,)

    return SegmentLengthEnv(
        sequence=chunk_seq,
        window_features=chunk_features,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
    )


def pretrain_dqn_on_genomes(
    records: list[dict[str, Any]],
    segment_lengths: list[int],
    episodes: int = 1000,
    window_size: int = 10000,
    level: int = 9,
    train_chunk_bp: int = TRAIN_CHUNK_BP,
    device: str | None = None,
    plots_out_dir: str | Path = "training_plots",
    log_action_dist_every: int = 20,
) -> DQNAgentTorch:

    warm_env = build_warm_env(
        records=records,
        segment_lengths=segment_lengths,
        window_size=window_size,
        level=level,
        train_chunk_bp=train_chunk_bp,
    )

    agent = DQNAgentTorch(
        state_dim=warm_env.state_dim,
        action_dim=warm_env.action_dim,
        device=device,
    )

    print(f"[model] state_dim={warm_env.state_dim}")
    print(f"[model] action_dim={warm_env.action_dim}")
    print(f"[model] segment_lengths={segment_lengths}")

    history = init_training_history()
    plots_out_dir = Path(plots_out_dir)
    plots_out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        record = choose_record_weighted_by_length(records)

        chunk_seq, chunk_start = sample_sequence_chunk(
            sequence=record["sequence"],
            window_size=window_size,
            train_chunk_bp=train_chunk_bp,
        )

        feature_start = time.time()
        chunk_features = build_window_feature_cache(chunk_seq, window_size=window_size)
        feature_time = time.time() - feature_start

        env = SegmentLengthEnv(
            sequence=chunk_seq,
            window_features=chunk_features,
            segment_lengths=segment_lengths,
            window_size=window_size,
            level=level,
        )

        state = env.reset()
        done = False

        # ---------------- Episode-level statistics ----------------
        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0
        episode_action_counts = Counter()

        total_loss = 0.0
        num_updates = 0

        # human-readable counts for printing
        action_counts = {seg_len: 0 for seg_len in segment_lengths}

        while not done:   # in every episode there is done a full segmentation
            action = agent.select_action(state)  # deterministic=False

            # log action counts by action index
            episode_action_counts[action] += 1
            history["action_counts_total"][action] += 1

            # human-readable count by segment length
            chosen_seg_len = segment_lengths[action]
            action_counts[chosen_seg_len] += 1

            next_state, reward, done, _info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                total_loss += float(loss)
                num_updates += 1
                episode_losses.append(float(loss))

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1

        # ---------------- End of episode updates ----------------
        agent.decay_epsilon()

        avg_loss = total_loss / max(1, num_updates)
        agent.step_scheduler(avg_loss, num_updates)

        current_lr = agent.current_lr

        # ---------------- Save history ----------------
        history["episodes"].append(ep + 1)
        history["rewards"].append(float(episode_reward))
        history["epsilons"].append(float(agent.epsilon))
        history["learning_rates"].append(float(current_lr))
        history["avg_losses"].append(float(avg_loss))
        history["steps"].append(int(episode_steps))
        history["action_counts_per_episode"].append(dict(episode_action_counts))

        print(
            f"[pretrain] episode={ep + 1}/{episodes} "
            f"source={Path(record['source']).name} "
            f"record={record['header'][:40]} "
            f"chunk_start={chunk_start:,} "
            f"chunk_len={len(chunk_seq):,} "
            f"steps={episode_steps} "
            f"reward={episode_reward:.4f} "
            f"avg_loss={avg_loss:.6f} "
            f"epsilon={agent.epsilon:.4f} "
            f"lr={current_lr:.6e} "
            f"actions={action_counts} "
            f"feature_time={feature_time:.2f}s"
        )

        # ---------------- Optional cumulative action distribution log ----------------
        if (ep + 1) % log_action_dist_every == 0:
            total_actions = sum(history["action_counts_total"].values())
            if total_actions > 0:
                action_dist_parts = []
                for action_idx, seg_len in enumerate(segment_lengths):
                    count = history["action_counts_total"].get(action_idx, 0)
                    pct = 100.0 * count / total_actions
                    action_dist_parts.append(f"{seg_len}:{count} ({pct:.1f}%)")

                print(
                    f"[action_dist] episode={ep + 1} "
                    + " | ".join(action_dist_parts)
                )

    # ---------------- Save history + plots after training ----------------
    save_training_history(history, plots_out_dir)
    plot_training_history(
        history,
        plots_out_dir,
        action_labels=[str(x) for x in segment_lengths],
    )

    print(f"[plots] saved to: {plots_out_dir}")


    agent.training_history = history

    return agent  # trained agent


# ---------------------------- CLI ----------------------------
def init_training_history():
    return {
        "episodes": [],
        "rewards": [],
        "epsilons": [],
        "learning_rates": [],
        "avg_losses": [],
        "steps": [],
        "action_counts_total": Counter(),
        "action_counts_per_episode": []
    }


def save_training_history(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable_history = {
        "episodes": history["episodes"],
        "rewards": history["rewards"],
        "epsilons": history["epsilons"],
        "learning_rates": history["learning_rates"],
        "avg_losses": history["avg_losses"],
        "steps": history["steps"],
        "action_counts_total": dict(history["action_counts_total"]),
        "action_counts_per_episode": history["action_counts_per_episode"],
    }

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)


def moving_average(values, window=20):
    if not values:
        return []
    result = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        if i >= window:
            running_sum -= values[i - window]
        result.append(running_sum / min(i + 1, window))
    return result


def plot_training_history(history, out_dir, action_labels=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = history["episodes"]
    rewards = history["rewards"]
    epsilons = history["epsilons"]
    learning_rates = history["learning_rates"]
    action_counts_total = history["action_counts_total"]
    avg_losses = history["avg_losses"]

    # 1) Reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Episode reward")
    if len(rewards) >= 2:
        plt.plot(episodes, moving_average(rewards, window=20), label="Moving average (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png")
    plt.close()

    # 2) Epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_decay.png")
    plt.close()

    # 3) Learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, learning_rates)
    plt.xlabel("Episode")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Episodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate.png")
    plt.close()

    # 4) Action histogram
    if action_labels is None:
        max_action = max(action_counts_total.keys(), default=-1)
        action_labels = [str(i) for i in range(max_action + 1)]

    counts = [action_counts_total.get(i, 0) for i in range(len(action_labels))]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(action_labels)), counts)
    plt.xticks(range(len(action_labels)), action_labels, rotation=45)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Selection Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "action_histogram.png")
    plt.close()

    # 5) Average Loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Average Loss over Episodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "avg_losses.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Pretrain DQN segment-length policy on chromosome FASTA files")

    parser.add_argument("--fasta-folder", required=True, help="Folder containing chromosome FASTA files",)

    parser.add_argument("--model-out", required=True, help="Output pretrained model path, e.g. pretrained_dqn_seglen.pt",)

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--train-chunk-bp", type=int, default=TRAIN_CHUNK_BP)
    parser.add_argument("--segment-lengths", type=int, nargs="+", default=DEFAULT_SEGMENT_LENGTHS,)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fasta_files = collect_fasta_files_from_folder(args.fasta_folder)
    records = load_training_records(fasta_files)

    start = time.time()

    agent = pretrain_dqn_on_genomes(
        records=records,
        segment_lengths=args.segment_lengths,
        episodes=args.episodes,
        window_size=args.window_size,
        level=args.level,
        train_chunk_bp=args.train_chunk_bp,
        device=args.device,
    )

    elapsed = time.time() - start

    agent.save(args.model_out)

    print(f"[done] pretrained model saved to: {args.model_out}")
    print(f"[done] total pretraining time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

# py pretrain_model.py --fasta-folder "../Genomes/Genomes" --model-out pretrained_dqn_model_4.pt --episodes 1000 --window-size 10000 --level 9