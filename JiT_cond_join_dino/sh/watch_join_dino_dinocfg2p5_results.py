#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT.parent / "exp1113.tex"
EXPERIMENT = "JiT-B16-JoinDino-dinov3b-B200"
CHECKPOINT_ROOT = ROOT / "checkpoints" / EXPERIMENT
TRAIN_LOG = CHECKPOINT_ROOT / "train.log"
WATCH_LOG = CHECKPOINT_ROOT / "eval_dinocfg2p5_e20-100_queue.log"
FINAL_MARKER = "DINO CFG 2.5 五组采样全部完成。"
EPOCHS = (20, 40, 60, 80, 100)
COMMENTED_EPOCHS = (120, 140)
SCALES = ("1.0", "1.5", "2.0", "2.5")
LOGS = {
    "1.0": TRAIN_LOG,
    "1.5": CHECKPOINT_ROOT / "eval_dinocfg1p5_e20-100_queue.log",
    "2.0": CHECKPOINT_ROOT / "eval_dinocfg2p0_e20-100_queue.log",
    "2.5": WATCH_LOG,
}


def read_text(path):
    return path.read_text(errors="replace") if path.exists() else ""


def parse_metrics(path):
    results = {}
    current_epoch = None
    for line in read_text(path).splitlines():
        match = re.search(r"Epoch: \[(\d+)\] Total time:", line)
        if match:
            current_epoch = int(match.group(1)) + 1
        match = re.search(r"Evaluating checkpoint at (\d+) epoch", line)
        if match:
            current_epoch = int(match.group(1))
        match = re.search(r"开始 epoch (\d+)", line)
        if match:
            current_epoch = int(match.group(1))
        match = re.search(r"FID: ([0-9.]+), Inception Score: ([0-9.]+)", line)
        if match and current_epoch is not None:
            results[current_epoch] = (float(match.group(1)), float(match.group(2)))
    return results


def parse_losses():
    losses = {}
    pattern = re.compile(
        r"Epoch: \[(\d+)\]\s+\[1250/1251\].*?"
        r"loss_jit: [0-9.]+ \(([0-9.]+)\)\s+"
        r"loss_dino: [0-9.]+ \(([0-9.]+)\)"
    )
    for line in read_text(TRAIN_LOG).splitlines():
        match = pattern.search(line)
        if match:
            losses[int(match.group(1)) + 1] = (float(match.group(2)), float(match.group(3)))
    return losses


def build_tabular(metrics, losses):
    row_count = sum(epoch in metrics[scale] for epoch in EPOCHS for scale in SCALES)
    if row_count == 0:
        raise RuntimeError("No completed metrics found")
    lines = [
        r"    \begin{tabular}{c c c c c c c c}",
        r"        \toprule",
        r"        Method & Epoch & DINO & JiT & loss\_jit & loss\_dino & FID $\downarrow$ & IS $\uparrow$ \\",
        r"        \midrule",
        r"        \multicolumn{8}{@{}l}{\textit{c = $t + y + t_{\mathrm{dino}} + W_g z_{\mathrm{cond}}$ (DINOv3-B/16 patch tokens)}} \\",
        rf"        \multirow{{{row_count}}}{{*}}{{Patch-level AdaLN}}",
    ]
    for epoch in EPOCHS:
        first_in_group = True
        for scale in SCALES:
            if epoch not in metrics[scale]:
                continue
            fid, inception_score = metrics[scale][epoch]
            if first_in_group:
                loss_jit, loss_dino = losses[epoch]
                lines.append(
                    f"        & {epoch} & {scale};[0,1] & 2.9;[0.1,1] & "
                    f"{loss_jit:.4f} & {loss_dino:.4f} & {fid:.2f} & {inception_score:.1f} \\\\"
                )
                first_in_group = False
            else:
                lines.append(
                    f"        & & {scale};[0,1] & 2.9;[0.1,1] & & & "
                    f"{fid:.2f} & {inception_score:.1f} \\\\"
                )
        lines.append(r"        \hdashline")
    for epoch in COMMENTED_EPOCHS:
        loss_jit, loss_dino = losses[epoch]
        fid, inception_score = metrics["1.0"][epoch]
        lines.append(
            f"        % & {epoch} & 1.0;[0,1] & 2.9;[0.1,1] & "
            f"{loss_jit:.4f} & {loss_dino:.4f} & {fid:.2f} & {inception_score:.1f} \\\\"
        )
        lines.append(r"        % \hdashline")
    lines.extend([r"        \bottomrule", r"    \end{tabular}%"])
    return "\n".join(lines), row_count


def replace_tabular(text, tabular):
    label = r"\label{tab:join_dino_dinov3b_results}"
    label_pos = text.index(label)
    table_start = text.rfind(r"\begin{table}", 0, label_pos)
    table_end = text.index(r"\end{table}", label_pos)
    table = text[table_start:table_end]
    tabular_start = table.index(r"    \begin{tabular}")
    end_marker = r"    \end{tabular}%"
    tabular_end = table.index(end_marker) + len(end_marker)
    new_table = table[:tabular_start] + tabular + table[tabular_end:]
    return text[:table_start] + new_table + text[table_end:]


def update_tex():
    metrics = {scale: parse_metrics(path) for scale, path in LOGS.items()}
    tabular, row_count = build_tabular(metrics, parse_losses())
    text = TEX_PATH.read_text()
    updated = replace_tabular(text, tabular)
    if updated == text:
        return row_count, False
    temporary = TEX_PATH.with_suffix(".tex.autofill.tmp")
    temporary.write_text(updated)
    os.replace(temporary, TEX_PATH)
    return row_count, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll", type=float, default=20.0)
    args = parser.parse_args()

    row_count, changed = update_tex()
    print(f"Table 2 rows={row_count}, updated={changed}", flush=True)
    if args.once:
        return
    print(f"Watching {WATCH_LOG}", flush=True)
    while True:
        log_text = read_text(WATCH_LOG)
        if FINAL_MARKER in log_text:
            row_count, changed = update_tex()
            print(f"Sampling complete; final rows={row_count}, updated={changed}", flush=True)
            return
        if "采样队列失败" in log_text:
            raise RuntimeError("CFG 2.5 sampling queue failed; see its queue log")
        time.sleep(args.poll)
        row_count, changed = update_tex()
        if changed:
            print(f"[{time.strftime('%F %T')}] Table 2 updated, rows={row_count}", flush=True)


if __name__ == "__main__":
    main()
