import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation')
TEX = ROOT / 'exp1113.tex'
NOISED_ROOT = ROOT / 'jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-noised-e200'
PERFILE_ROOT = ROOT / 'jit_cond_dino_two_stage_1122_ddim_recon_noflip_perfile/checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320'


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def metric(path):
    if not path.exists():
        return None
    text = path.read_text(errors='replace')
    if 'Traceback' in text or 'ChildFailedError' in text:
        raise RuntimeError(f"Sampling failed: {path}")
    matches = re.findall(r'FID: ([0-9.]+), Inception Score: ([0-9.]+)', text)
    return tuple(map(float, matches[-1])) if matches else None


def all_metrics():
    return {
        ('noised', scale): metric(NOISED_ROOT / f'eval_checkpoint-160_jitcfg{scale}_dinocfg2.0_dino25_heun50_image50000/sampling.log')
        for scale in ('2.9', '2.1')
    } | {
        ('perfile', scale): metric(PERFILE_ROOT / f'eval_checkpoint-80_jitcfg{scale}_dinocfg2.0_dino25_heun50_image50000/sampling.log')
        for scale in ('2.9', '2.1')
    }


def loss(path, epoch, key):
    pattern = rf'Epoch: \[{epoch}\].*{key}: [0-9.]+ \(([0-9.]+)\)'
    matches = re.findall(pattern, path.read_text(errors='replace'))
    if not matches:
        raise ValueError(f'Missing {key} at epoch {epoch} in {path}')
    return float(matches[-1])


def update_tex(metrics):
    text = TEX.read_text()
    noised_loss = loss(NOISED_ROOT / 'train.log', 159, 'loss_pixel')
    noised_dino = loss(NOISED_ROOT.parent / 'stage1-dinov3b-dinodit-b-e160/train.log', 159, 'loss_dino')
    perfile_loss = loss(PERFILE_ROOT / 'train.log', 79, 'loss_pixel')
    perfile_dino = loss(ROOT / 'jit_cond_dino_two_stage_1122_ddim_recon_noflip/checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e200/train.log', 79, 'loss_dino')

    if '0.0449 & 0.3687' not in text:
        anchor = r'        & & & 2.0;[0,1] & 2.1;[0.1,1] & & & \textbf{4.91} & 267.7 \\'
        rows = '\n'.join([
            anchor,
            rf'        & 160 & 160 & 2.0;[0,1] & 2.9;[0.1,1] & {noised_loss:.4f} & {noised_dino:.4f} & {metrics["noised", "2.9"][0]:.2f} & {metrics["noised", "2.9"][1]:.1f} \\',
            rf'        & & & 2.0;[0,1] & 2.1;[0.1,1] & & & {metrics["noised", "2.1"][0]:.2f} & {metrics["noised", "2.1"][1]:.1f} \\',
        ])
        if anchor not in text:
            raise ValueError('Noised E160-E80 table anchor not found')
        text = text.replace(anchor, rows, 1)
        text = text.replace(r'\multirow{9}{*}{JiTb\_cond\_join\_dinov3b}',
                            r'\multirow{11}{*}{JiTb\_cond\_join\_dinov3b}', 1)

    if 'tab:cached_recon_perfile_results' not in text:
        best = min(metrics['perfile', '2.9'][0], metrics['perfile', '2.1'][0])

        def fid(scale):
            value = metrics['perfile', scale][0]
            return rf'\textbf{{{value:.2f}}}' if value == best else f'{value:.2f}'

        new_table = fr'''\begin{{table}}[t]
    \centering
    \small
    \setlength{{\tabcolsep}}{{3pt}}
    \caption{{Cached-reconstruction Pixel JiT results. Stage 2 uses 25-step Heun-reconstructed DINO patch tokens with DINO CFG 1.0 for offline cache generation and DINO CFG 2.0 for free sampling. $E_d$ and $E_p$ denote DinoDiT and Pixel JiT training epochs, respectively. $t_{{\mathrm{{dino\_pixel}}}}=1$ for cached conditions.}}
    \label{{tab:cached_recon_perfile_results}}
    \resizebox{{1.0\textwidth}}{{!}}{{%
    \begin{{tabular}}{{c c c c c c c c c}}
        \toprule
        Method & E\_D & E\_P & D\_heun25 & P\_heun50 & loss\_jit & loss\_dino & FID $\downarrow$ & IS $\uparrow$ \\
        \midrule
        \multicolumn{{9}}{{@{{}}l}}{{\textit{{c = $t_{{\mathrm{{pixel}}}} + y + 1 + W_g z_{{\mathrm{{cond}}}}$ (25-step reconstructed DINO patch tokens, $t_{{\mathrm{{dino\_pixel}}}}=1$)}}}} \\
        \multirow{{2}}{{*}}{{JiTb\_cached\_recon\_perfile}}
        & 80 & 80 & 2.0;[0,1] & 2.9;[0.1,1] & {perfile_loss:.4f} & {perfile_dino:.4f} & {metrics['perfile', '2.9'][0]:.2f} & {metrics['perfile', '2.9'][1]:.1f} \\
        & & & 2.0;[0,1] & 2.1;[0.1,1] & & & {fid('2.1')} & {metrics['perfile', '2.1'][1]:.1f} \\
        \bottomrule
    \end{{tabular}}%
    }}
\end{{table}}

'''
        marker = '\n\\begin{table}[t]\n    \\centering\n    \\caption{\\textbf{-B and -L comparison'
        if marker not in text:
            raise ValueError('Table 4 insertion anchor not found')
        text = text.replace(marker, '\n' + new_table + marker, 1)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    while True:
        metrics = all_metrics()
        if args.dry_run:
            log(f'Metrics currently available: {metrics}')
            return
        if all(value is not None for value in metrics.values()):
            break
        time.sleep(30)
    log(f'All sampling results are available: {metrics}')
    updated = update_tex(metrics)
    build = ROOT / '.cache/cached_recon_table_build'
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    candidate = build / 'exp1113.tex'
    candidate.write_text(updated)
    source_pdf = TEX.parent / '2026.11.20jit_cond.pdf'
    if source_pdf.exists():
        shutil.copy2(source_pdf, build / source_pdf.name)
    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                    '-output-directory', str(build), str(candidate)],
                   cwd=build, check=True, stdout=subprocess.DEVNULL)
    shutil.copy2(candidate, TEX)
    shutil.rmtree(build)
    log(f'Updated {TEX}; LaTeX compilation passed')


if __name__ == '__main__':
    main()
