#!/usr/bin/env python3
"""15-minute monitor/orchestrator for the AVerImaTeC VILLAIN sweep."""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/aied_test/VILLAIN')
PYTHON = sys.executable
CHECK_SECONDS = 15 * 60
GPUS = [0, 1, 2]
SHARDS = [(0, 51), (51, 102), (102, 152)]
GENERATION_STALE_SECONDS = 60 * 60
EVAL_STALE_SECONDS = 60 * 60
STATE_PATH = ROOT / 'outputs-qwen25-cot-sweep-monitor-state.json'
LOG_PATH = ROOT / 'outputs-qwen25-cot-sweep-monitor.log'

BASELINE = {
    'name': 'baseline',
    'output': 'outputs-qwen25-cot-json-full-budget20',
    'config': 'scripts/cfg/qwen25-cot-json-full-budget20.yaml',
    'kind': 'existing_full',
}
RUNS = [
    BASELINE,
    {
        'name': 'e_vj_supp',
        'output': 'outputs-qwen25-cot-parammute-l12-e-vj-supp-budget20',
        'config': 'scripts/cfg/qwen25-cot-parammute-l12-e-vj-supp-budget20.yaml',
        'kind': 'full',
    },
    {
        'name': 'e_supp',
        'output': 'outputs-qwen25-cot-parammute-l12-e-supp-budget20',
        'config': 'scripts/cfg/qwen25-cot-parammute-l12-e-supp-budget20.yaml',
        'kind': 'agent5',
        'source': 'outputs-qwen25-cot-parammute-l12-e-vj-supp-budget20',
        'suppression_strength': '1.0',
        'suppression_layers': '',
    },
    {
        'name': 'vj_supp',
        'output': 'outputs-qwen25-cot-parammute-l12-vj-supp-budget20',
        'config': 'scripts/cfg/qwen25-cot-parammute-l12-vj-supp-budget20.yaml',
        'kind': 'agent5',
        'source': 'outputs-qwen25-cot-json-full-budget20',
        'suppression_strength': '0.5',
        'suppression_layers': '12',
    },
]
RUN_BY_NAME = {r['name']: r for r in RUNS}
ORDER = ['baseline', 'e_vj_supp', 'e_supp', 'vj_supp']


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG_PATH.open('a') as f:
        f.write(line + '\n')


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {'baseline_low_attempts': 0, 'launched': {}, 'eval_launched': {}, 'accepted': {}, 'done': False}


def save_state(state):
    tmp = STATE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def output_path(run):
    return ROOT / run['output']


def pids_file(run):
    return output_path(run) / 'pids.txt'


def eval_dir(run):
    return output_path(run) / 'eval_openrouter_gemini25flash'


def eval_output(run):
    return eval_dir(run) / 'eval_results_local.json'


def eval_pid_file(run):
    return eval_dir(run) / 'eval.pid'


def pid_alive(pid):
    try:
        stat_path = Path(f'/proc/{int(pid)}/stat')
        if not stat_path.exists():
            return False
        fields = stat_path.read_text().split()
        return len(fields) > 2 and fields[2] != 'Z'
    except Exception:
        return False


def active_pids(path):
    if not path.exists():
        return []
    active = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and pid_alive(parts[0]):
            active.append(parts[0])
    return active


def kill_pid(pid):
    try:
        os.kill(int(pid), 15)
        return True
    except ProcessLookupError:
        return False
    except Exception as exc:
        log(f'failed to terminate pid={pid}: {exc}')
        return False


def newest_log_age_seconds(directory):
    directory = Path(directory)
    if not directory.exists():
        return None
    files = [p for p in directory.glob('*.log') if p.is_file()]
    if not files:
        return None
    newest = max(p.stat().st_mtime for p in files)
    return time.time() - newest


def eval_log_age_seconds(run):
    p = eval_dir(run) / 'eval.log'
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def complete_count(run):
    sub = output_path(run) / 'submission'
    if not sub.exists():
        return 0
    return sum(1 for d in sub.iterdir() if d.is_dir() and d.name.isdigit() and (d / 'submission.json').exists())


def generation_complete(run):
    return complete_count(run) >= 152


def summarize_generation(run):
    sub = output_path(run) / 'submission'
    counts = {'complete': complete_count(run), 'agent4': 0, 'agent5': 0, 'zero_qa': [], 'qa_counts': [], 'bad4': [], 'bad5': []}
    if not sub.exists():
        return counts
    for d in sub.iterdir():
        if not (d.is_dir() and d.name.isdigit()):
            continue
        cid = int(d.name)
        a4 = d / 'agent4_qa_generation.json'
        a5 = d / 'agent5_verdict.json'
        if a4.exists():
            counts['agent4'] += 1
            try:
                x = json.loads(a4.read_text())
                meta = x.get('metadata') or {}
                q = meta.get('qa_pairs') or meta.get('all_qa_pairs') or x.get('all_qa_pairs') or x.get('qa_pairs') or []
                counts['qa_counts'].append(len(q))
                if len(q) == 0:
                    counts['zero_qa'].append(cid)
                if x.get('parse_error') or meta.get('parse_error'):
                    counts['bad4'].append(cid)
            except Exception:
                counts['bad4'].append(cid)
        if a5.exists():
            counts['agent5'] += 1
            try:
                x = json.loads(a5.read_text())
                meta = x.get('metadata') or {}
                if x.get('parse_error') or meta.get('parse_error'):
                    counts['bad5'].append(cid)
            except Exception:
                counts['bad5'].append(cid)
    return counts


def merge(run):
    out = output_path(run)
    subprocess.run([PYTHON, 'src/utils/merge_output.py', '--input_dir', str(out), '--output_dir', str(out)], cwd=ROOT, check=True)


def launch_full(run):
    out = output_path(run)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    records = []
    for gpu, (start, end) in zip(GPUS, SHARDS):
        log_file = out / 'logs' / f'shard_{start}_{end}.log'
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['TOKENIZERS_PARALLELISM'] = 'false'
        env['PYTHONUNBUFFERED'] = '1'
        fh = log_file.open('a')
        proc = subprocess.Popen([
            PYTHON, 'src/run_multi_agent.py',
            '--config', run['config'],
            '--output_dir', run['output'],
            '--start_idx', str(start),
            '--end_idx', str(end),
            '--device', 'cuda:0',
        ], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        records.append(f'{proc.pid} {gpu} {start} {end} {log_file}')
    pids_file(run).write_text('\n'.join(records) + '\n')
    log(f"launched full run {run['name']} pids={records}")


def launch_agent5(run):
    out = output_path(run)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    records = []
    for gpu, (start, end) in zip(GPUS, SHARDS):
        log_file = out / 'logs' / f'agent5_shard_{start}_{end}.log'
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['TOKENIZERS_PARALLELISM'] = 'false'
        env['PYTHONUNBUFFERED'] = '1'
        cmd = [
            PYTHON, 'src/run_agent5_from_agent4.py',
            '--config', run['config'],
            '--source_output_dir', run['source'],
            '--output_dir', run['output'],
            '--start_idx', str(start),
            '--end_idx', str(end),
            '--device', 'cuda:0',
            '--suppression_strength', run['suppression_strength'],
        ]
        if run['suppression_layers']:
            cmd += ['--suppression_layers', run['suppression_layers']]
        fh = log_file.open('a')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        records.append(f'{proc.pid} {gpu} {start} {end} {log_file}')
    pids_file(run).write_text('\n'.join(records) + '\n')
    log(f"launched agent5 run {run['name']} pids={records}")


def launch_generation(run):
    if run['kind'] in {'full', 'existing_full'}:
        launch_full(run)
    elif run['kind'] == 'agent5':
        launch_agent5(run)
    else:
        raise ValueError(run['kind'])


def launch_eval(run):
    if not os.environ.get('OPENROUTER_API_KEY'):
        log('OPENROUTER_API_KEY missing; cannot launch eval')
        return False
    out = output_path(run)
    if not (out / 'submission.json').exists():
        merge(run)
    ed = eval_dir(run)
    ed.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONUNBUFFERED'] = '1'
    log_file = ed / 'eval.log'
    fh = log_file.open('a')
    proc = subprocess.Popen([
        PYTHON, 'src/evaluation/eval_official_standalone.py',
        '--submission_path', str(out / 'submission.json'),
        '--ground_truth_path', 'dataset/AVerImaTeC/val.json',
        '--image_dir', 'dataset/AVerImaTeC/images',
        '--output_path', str(eval_output(run)),
        '--eval_model', 'google/gemini-2.5-flash',
        '--justification',
    ], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    eval_pid_file(run).write_text(str(proc.pid) + '\n')
    log(f"launched eval {run['name']} pid={proc.pid}")
    return True


def eval_active(run):
    pf = eval_pid_file(run)
    return pf.exists() and pid_alive(pf.read_text().strip())


def eval_scores(run):
    p = eval_output(run)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get('component_scores')


def diagnose_and_maybe_rerun_baseline(state):
    run = BASELINE
    summary = summarize_generation(run)
    attempts = int(state.get('baseline_low_attempts', 0))
    diag_path = output_path(run) / f'baseline_low_evidence_diagnostic_attempt{attempts + 1}.json'
    low_ids = sorted(set(summary['zero_qa'] + [cid for cid, q in []]))
    # Include claims with fewer than 10 QA pairs as a second-pass target.
    sub = output_path(run) / 'submission'
    low_qa = []
    if sub.exists():
        for d in sub.iterdir():
            if d.is_dir() and d.name.isdigit():
                a4 = d / 'agent4_qa_generation.json'
                if a4.exists():
                    try:
                        x = json.loads(a4.read_text())
                        meta = x.get('metadata') or {}
                        q = meta.get('qa_pairs') or meta.get('all_qa_pairs') or x.get('all_qa_pairs') or x.get('qa_pairs') or []
                        if len(q) < 10:
                            low_qa.append(int(d.name))
                    except Exception:
                        low_qa.append(int(d.name))
    target_ids = sorted(set(summary['zero_qa'] if attempts == 0 else low_qa))
    diag = {'attempt': attempts + 1, 'summary': summary, 'target_rerun_claim_ids': target_ids}
    diag_path.write_text(json.dumps(diag, indent=2))
    log(f'baseline evidence <0.1 diagnostic attempt {attempts + 1}: target_ids={target_ids[:30]} count={len(target_ids)}')
    state['baseline_low_attempts'] = attempts + 1
    if target_ids:
        for cid in target_ids:
            shutil.rmtree(output_path(run) / 'submission' / str(cid), ignore_errors=True)
        for p in [output_path(run) / 'submission.json', eval_output(run), eval_pid_file(run)]:
            if p.exists():
                p.unlink()
        pids_file(run).unlink(missing_ok=True)
        launch_full(run)
        state['eval_launched'].pop(run['name'], None)
        state['accepted'].pop(run['name'], None)
        log(f'rerunning baseline target claims by relaunching shards; existing complete claims will be skipped')
    else:
        log('no rerunnable zero/low-QA targets found for baseline; will continue after max diagnostics')
    save_state(state)


def maybe_handle_run(state, run):
    name = run['name']
    out = output_path(run)
    out.mkdir(parents=True, exist_ok=True)

    if generation_complete(run):
        summary = summarize_generation(run)
        qa = summary['qa_counts']
        qa_text = f"qa_mean={sum(qa)/len(qa):.2f} zero={summary['zero_qa']}" if qa else 'qa_mean=n/a'
        if not (out / 'submission.json').exists():
            merge(run)
            log(f"merged {name}: complete={summary['complete']} {qa_text} bad4={summary['bad4']} bad5={summary['bad5']}")
        if eval_scores(run) is None:
            if eval_active(run):
                age = eval_log_age_seconds(run)
                if age is not None and age > EVAL_STALE_SECONDS:
                    pid = eval_pid_file(run).read_text().strip()
                    log(f"eval stale for {name}: pid={pid} log_age={age/60:.1f}m; terminating and relaunching")
                    kill_pid(pid)
                    eval_pid_file(run).unlink(missing_ok=True)
                    launch_eval(run)
                else:
                    age_text = 'unknown' if age is None else f'{age/60:.1f}m'
                    log(f"eval active for {name}; log_age={age_text}")
            else:
                launch_eval(run)
            return False
        scores = eval_scores(run)
        log(f"scores {name}: {scores}")
        if name == 'baseline' and scores.get('evidence_retrieval', 0.0) < 0.1 and state.get('baseline_low_attempts', 0) < 2:
            diagnose_and_maybe_rerun_baseline(state)
            return False
        state.setdefault('accepted', {})[name] = True
        save_state(state)
        return True

    active = active_pids(pids_file(run))
    if active:
        summary = summarize_generation(run)
        age = newest_log_age_seconds(output_path(run) / 'logs')
        if age is not None and age > GENERATION_STALE_SECONDS:
            log(f"generation stale for {name}: active_pids={active} log_age={age/60:.1f}m; terminating and relaunching to resume")
            for pid in active:
                kill_pid(pid)
            pids_file(run).unlink(missing_ok=True)
            launch_generation(run)
        else:
            age_text = 'unknown' if age is None else f'{age/60:.1f}m'
            log(f"running {name}: active_pids={active} complete={summary['complete']}/152 agent4={summary['agent4']} agent5={summary['agent5']} zero_qa={summary['zero_qa'][:10]} log_age={age_text}")
        return False

    # Not complete and no active pids.
    if run['kind'] == 'existing_full' and complete_count(run) > 0:
        log(f"baseline has partial outputs but no active pids; relaunching shards to resume")
    launch_generation(run)
    return False


def prerequisites_met(state, run):
    if run['name'] == 'baseline':
        return True
    if run['name'] == 'e_vj_supp':
        return bool(state.get('accepted', {}).get('baseline'))
    if run['name'] == 'e_supp':
        return bool(state.get('accepted', {}).get('e_vj_supp'))
    if run['name'] == 'vj_supp':
        return bool(state.get('accepted', {}).get('e_supp'))
    return False


def tick():
    state = load_state()
    for name in ORDER:
        run = RUN_BY_NAME[name]
        if not prerequisites_met(state, run):
            log(f"waiting {name}: prerequisites not met")
            return
        done = maybe_handle_run(state, run)
        if not done:
            return
    state['done'] = True
    save_state(state)
    log('ALL VILLAIN SWEEP RUNS AND EVALS COMPLETE')


def main():
    os.chdir(ROOT)
    log('monitor started: 15-minute cadence, GPUs 0/1/2 only, OpenRouter eval CPU/network only')
    while True:
        try:
            tick()
            state = load_state()
            if state.get('done'):
                break
        except Exception as exc:
            log(f'ERROR: {type(exc).__name__}: {exc}')
        time.sleep(CHECK_SECONDS)


if __name__ == '__main__':
    main()
