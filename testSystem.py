import pandas as pd
import psutil
import os
import time

from baseline_model import load_model_artifact


def get_tslpi(pid):
    tslpi = 0
    task_dir = f'/proc/{pid}/task'

    try:
        for tid in os.listdir(task_dir):
            status_file = os.path.join(task_dir, tid, 'status')
            with open(status_file, 'r') as f:
                for line in f:
                    if line.startswith('State:'):
                        state = line.split()[1]  # e.g., 'S' for interruptible sleep
                        if state == 'S':
                            tslpi += 1
                        break
    except Exception as e:
        print(f"Error accessing task info: {e}")
        return None

    return tslpi

def get_tslpu(pid):
    tslpu = 0
    task_dir = f'/proc/{pid}/task'

    try:
        for tid in os.listdir(task_dir):
            status_file = os.path.join(task_dir, tid, 'status')
            with open(status_file, 'r') as f:
                for line in f:
                    if line.startswith('State:'):
                        state = line.split()[1]  # e.g., 'D' for uninterruptible sleep
                        if state == 'D':
                            tslpu += 1
                        break
    except Exception as e:
        print(f"Error accessing task info: {e}")
        return None

    return tslpu

def get_process_state(pid):
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('State:'):
                    state_code = line.split()[1]  # e.g., 'S', 'R', 'D', etc.
                    return state_code.lower()  # return as 's', 'r', etc. to match your dataset
    except FileNotFoundError:
        return 'e'  # Process exited
    except Exception as e:
        print(f"Error reading status for PID {pid}: {e}")
        return '?'

def get_trun(pid):
    trun = 0
    task_dir = f'/proc/{pid}/task'

    try:
        for tid in os.listdir(task_dir):
            status_file = os.path.join(task_dir, tid, 'status')
            with open(status_file, 'r') as f:
                for line in f:
                    if line.startswith('State:'):
                        state = line.split()[1]  # 'R', 'S', etc.
                        if state == 'R':
                            trun += 1
                        break
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    return trun



artifact = load_model_artifact()
rf_classifier = artifact["model"]
metadata = artifact["metadata"]
feature_columns = metadata["feature_columns"]
threshold = float(os.getenv("PROCESS_SCAN_THRESHOLD", metadata.get("threshold", 0.5)))
max_alerts = int(os.getenv("PROCESS_SCAN_MAX_ALERTS", "25"))


# Columns: ['TRUN', 'TSLPI', 'TSLPU', 'POLI', 'NICE', 'PRI', 'RTPR', 'CPUNR', 'Status', 'State', 'CPU', 'CMD', 'label']


scan_started = time.perf_counter()
process_metadata = []
processes = []
for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent']):
    try:
        info = proc.info
        pid = info['pid']

        poli = os.sched_getscheduler(pid)
        param = os.sched_getparam(pid)
        rtpr = param.sched_priority
        tslpi = get_tslpi(pid)
        tslpu = get_tslpu(pid)
        state = get_process_state(pid).upper()
        trun = get_trun(pid)
        nice = proc.nice()

        processes.append(
            {
                'TRUN': trun,
                'TSLPI': tslpi,
                'TSLPU': tslpu,
                'POLI': poli,
                'NICE': nice,
                'PRI': 20 + nice + 100,
                'RTPR': rtpr,
                'Status': info['status'],
                'State': state,
                'CPU': info['cpu_percent'],
                'CMD': info['name'],
            }
        )
        process_metadata.append({'pid': pid, 'name': info['name']})
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ProcessLookupError):
        continue

if not processes:
    raise SystemExit("No readable processes were collected.")

test_data = pd.DataFrame(processes).reindex(columns=feature_columns)
y_proba = rf_classifier.predict_proba(test_data)

malicious = 0
benign = 0
findings = []

for process_info, probs in zip(process_metadata, y_proba):
    prob_malicious = probs[1]
    pred_class = 1 if prob_malicious >= threshold else 0

    if pred_class == 1:
        findings.append(
            {
                'pid': process_info['pid'],
                'name': process_info['name'],
                'confidence': float(prob_malicious),
            }
        )
        malicious += 1
    else:
        benign += 1

findings.sort(key=lambda finding: finding['confidence'], reverse=True)
for finding in findings[:max_alerts]:
    print(
        f"Process pid: {finding['pid']} ({finding['name']}) "
        f"is malicious (confidence: {finding['confidence']:.2f})"
    )

if len(findings) > max_alerts:
    print(f"... {len(findings) - max_alerts} additional alerts omitted")

elapsed = time.perf_counter() - scan_started
print(
    f"Threshold: {threshold:.2f}, Malicious: {malicious}, "
    f"Benign: {benign}, Total: {len(processes)}, Scan time: {elapsed:.3f}s"
)
