import psutil
import time
import csv
import socket

# Define which attributes to collect per process
fieldnames = [
    'timestamp','pid','name',
    # CPU
    'cpu_percent','cpu_user_time','cpu_system_time','num_threads',
    'vol_ctx_switches','invol_ctx_switches',
    # Memory
    'memory_rss','memory_vms','memory_percent','memory_shared',
    'page_faults_minor','page_faults_major',
    # I/O
    'io_read_count','io_write_count','io_read_bytes','io_write_bytes',
    # Files
    'num_open_files','num_fds',
    # Network
    'num_connections','num_conn_tcp','num_conn_udp',
    # Hierarchy
    'num_children','status','nice',

    # label
    'label'
]

with open('proc_features.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for proc in psutil.process_iter():
        try:
            with proc.oneshot():
                info = {}
                info['timestamp'] = time.time()
                info['pid'] = proc.pid
                info['name'] = proc.name()
                # CPU info
                info['cpu_percent'] = proc.cpu_percent(interval=None)
                cpu_times = proc.cpu_times()
                info['cpu_user_time'] = cpu_times.user
                info['cpu_system_time'] = cpu_times.system
                info['num_threads'] = proc.num_threads()
                ctx = proc.num_ctx_switches()
                info['vol_ctx_switches'] = ctx.voluntary
                info['invol_ctx_switches'] = ctx.involuntary
                info['nice'] = proc.nice()
                # Memory info
                mem = proc.memory_info()
                info['memory_rss'] = mem.rss
                info['memory_vms'] = mem.vms
                info['memory_shared'] = getattr(mem, 'shared', 0)
                info['memory_percent'] = proc.memory_percent()
                # Page faults from /proc (psutil does not provide directly)
                status = proc.as_dict(attrs=['status'], ad_value='')
                info['status'] = status.get('status', '')
                # For page faults, read from /proc/[pid]/status
                with open(f'/proc/{proc.pid}/status', 'r') as f:
                    text = f.read()
                for line in text.splitlines():
                    if line.startswith('VmSwap:'):
                        # skip; no direct need
                        continue
                    if line.startswith('Minor'):
                        info['page_faults_minor'] = int(line.split()[1])
                    if line.startswith('Major'):
                        info['page_faults_major'] = int(line.split()[1])
                # I/O counters
                try:
                    io = proc.io_counters()
                    info['io_read_count'] = io.read_count
                    info['io_write_count'] = io.write_count
                    info['io_read_bytes'] = io.read_bytes
                    info['io_write_bytes'] = io.write_bytes
                except (psutil.AccessDenied, AttributeError):
                    info['io_read_count'] = info['io_write_count'] = 0
                    info['io_read_bytes'] = info['io_write_bytes'] = 0
                # Open files and fds
                try:
                    info['num_open_files'] = len(proc.open_files())
                except (psutil.AccessDenied, NotImplementedError):
                    info['num_open_files'] = 0
                try:
                    info['num_fds'] = proc.num_fds()
                except (AttributeError, NotImplementedError, psutil.AccessDenied):
                    info['num_fds'] = 0
                # Network connections
                try:
                    conns = proc.connections(kind='inet')
                    info['num_connections'] = len(conns)
                    info['num_conn_tcp'] = sum(1 for c in conns if c.type == socket.SOCK_STREAM)
                    info['num_conn_udp'] = sum(1 for c in conns if c.type == socket.SOCK_DGRAM) 
                except (psutil.AccessDenied, NotImplementedError):
                    info['num_connections'] = info['num_conn_tcp'] = info['num_conn_udp'] = 0
                # Children
                try:
                    info['num_children'] = len(proc.children())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    info['num_children'] = 0

                info['label'] = 0

                writer.writerow(info)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
