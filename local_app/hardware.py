from __future__ import annotations
import csv
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

def available_ram_mib():
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                ('total_physical', 'available_physical', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical / 1024**2
    elif os.path.isfile('/proc/meminfo'):
        with open('/proc/meminfo') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    return None

@dataclass(frozen=True)
class Profile:
    device: str = 'cpu'
    compute_type: str = 'int8'
    concurrency: int = 1
    device_index: int = 0
    cpu_threads: int = max(1, min(8, (os.cpu_count() or 2) // 2))
    reason: str = ''
    device_name: str = 'CPU'
    provisional: bool = False

    def json(self):
        return asdict(self)

def detect(preference='auto'):
    """Driver telemetry only. CUDA inference is initialized in children only."""
    result = {'cpu_threads': os.cpu_count() or 1, 'cpu_name': platform.processor() or 'CPU', 'ram_available_mib': available_ram_mib(), 'gpus': [], 'warning': None}
    if preference == 'cpu':
        return result
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if output.returncode:
            raise RuntimeError('The graphics driver could not report GPU resources.')
        for row in csv.reader(output.stdout.splitlines()):
            if len(row) == 5:
                index, name, total, free, driver = [v.strip() for v in row]
                result['gpus'].append({'index': int(index), 'name': name, 'total_mib': int(total), 'free_mib': int(free), 'driver': driver})
    except Exception as error:
        result['warning'] = 'No compatible GPU was detected. Using CPU.' if isinstance(error, FileNotFoundError) else str(error)
    return result

def candidates(config, hardware, preference='auto'):
    """Policy order is not a VRAM estimate. Every candidate needs a real probe."""
    provisional = config['benchmark_status'] not in ('validated', 'user_specified')
    if preference != 'cpu':
        order = ([config['sweet_spot']] if config.get('sweet_spot') else []) + config['fallback_order']
        if not order:
            order = config['provisional_gpu_order']
        for gpu in sorted(hardware['gpus'], key=lambda g: g['free_mib'], reverse=True):
            for entry in order:
                if entry:
                    yield Profile('cuda', entry['compute_type'], entry['concurrency'], gpu['index'],
                                  device_name=gpu['name'], provisional=provisional)
    for entry in config['cpu_fallback_order']:
        yield Profile('cpu', entry['compute_type'], entry['concurrency'],
                      device_name=hardware.get('cpu_name', 'CPU'), provisional=provisional,
                      reason='CPU only is selected.' if preference == 'cpu' else (hardware.get('warning') or ''))
