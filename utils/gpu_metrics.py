import subprocess
import time

def get_gpu_metrics(gpu_indices=None):
    # Gets the power consumption (W), GPU usage (%) and memory used (MiB) of each GPU.

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )
    
    gpus_metrics = []

    for line in result.stdout.strip().split("\n"):
        index, power, utilization, memory_used = map(float, line.split(", "))

        if gpu_indices is None or int(index) in gpu_indices:
            gpus_metrics.append({
                "index": int(index),
                "power": power,
                "utilization": utilization,
                "memory_used": memory_used
            })
    
    return gpus_metrics


def get_gpu_memory_total(gpu_indices=None):
    # Gets the total memory of each GPU in MiB.

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )
    
    gpus = []
    for line in result.stdout.strip().split("\n"):
        index, mem_total = map(float, line.split(", "))

        if gpu_indices is None or int(index) in gpu_indices:
            gpus.append({
                "index": int(index),
                "memory_total": mem_total
            })

    return gpus

def record_sample(start_time, gpus):

    sample = {
        "timestamp": time.time() - start_time # Add current sample timestamp
    }

    # GPU metrics
    gpu_metrics = get_gpu_metrics(gpus)
    
    for gpu in gpu_metrics:
        gpu_id = gpu['index']
        prefix = f'gpu_{gpu_id}_'

        sample[prefix + 'utilization'] = gpu['utilization']
        sample[prefix + 'memory_used'] = gpu['memory_used']
        sample[prefix + 'power'] = gpu['power']

    return sample

if __name__ == "__main__":
    print("GPU metrics:", get_gpu_metrics())
    print("GPU memory total:", get_gpu_memory_total())
