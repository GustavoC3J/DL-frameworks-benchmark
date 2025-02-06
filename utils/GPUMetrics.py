import subprocess

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

if __name__ == "__main__":
    print("Métricas de GPU:", get_gpu_metrics())
    print("Memoria total de GPU:", get_gpu_memory_total())
