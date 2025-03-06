import pynvml

print("Testing GPU initialization...")

try:
    pynvml.nvmlInit()
    print("NVML initialized successfully!")
    
    # identificador da GPU
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("GPU handle obtained!")
    
    # nome da GPU
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    print(f"GPU Name: {name}")
    
    # informação da memória da GPU
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total Memory: {memory.total / 1024**2:.0f}MB")
    print(f"Used Memory: {memory.used / 1024**2:.0f}MB")
    print(f"Free Memory: {memory.free / 1024**2:.0f}MB")
    
    # taxas de utilização em percentagem
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Usage: {utilization.gpu}%")
    print(f"Memory Usage: {utilization.memory}%")
    
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    try:
        pynvml.nvmlShutdown()
        print("NVML shutdown successfully!")
    except:
        pass
