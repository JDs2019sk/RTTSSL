import pynvml

print("Testing GPU initialization...")

try:
    pynvml.nvmlInit()
    print("NVML initialized successfully!")
    
    # Get the first GPU handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("GPU handle obtained!")
    
    # Get GPU name
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"GPU Name: {name.decode('utf-8')}")
    
    # Get memory info
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total Memory: {memory.total / 1024**2:.0f}MB")
    print(f"Used Memory: {memory.used / 1024**2:.0f}MB")
    print(f"Free Memory: {memory.free / 1024**2:.0f}MB")
    
    # Get utilization rates
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
