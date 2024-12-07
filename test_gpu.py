import pynvml

print("Testando inicialização da GPU...")

try:
    pynvml.nvmlInit()
    print("NVML inicializado com sucesso!")
    
    # Get the first GPU handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("GPU handle obtido!")
    
    # Get GPU name
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"Nome da GPU: {name.decode('utf-8')}")
    
    # Get memory info
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Memória Total: {memory.total / 1024**2:.0f}MB")
    print(f"Memória Usada: {memory.used / 1024**2:.0f}MB")
    print(f"Memória Livre: {memory.free / 1024**2:.0f}MB")
    
    # Get utilization rates
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"Uso da GPU: {utilization.gpu}%")
    print(f"Uso da Memória: {utilization.memory}%")
    
except Exception as e:
    print(f"Erro: {str(e)}")
finally:
    try:
        pynvml.nvmlShutdown()
        print("NVML finalizado com sucesso!")
    except:
        pass
