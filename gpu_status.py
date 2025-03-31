import torch

def mostrar_status_gpu():
    print("=== RESUMO DA GPU ===\n")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

if __name__ == "__main__":
    mostrar_status_gpu()
