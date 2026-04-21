import torch

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Trovati {num_devices} dispositivi CUDA.")

    for i in range(num_devices):
        nome_device = torch.cuda.get_device_name(i)
        print(f"Device {i}: {nome_device}")
else:
    print("Nessun dispositivo CUDA rilevato o driver non configurati correttamente.")