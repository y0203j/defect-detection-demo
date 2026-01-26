import time
import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset_loader import DefectDataset, get_training_augmentation, get_preprocessing

epochs = 5  #increased to 5 since GPU should be faster
batchsize = 2


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("Using NVIDIA CUDA GPU")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Using Apple Silicon GPU (MPS)")
else:
    print("GPU not found. Using CPU.")

def train_model():
    encoder = 'mobilenet_v2'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    train_dataset = DefectDataset(
        './data/train_images', 
        './data/train_masks', 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    #model to gpu
    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    model.to(device) 
    
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"--- Starting Training ({epochs} Epochs) on {device} ---")
    
    total_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_recall = 0.0
        
        for i, (images, masks) in enumerate(train_loader):

            #run on gpu
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            probs = logits.sigmoid()
            preds = (probs > 0.5).float()
            tp = (preds * masks).sum().float()
            fn = ((1 - preds) * masks).sum().float()
            recall = tp / (tp + fn + 1e-7)

            running_recall += recall.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}: Loss = {loss.item():.4f} | Recall = {recall.item():.4f}")
            
            #force-free memory for the next batch
            del images, masks, logits, loss, probs, preds, tp, fn, recall
            if device == 'cuda':
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()
                
        avg_loss = running_loss / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} | Avg Recall: {avg_recall:.4f}")

    print(f"Training finished in {(time.time() - total_start)/60:.1f} minutes.")

    #save the gpu model as cpu compatible file
    torch.save(model.cpu(), "defect_model_float32.pth")
    return model

def quantize_and_benchmark(model):
    print("\n--- Quantization & Benchmarking ---")
    
    #dynamic quantization in PyTorch only works on CPU,move the model back to cpu.
    model.cpu()
    model.eval()
    
    #benchmark original
    print("Benchmarking Original Model (CPU)...")
    start = time.time()
    dummy = torch.randn(1, 3, 320, 320) #back on CPU
    for _ in range(20): _ = model(dummy)
    orig_time = (time.time() - start) / 20
    
    #quantize
    print("Quantizing...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model, "defect_model_quantized.pth")
    
    #benchmark quantized
    print("Benchmarking Quantized Model (CPU)...")
    start = time.time()
    for _ in range(20): _ = quantized_model(dummy)
    quant_time = (time.time() - start) / 20
    
    print(f"\nRESULTS:")
    print(f"Original Latency: {orig_time*1000:.2f} ms")
    print(f"Quantized Latency: {quant_time*1000:.2f} ms")
    print(f"Speedup: {orig_time/quant_time:.2f}x")

if __name__ == "__main__":
    trained_model = train_model()
    quantize_and_benchmark(trained_model)