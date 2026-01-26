import time
import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset_loader import DefectDataset, get_training_augmentation, get_preprocessing

epochs = 5 # Keep it small for quick testing
batchsize = 2 # Keep it small for quick testing
device = 'cpu' 

def get_low_memory_augmentation():
    """
    Forces images to 224x224 (Standard MobileNet size).
    This should reduces memory usage compared to 320x320.
    """
    train_transform = [
        albu.Resize(height=224, width=224), # <--- THE MEMORY SAVER
        albu.HorizontalFlip(p=0.5),
        albu.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.1, 0.1), p=0.5),
    ]
    return albu.Compose(train_transform)

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

    #use only 30 samples for quick testing
    train_dataset.ids = train_dataset.ids[:30]  
    train_dataset.images_fps = train_dataset.images_fps[:30]
    train_dataset.masks_fps = train_dataset.masks_fps[:30]

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"--- Starting Training ({epochs} Epochs) ---")
    
    total_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_recall = 0.0
        
        for i, (images, masks) in enumerate(train_loader):
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
            
            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i}: Loss = {loss.item():.4f} | Recall = {recall.item():.4f}")

            #force-delete variables to free RAM immediately
            del images, masks, logits, loss, probs, preds, tp, fn, recall
            gc.collect()

        avg_loss = running_loss / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} | Avg Recall: {avg_recall:.4f}")

    print(f"Training finished in {(time.time() - total_start)/60:.1f} minutes.")

    torch.save(model, "defect_model_float32.pth")
    return model

def quantize_and_benchmark(model):
    print("\n--- Quantization & Benchmarking ---")
    model.eval()
    
    #benchmark original
    start = time.time()
    dummy = torch.randn(1, 3, 320, 320)
    for _ in range(20): _ = model(dummy)
    orig_time = (time.time() - start) / 20
    
    #quantize (Float32 -> Int8)
    model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model, "defect_model_quantized.pth")
    
    #benchmark quantized
    start = time.time()
    for _ in range(20): _ = quantized_model(dummy)
    quant_time = (time.time() - start) / 20
    
    print(f"Original Latency: {orig_time*1000:.2f} ms")
    print(f"Quantized Latency: {quant_time*1000:.2f} ms")
    print(f"Speedup: {orig_time/quant_time:.2f}x")

if __name__ == "__main__":
    model = train_model()
    quantize_and_benchmark(model)