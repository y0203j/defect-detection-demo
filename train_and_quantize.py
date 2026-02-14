import time
import os
import gc
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as albu
from dataset_loader_ma import DefectDataset, get_training_augmentation, get_preprocessing

epochs = 5
batchsize = 2

#detect device (CUDA > MPS > CPU)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"Using Device: {device}")

def get_low_memory_augmentation():
    """
    Forces images to 224x224 (Standard MobileNet size). 
    Reduces memory usage significantly for CPU training.
    """
    train_transform = [
        albu.Resize(height=224, width=224),
        albu.HorizontalFlip(p=0.5),
        albu.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.1, 0.1), p=0.5),
    ]
    return albu.Compose(train_transform)

def train_model():
    encoder = 'mobilenet_v2'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    #data splits
    images_dir = './data/train_images'
    masks_dir = './data/train_masks'
    all_ids = [f for f in os.listdir(images_dir) if f.endswith('b.PNG')]
    random.shuffle(all_ids)
    val_split = int(len(all_ids) * 0.2)
    val_ids = all_ids[:val_split]
    train_ids = all_ids[val_split:]

    train_dataset = DefectDataset(
        images_dir, 
        masks_dir, 
        ids=train_ids, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    val_dataset = DefectDataset(
        images_dir, 
        masks_dir, 
        ids=val_ids, 
        augmentation=None, #no augmentation for validation
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    print(f"Data Split: {len(train_ids)} training, {len(val_ids)} validation images.")


    #model
    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    model.to(device)

    class CombinedLoss(torch.nn.Module):
        def __init__(self):
            super(CombinedLoss, self).__init__()
            self.dice_loss = smp.losses.DiceLoss(mode='binary')
            self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([40.0]).to(device))

        def forward(self, logits, masks):
            return self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

    loss_fn = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"--- Starting Training ({epochs} Epochs) ---")
    total_start = time.time()

    #early stopping parameters
    patience = 3
    min_delta = 0.01
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()


    for epoch in range(epochs):
        #training loop
        model.train()
        running_loss = 0.0
        running_recall = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            probs = logits.sigmoid()
            soft_recall = (probs * masks).sum() / (masks.sum() + 1e-7)
            running_recall += soft_recall.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}: Loss = {loss.item():.4f} | Recall = {soft_recall.item():.4f}")
            
            del images, masks, logits, loss, probs
            if device == 'cuda':
                torch.cuda.empty_cache()
            if device == 'mps':
                torch.mps.empty_cache()

        avg_loss = running_loss / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        print(f"Epoch {epoch+1} Finished. Avg Train Loss: {avg_loss:.4f} | Avg Train Recall: {avg_recall:.4f}")

        #validation loop
        model.eval()
        val_loss = 0.0
        val_recall = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()

                probs = logits.sigmoid()
                soft_recall = (probs * masks).sum() / (masks.sum() + 1e-7)
                val_recall += soft_recall.item()

                del images, masks, logits, loss, probs
                if device == 'cuda':
                    torch.cuda.empty_cache()
                if device == 'mps':
                    torch.mps.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recall = val_recall / len(val_loader)
        print(f"Epoch {epoch+1} VALIDATION. Avg Loss: {avg_val_loss:.4f} | Avg Recall: {avg_val_recall:.4f}")
        
        #early stopping check
        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training finished in {(time.time() - total_start)/60:.1f} minutes.")
    
    #load best model weights
    model.load_state_dict(best_model_wts)

    #save model to CPU for compatibility
    torch.save(model.cpu(), "defect_model_float32.pth")
    return model

def quantize_and_benchmark(model):
    print("\n--- Quantization & Benchmarking ---")
    model.eval()

    #benchmark original
    print("Benchmarking Original Model...")
    dummy = torch.randn(1, 3, 320, 320)
    start = time.time()
    for _ in range(20):
        _ = model(dummy)
    orig_time = (time.time() - start) / 20
    
    #apply dynamic quantization (Float32 -> Int8)
    print("Quantizing...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model, "defect_model_quantized.pth")
    
    #benchmark quantized
    print("Benchmarking Quantized Model...")
    start = time.time()
    for _ in range(20):
        _ = quantized_model(dummy)
    quant_time = (time.time() - start) / 20

    print(f"\nRESULTS:")
    print(f"Original Latency: {orig_time*1000:.2f} ms")
    print(f"Quantized Latency: {quant_time*1000:.2f} ms")
    print(f"Speedup: {orig_time/quant_time:.2f}x")


if __name__ == "__main__":
    trained_model = train_model()
    quantize_and_benchmark(trained_model)
