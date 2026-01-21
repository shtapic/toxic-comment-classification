import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, n_epochs=5, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train() # Ensure train mode
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for text, labels in train_loader:
            text = text.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(text)
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
        loss_history.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')

    plt.plot(range(1, n_epochs+1), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

def evaluate_model(model, val_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for text, labels in val_loader:
            text = text.to(device)
            output = model(text)
            all_outputs.append(output.cpu().numpy())
            all_labels.append(labels.numpy())
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    return all_labels, all_outputs