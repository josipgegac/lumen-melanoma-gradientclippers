from tqdm import tqdm

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate(model, dataloader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluating...'):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_probs.append(probs.cpu().numpy().squeeze(1))
            all_preds.append(preds.cpu().numpy().squeeze(1))
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs)

    return accuracy, precision, recall, f1, auroc

def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}...')

        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc='Training...'):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits.squeeze(1), y.float())
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        
        accuracy, precision, recall, f1, auroc = evaluate(model, test_loader, device)
        print(f'Loss: {total_loss / len(train_loader):.4f}')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}')

    return model