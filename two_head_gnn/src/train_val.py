import torch
from sklearn.metrics import f1_score, accuracy_score

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    epoch_wire_loss = 0
    epoch_action_loss = 0

    all_wire_preds, all_wire_labels = [], []
    all_action_preds, all_action_labels = [], []

    for data in loader:
        # Node type masks
        wire_mask = data.wire_mask
        terminal_mask = data.terminal_mask

        # Extract global node indices for masked nodes
        wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
        data = data.to(device)
        optimizer.zero_grad()

        wire_logits, action_logits = model(
            data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
        )
        wire_label_local = torch.tensor(data.y_wire_local.squeeze()).to(device)
        action_label = torch.tensor(data.y_action).float().to(device)
        if action_label.ndim == 2: # Shape [B, 4]
            action_label = action_label.argmax(dim=1)
        else: # Shape [4]
            action_label = action_label.unsqueeze(0).argmax(dim=1)

        # Loss weights 
        wire_weight = 1.0
        action_weight = 2.0

        # Compute Loss
        wire_loss = criterion(wire_logits, wire_label_local)
        action_loss = criterion(action_logits, action_label)
        epoch_wire_loss += wire_loss.item()
        epoch_action_loss += action_loss.item()
        
        loss = (wire_weight * wire_loss) + (action_weight * action_loss)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Predictions
        wire_pred_local = wire_logits.argmax().item()
        wire_label_local = wire_label_local.item()
        all_wire_preds.append(wire_pred_local)
        all_wire_labels.append(wire_label_local)
        all_action_preds.append(action_logits.argmax().item())
        all_action_labels.append(action_label.item())

    # Accuracy per head
    wire_acc = accuracy_score(all_wire_labels, all_wire_preds)
    action_acc = accuracy_score(all_action_labels, all_action_preds)

    # F1 Score
    wire_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    action_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")

    # Epoch loss
    averaged_total_loss = total_loss / len(loader)
    averaged_wire_loss = epoch_wire_loss / len(loader)
    averaged_action_loss = epoch_action_loss / len(loader)

    return averaged_total_loss, wire_acc, wire_f1, averaged_wire_loss, action_acc, action_f1, averaged_action_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    epoch_val_wire_loss = 0
    epoch_val_action_loss = 0

    all_wire_preds, all_wire_labels = [], []
    all_action_preds, all_action_labels = [], []

    with torch.no_grad():
        for data in loader:
            # Node masks
            wire_mask = data.wire_mask
            terminal_mask = data.terminal_mask
            # Extract global node indices for masked nodes
            wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()

            data = data.to(device)

            wire_logits, action_logits = model(
                data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
            )
            wire_label_local = torch.tensor(data.y_wire_local.squeeze()).to(device)
            action_label = torch.tensor(data.y_action).float().to(device)

            if action_label.ndim == 2: # Shape [B, 4]
                action_label = action_label.argmax(dim=1)
            else: # shape [4]
                action_label = action_label.unsqueeze(0).argmax(dim=1)

            # Loss weights
            wire_weight = 1.0
            action_weight = 2.0

            wire_loss = criterion(wire_logits, wire_label_local)
            epoch_val_wire_loss += wire_loss.item()
            action_loss = criterion(action_logits, action_label)
            epoch_val_action_loss += action_loss.item()

            loss = (wire_weight * wire_loss) + (action_weight * action_loss)
            total_loss += loss.item()

            # Predictions
            wire_pred_local = wire_logits.argmax().item()
            wire_label_local = wire_label_local.item()
            all_wire_preds.append(wire_pred_local)
            all_wire_labels.append(wire_label_local)
            all_action_preds.append(action_logits.argmax().item())
            all_action_labels.append(action_label.item())

    # Accuracy per head
    wire_val_acc = accuracy_score(all_wire_labels, all_wire_preds)
    action_val_acc = accuracy_score(all_action_labels, all_action_preds)

    # F1 Scores
    wire_val_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    action_val_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")

    # Epoch loss
    averaged_val_loss = total_loss / len(loader)
    averaged_val_wire_loss = epoch_val_wire_loss / len(loader)
    averaged_val_action_loss = epoch_val_action_loss / len(loader)

    return averaged_val_loss, wire_val_acc, wire_val_f1, averaged_val_wire_loss, action_val_acc, action_val_f1, averaged_val_action_loss



