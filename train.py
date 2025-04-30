import torch
from data import DataLoader as MyDataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader
from tqdm import tqdm
from model import MANAGER, MyModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

annotation_path = "data_annotation_politics_economy_label.csv" #"data_annotation_label.csv"
batch_size = 1
shuffle = True                                                      
learning_rate = 0.001
epochs = 10

def train_batch(model:MyModel, loader:DataLoader, epochs, optimizer, criterion, device):
    
    loss_history = []
    for epoch in tqdm(range(epochs),desc="Training"):
        total_loss = 0
        model.train()
        for n, (t_emb,a_emb,v_emb,q_emb,label) in enumerate(loader):
            t_emb,a_emb,v_emb,q_emb,label = t_emb.to(device), a_emb.to(device), v_emb.to(device), q_emb.to(device), label.to(device)
            # print(t_emb.shape, a_emb.shape, v_emb.shape, q_emb.shape, label.shape)
            optimizer.zero_grad()
            out = model(a_emb, v_emb, t_emb, q_emb) #batch.edge_index,batch.batch, batch.question)
            loss = criterion(out.squeeze(), label.squeeze().float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)

        print(f"{epoch+1}/{epochs} Avg. Loss: {avg_loss}")

        model.eval()
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for n, (t_emb,a_emb,v_emb,q_emb,label) in enumerate(loader):
                t_emb,a_emb,v_emb,q_emb,label = t_emb.to(device), a_emb.to(device), v_emb.to(device), q_emb.to(device), label.to(device)
                out = model(a_emb, v_emb, t_emb, q_emb)
                preds = torch.sigmoid(out).squeeze().cpu().numpy()
                preds = [1 if preds > 0.5 else 0]
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds)

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)

            print(f"Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")

        torch.save(model.state_dict(), "model.pth")
    
    return loss_history

if __name__ == "__main__":
    # Example usage
    input_dim = 768
    hidden_dim1 =768
    hidden_dim2 = 768
    output_dim = 1
    num_encoder_layers = 24
    dropout_p = 0.25

    pos_weight = torch.tensor(0.192)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(input_dim, hidden_dim1, hidden_dim2).to(device)   #.half()

    # import time
    # time.sleep(30)

    # exit()

    myData = MyDataLoader(annotation_path)
    loader = DataLoader(myData, batch_size=batch_size, shuffle=True)
    
    # data = myData() # Data object(x=emb, edge_index=graph, y=label)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    avg_loss = train_batch(model, loader, epochs, optimizer, criterion, device)
    print("Average Loss:", avg_loss)

    torch.save(model.state_dict(), "model.pth")
