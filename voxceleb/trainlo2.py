import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.35):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        # Define a trainable weight matrix for the AMSoftmax loss
        self.weight = nn.Parameter(torch.randn(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weight matrix
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

       # Apply the margin to the cosine similarity
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits_with_margin = self.s * (cosine - one_hot * self.m)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_with_margin, labels)
        return loss

def eval_loss_only(pretrained_model, criterion, data_loader, device):
    # Freeze all parameters in the pretrained model
    for param in pretrained_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([criterion.weight], lr=0.001)  # Train only AMSoftmax weight matrix

    pretrained_model.eval()
    for epoch in range(5):  # Set a fixed number of epochs for loss training
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through pretrained model to get embeddings
            with torch.no_grad():  # Ensure model is not updated
                embeddings = pretrained_model(inputs)

            # Compute loss with AMSoftmax
            loss = criterion(embeddings, labels)

            # Update only the weight matrix in AMSoftmax
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(data_loader):.4f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained model
    pretrained_model = torch.load("path/to/pretrained_model.pth", map_location=device)
    pretrained_model.to(device)

    # Set up AMSoftmax loss with the correct dimensions
    embedding_dim = pretrained_model.fc.out_features  # Final embedding size
    num_classes = 1000  # Number of classes in VoxCeleb (adjust if needed)
    amsoftmax_loss = AMSoftmaxLoss(embedding_dim=embedding_dim, num_classes=num_classes, s=30.0, m=0.35).to(device)

    # Prepare the data loader for VoxCeleb
    data_loader = DataLoader(..., batch_size=32, shuffle=True)  # Replace `...` with your dataset setup

    # Run evaluation to train only the loss
    eval_loss_only(pretrained_model, amsoftmax_loss, data_loader, device)
