import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import voxceleb.loss.amsoftmax as vc_amsoftmax

class AMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.35):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        # Define a trainable weight matrix for the AMSoftmax loss
        self.weight = nn.Parameter(torch.randn(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        if self.weight is None:
            self.weight = nn.Parameter(torch.randn(embeddings.size(1), labels.max().item() + 1)).to(embeddings.device)
            nn.init.xavier_normal_(self.weight)

        # Normalize embeddings and weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # Add margin to cosine similarity
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits_with_margin = self.s * (cosine - one_hot * self.m)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_with_margin, labels)
        return loss

def train_loss_with_pretrained_model(model_path, loss_fn, train_loader, device):
    # Load pretrained model
    pretrained_model = torch.load(model_path, map_location=device)
    pretrained_model.eval()

    # Freeze pretrained layers if only the final layer needs training
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Replace or add a new output layer, if needed
    num_classes = 1000  # Number of classes in VoxCeleb
    embedding_dim = pretrained_model.fc.out_features  # Check your model's final embedding size
    pretrained_model.fc = nn.Linear(embedding_dim, num_classes)  # Replace with your desired output layer

    # Load data for fine-tuning (e.g., from VoxCeleb)
    #train_loader = ...  # Your DataLoader here

    # Set up optimizer, AM-Softmax loss, and train
    #criterion = loss
    optimizer = torch.optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

    # Fine-tune with AMSoftmax
    pretrained_model.train()
    for epoch in range(5):  # Fine-tune for a few epochs
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            embeddings = pretrained_model(inputs)
            loss = loss_fn(embeddings, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Loss with Pretrained Model')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to model file')
    parser.add_argument('--loss', '-l', type=str, default="AMSoftmaxLoss", help='Loss function')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')

    args = parser.parse_args()

    # Set up training DataLoader
    train_dataset = ...  # Define your dataset here, e.g., a Dataset class
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Set device
    device = torch.device(args.device)

    # Initialize the AMSoftmax loss
    if args.loss == "AMSoftmaxLoss":
        loss = AMSoftmaxLoss(s=30.0, m=0.35)
    else:
        raise ValueError(f"Unknown loss function {args.loss}")

    # Train model with specified loss
    train_loss_with_pretrained_model(args.model, loss, train_loader, device)
