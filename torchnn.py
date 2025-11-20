import torch
import torch.nn as nn
from torch import save, load
from torch.optim import Adam  # Optimiseur Adam
from torch.utils.data import DataLoader  # Charge données en batch
from torchvision import datasets, transforms  # Jeux de données images et transforms

# Prépare le jeu de données MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ImageClassifier(nn.Module):  # Déclare modèle CNN
    def __init__(self):  # Constructeur
        super().__init__()  # Init parent
        # Trois convolutions sans padding réduisent 28->26->24->22
        self.model = nn.Sequential(  # Bloc séquentiel
            nn.Conv2d(1, 32, 3),  # Conv 1→32
            nn.ReLU(),  # Activation ReLU
            nn.Conv2d(32, 64, 3),  # Conv 32→64
            nn.ReLU(),  # Activation ReLU
            nn.Conv2d(64, 64, 3),  # Conv 64→64
            nn.ReLU(),  # Activation ReLU
            nn.Flatten(),  # Aplatissement tenseur
            nn.Linear(64 * 22 * 22, 128),  # FC →128 (22 = 28 - 3*2)
            nn.ReLU(),
            nn.Linear(128, 10),  # Sortie 10 classes MNIST
        )  # Fin modèle séquentiel

    def forward(self, x):  # Passage avant
        return self.model(x)  # Renvoie sortie modèle

if __name__ == "__main__":  # Point d'entrée script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = ImageClassifier().to(device)  # Instancie modèle sur device
    opt = Adam(clf.parameters(), lr=1e-3)  # Optimiseur Adam
    loss_fn = nn.CrossEntropyLoss()  # Perte Cross-Entropy

    for epoch in range(10):  # Boucle époques
        for batch in train_loader:  # Itération lots
            x, y = batch  # Sépare images/labels
            x, y = x.to(device), y.to(device)  # Envoie sur device
            yhat = clf(x)  # Prédiction
            loss = loss_fn(yhat, y)  # Calcul perte

            opt.zero_grad()  # Réinitialise gradients
            loss.backward()  # Rétropagation
            opt.step()  # Mise à jour paramètres

            print(f"Epoch {epoch} loss is {loss.item()}")  # Affiche perte

    # Sauvegarde finale de l'état du modèle
    save(clf.state_dict(), "model_state.pt")
