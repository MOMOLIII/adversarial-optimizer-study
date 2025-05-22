import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import yaml

from utils.visualization import plot_metrics

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def get_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    return model

def pgd_attack(model, images, labels, eps=0.031, alpha=0.007, iters=7):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    ori_images = images.data

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

@torch.no_grad()
def evaluate(model, dataloader, device, adv=False, config=None):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        if adv:
            images = pgd_attack(model, images, labels, eps=config['eps'], alpha=config['alpha'], iters=config['pgd_steps'])
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_data_loaders(config['batch_size'])
    model = get_model().to(device)

    if config['optimizer'] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    else:
        raise ValueError("Unsupported optimizer")

    criterion = nn.CrossEntropyLoss()

    # Metrics recording
    metrics = {'loss': [], 'clean_acc': [], 'adv_acc': []}

    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            adv_inputs = pgd_attack(model, inputs, labels, eps=config['eps'], alpha=config['alpha'], iters=config['pgd_steps'])

            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        clean_acc = evaluate(model, testloader, device, adv=False, config=config)
        adv_acc = evaluate(model, testloader, device, adv=True, config=config)

        metrics['loss'].append(avg_loss)
        metrics['clean_acc'].append(clean_acc)
        metrics['adv_acc'].append(adv_acc)

        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.2f}% | PGD Acc: {adv_acc:.2f}%")

    torch.save(model.state_dict(), "model_adv.pth")
    os.makedirs("results", exist_ok=True)
    plot_metrics(metrics, save_path_prefix="results/train")
    print("Training finished. Model and plots saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
