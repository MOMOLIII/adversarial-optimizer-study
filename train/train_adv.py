import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def get_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    return model

def pgd_attack(model, images, labels, eps=0.031, alpha=0.007, iters=7):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_data_loaders()
    model = get_model().to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unsupported optimizer")

    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            adv_inputs = pgd_attack(model, inputs, labels)

            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(trainloader):.4f}")

    torch.save(model.state_dict(), "model_adv.pth")
    print("Training finished and model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train(args)

