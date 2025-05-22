import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os

from train_adv import get_model, pgd_attack  # Reuse training components

def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return testloader

def evaluate(model, testloader, device, attack=False):
    model.eval()
    correct = 0
    total = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        if attack:
            images = pgd_attack(model, images, labels)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    testloader = get_test_loader()

    clean_acc = evaluate(model, testloader, device, attack=False)
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    if args.attack:
        adv_acc = evaluate(model, testloader, device, attack=True)
        print(f"PGD Robust Accuracy: {adv_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model_adv.pth", help="Path to trained model")
    parser.add_argument("--attack", action="store_true", help="Enable PGD attack evaluation")
    args = parser.parse_args()
    main(args)
