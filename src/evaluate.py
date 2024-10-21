from tqdm import tqdm
import os
import argparse
from datasets import *
from config import *


#Superclass mapping
superclass_mapping = {
    0: [4, 30, 55, 72, 95],      # aquatic mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food containers
    4: [0, 51, 53, 57, 83],      # fruit and vegetables
    5: [22, 39, 40, 86, 87],     # household electrical devices
    6: [5, 20, 25, 84, 94],      # household furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large carnivores
    9: [12, 17, 37, 68, 76],     # large man-made outdoor things
    10: [23, 33, 49, 60, 71],    # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],    # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],    # medium-sized mammals
    13: [26, 45, 77, 79, 99],    # non-insect invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small mammals
    17: [47, 52, 56, 59, 96],    # trees
    18: [8, 13, 48, 58, 90],     # vehicles 1
    19: [41, 69, 81, 85, 89]     # vehicles 2
}

def get_superclass(label):
    for super_class, classes in superclass_mapping.items():
        if label in classes:
            return super_class
    return None


# TA Code
# Define a function to calculate top-1, top-5 accuracy and superclass accuracy
def evaluate(model, loader, criterion, device):
    model.eval()

    valid_losses = []

    total = 0
    superclass_total = 0

    top1_correct = 0
    top5_correct = 0
    superclass_correct = 0
    criterion = criterion if criterion else nn.CrossEntropyLoss()

    # CIFAR-100 superclass mapping
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Testing: "):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            valid_losses.append(loss.item())

            total += labels.size(0)

            _, preds = torch.max(outputs, 1)
            top1_correct += torch.sum(preds == labels).item()

            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_correct += torch.sum(top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))).item()

            # Superclass accuracy
            super_preds = torch.tensor([get_superclass(p.item()) for p in preds], dtype=torch.long)
            super_labels = torch.tensor([get_superclass(t.item()) for t in labels], dtype=torch.long)
            superclass_correct += torch.sum(super_preds == super_labels).item()
            superclass_total += super_labels.size(0)

    epoch_loss = torch.tensor(valid_losses).mean().item()
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    superclass_acc = superclass_correct / superclass_total

    return epoch_loss, top1_acc, top5_acc, superclass_acc
# END

def print_predicted_results(model, loader, criterion, device):
    test_loss, test_top1_acc, test_top5_acc, test_superclass_acc = \
        evaluate(model, loader, criterion, device)

    print(f"Test Loss: {test_loss:.2f}")
    print(f"Test Top-1 Accuracy: {test_top1_acc * 100:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5_acc * 100:.2f}%")
    print(f"Test Top-1 Super Accuracy: {test_superclass_acc * 100:.2f}%")


def main(args):
    # loader 정의
    _, _, test_loader = get_dataloaders(args.root,
                                        args.select_transform,
                                        args.train_ratio,
                                        args.batch_size,
                                        args.num_workers,
                                        args.prefetch_factor,
                                        args.split)

    if os.path.isfile(args.model_path):
        model = MODEL_DICT[args.model_name]()
        checkpoint = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"No checkpoint found at model path: {args.model_path}")

    if args.criterion_name == "FocalLoss":
        criterion = CRITERION_DICT[args.criterion_name](args.gamma)
    elif args.criterion_name == "LabelSmoothingLoss":
        criterion = CRITERION_DICT[args.criterion_name](args.label_smoothing)
    elif args.criterion_name == "CrossEntropyLoss":
        criterion = CRITERION_DICT[args.criterion_name]()
    else:
        raise ValueError(f"Unsupported model: {args.criterion_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_predicted_results(model, test_loader, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./best_shake_pyramidnet_110_epoch_150.pth")
    parser.add_argument("--model_name", type=str, default="shake_pyramidnet_110")
    parser.add_argument("--criterion_name", type=str, default="LabelSmoothingLoss")
    parser.add_argument("--select_transform", type=str, default='RandomCrop,RandomHorizontalFlip,AutoAugment,Cutout')
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--split", type=bool, default=False)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--label_smoothing", type=float, default=0.1)  # Label Smoothing Loss
    args = parser.parse_args()
    main(args)
