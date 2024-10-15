import os
import gc
import time

import utils
import argparse
from tqdm import tqdm
import random

import torch.backends.cudnn as cudnn
from datasets import *
from evaluate import print_predicted_results
from config import *
from visualizations import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    seed_everything(args.seed)  # Seed 고정

    # loader 정의
    train_loader, valid_loader, test_loader = get_dataloaders(
                                                args.root,
                                                args.select_transform,
                                                args.train_ratio,
                                                args.batch_size,
                                                args.num_workers,
                                                args.split)
    if not valid_loader:
        test_loader = valid_loader

    # 모델 선택 및 초기화
    if args.model_name in MODEL_DICT:
        model = MODEL_DICT[args.model_name]().to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    cudnn.benckmark = True

    # loss function 정의
    if args.criterion_name == "FocalLoss":
        criterion = CRITERION_DICT[args.criterion_name](args.gamma)
    elif args.criterion_name == "LabelSmoothingLoss":
        criterion = CRITERION_DICT[args.criterion_name](args.label_smoothing)
    elif args.criterion_name == "CrossEntropyLoss":
        criterion = CRITERION_DICT[args.criterion_name]()
    else:
        raise ValueError(f"Unsupported model: {args.criterion_name}")

    # optimizer 정의
    if args.optimizer_name == "SGD":  # Best Top-1 accuracy
        optimizer = OPTIMIZER_DICT[args.optimizer_name](
                                            model.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay,
                                            nesterov=args.nesterov)
    elif args.optimizer_name == "Adam":
        optimizer = OPTIMIZER_DICT[args.optimizer_name](
                                            model.parameters(),
                                            lr=args.lr,
                                            betas=args.betas,
                                            eps=args.eps,
                                            weight_decay=args.weight_decay)

    elif args.optimizer_name == "AdamW":
        optimizer = OPTIMIZER_DICT[args.optimizer_name](
                                            model.parameters(),
                                            lr=args.lr,
                                            betas=args.betas,
                                            eps=args.eps,
                                            weight_decay=args.weight_decay)

    elif args.optimizer_name == "SAM":
        optimizer = OPTIMIZER_DICT[args.optimizer_name](
                                            model.parameters(),
                                            rho=args.rho,
                                            lr=args.lr,
                                            nesterov=args.nesterov,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_name}")



    # scheduler 정의
    if args.scheduler_name in ["MultiStepLR", "CombinedScheduler"]:
        scheduler = SCHEDULER_DICT[args.scheduler_name](optimizer, args.epochs)
    else:
        scheduler = SCHEDULER_DICT[args.scheduler_name](optimizer)

    # Checkpoint에서 모델 및 옵티마이저 상태 불러오기
    start_epoch = 0
    if args.resume_epoch > 0:
        checkpoint_path = os.path.join(args.checkpoint, f"{args.resume_epoch}.tar")
        if os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = args.resume_epoch
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    logger = utils.Logger(args.checkpoint, headers, resume=args.resume_epoch > 0)
    best_top1_acc, best_epoch = 0, 0
    lst_train_loss, lst_test_loss = [], []

    start_time = time.time()
    time_limit = 24 * 60 * 60

    for epoch in range(start_epoch, args.epochs):
        current_time = time.time()
        elapsed_time = current_time - start_time

        # 24시간 초과 시 학습 종료
        if args.limit_24h and epoch > elapsed_time + 600 > time_limit:
            print("Learning terminated as it is expected to exceed 24 hours")
            break

        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(train_loader), leave=False)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            if args.mixup:
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, 1.0)

            def closure():
                optimizer.zero_grad()
                if args.mixup:
                    outputs = model(mixed_images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)  # Assuming mixup is used
                loss.backward()
                if args.grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                return loss

            if args.optimizer_name == "SAM":
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

            with torch.no_grad():
                outputs = model(images)
                train_acc += utils.accuracy(outputs, labels).item()
                train_loss += loss.item() * labels.size(0)
                train_n += labels.size(0)

            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                train_loss / train_n, train_acc / train_n * 100), refresh=True)
            bar.update()
        bar.close()
        lst_train_loss.append(train_loss / train_n)

        model.eval()
        test_loss, test_acc, test_n = 0, 0, 0
        for images, labels in tqdm(test_loader, total=len(test_loader), leave=False):
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                test_acc += utils.accuracy(outputs, labels).item()
                test_n += labels.size(0)

        scheduler.step(test_loss / test_n)
        test_top1_acc = test_acc / test_n
        lst_test_loss.append(test_loss / test_n)

        if (epoch + 1) % args.snapshot_interval == 0:
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(args.checkpoint, "{}.tar".format(epoch + 1)))

        lr = optimizer.param_groups[0]["lr"]
        logger.write(epoch+1, lr, train_loss / train_n, test_loss / test_n,
                     train_acc / train_n * 100, test_top1_acc * 100)

        if test_top1_acc > best_top1_acc:
            best_top1_acc = test_top1_acc
            best_epoch = epoch
            print(f"Best model: Epoch {epoch}/{args.epochs} - Accuracy: {best_top1_acc:.2f}%")
            torch.save(model.state_dict(), f"best_{args.model_name}_epoch_{epoch}.pth")


    if args.plot_loss:
        plot_loss_curves(lst_train_loss, lst_test_loss)

    # 메모리 정리
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    best_model = MODEL_DICT[args.model_name]().to(device)
    best_model.load_state_dict(torch.load(f"best_{args.model_name}_epoch_{best_epoch}.pth"))
    print_predicted_results(best_model, test_loader, criterion, device)

    # 메모리 정리
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from. 0 starts from scratch.")

    # For Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--select_transform", type=tuple, default=('RandomCrop', 'RandomHorizontalFlip', 'Cutout'))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--split", type=bool, default=False)
    parser.add_argument("--grad_clip", type=float, default=0)
    parser.add_argument("--mixup", type=bool, default=False)
    parser.add_argument("--limit_24h", type=bool, default=False)

    # For Networks
    parser.add_argument("--model_name", type=str, default="shake_pyramidnet_110")

    # For Loss Function
    parser.add_argument("--criterion_name", type=str, default="CrossEntropyLoss")
    parser.add_argument("--gamma", type=float, default=2.0) #  Focal Loss
    parser.add_argument("--label_smoothing", type=float, default=0.1) # Label Smoothing Loss

    # For Optimizer
    parser.add_argument("--optimizer_name", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=0.1) # SGD, Adam, AdamW, SAM
    parser.add_argument("--momentum", type=float, default=0.9) # SGD, SAM
    parser.add_argument("--weight_decay", type=float, default=0.0001) # SGD, Adam, AdamW, SAM
    parser.add_argument("--nesterov", type=bool, default=True) # SGD, SAM
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999)) # Adam, AdamW
    parser.add_argument("--eps", type=float, default=1e-08) # Adam, AdamW
    parser.add_argument("--rho", type=float, default=0.05) # SAM
    parser.add_argument()

    # For Scheduler
    parser.add_argument("--scheduler_name", type=str, default="CombinedScheduler")

    # For Visualization
    # parser.add_argument("--plot_count", type=bool, default=False)
    parser.add_argument("--plot_loss", type=bool, default=False)

    args = parser.parse_args()
    main(args)
