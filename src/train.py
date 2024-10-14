import os
import gc
import utils
import argparse
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from losses import *
from optimizers import *

from models.ResNet import *
from models.ResNext import *
from models.EfficientNet import *
from models.DenseNet import *
from models.ViT import *
from models.WideResNet import *
from models.SparseSwin import *
from models.ShakePyramidNet import *
from datasets import get_dataloaders
from evaluate import evaluate, print_predicted_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
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

    # TODO 1: 모델명, 함수명 추가
    # model 정의
    model_dict = {
        "resnet9": resnet9,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "wide_resnet28_10": wide_resnet28_10,
        "shake_pyramidnet_110": shake_pyramidnet_110,
        "swin1": swin1,
        "swin2": swin2,
        "swin3": swin3,
        "swin4": swin4,
        "swin5": swin5,
        "swin6": swin6,
        "efficientnet": efficientnet,
        "densenet201": densenet201
    }

    # 모델 선택 및 초기화
    if args.model in model_dict:
        model = model_dict[args.model]().to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    cudnn.benckmark = True

    # loss function 정의
    # TODO 2: criterion_dict 작성
    criterion_dict = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        # "FocalLoss": ,
        "LabelSmoothingLoss": LabelSmoothingLoss(args.label_smoothing)
    }
    criterion = criterion_dict[args.criterion_name]

    # optimizer 정의
    optimizer_dict = {
        "SGD": optim.SGD(model.parameters(),
                         lr=args.lr,
                         momentum=0.9,
                         weight_decay=args.weight_decay,
                         nesterov=args.nesterov),
        "Adam": optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=args.betas,
                           eps=args.eps,
                           weight_decay=args.weight_decay),
        "AdamW": optim.AdamW(model.parameters(),
                             lr=args.lr,
                             betas=args.betas,
                             eps=args.eps,
                             weight_decay=args.weight_decay),
        "SAM": SAMSGD(model.parameters(),
                      rho=args.rho,
                      lr=args.lr,
                      nesterov=args.nesterov,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    }
    optimizer = optimizer_dict[args.optimizer_name]

    # scheduler 정의
    # TODO 5: 파라미터 모두 args.? 로 변경
    scheduler_dict = {
        "LambdaLR": lambda optimizer: lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 0.95 ** epoch  # 예시 함수, 필요에 따라 수정
        ),
        "MultiplicativeLR": lambda optimizer: lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: 0.95  # 예시 함수, 필요에 따라 수정
        ),
        "StepLR": lambda optimizer: lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        ),
        "MultiStepLR": lambda optimizer: lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        ),
        "ConstantLR": lambda optimizer: lr_scheduler.ConstantLR(
            optimizer,
            factor=0.5,
            total_iters=5
        ),
        "LinearLR": lambda optimizer: lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5,
            total_iters=5
        ),
        "ExponentialLR": lambda optimizer: lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.9
        ),
        "PolynomialLR": lambda optimizer: lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=100,
            power=1.0
        ),
        "CosineAnnealingLR": lambda optimizer: lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100
        ),
        "ChainedScheduler": lambda optimizer: lr_scheduler.ChainedScheduler(
            [
                lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
                lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            ]
        ),
        "SequentialLR": lambda optimizer: lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
                lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            ],
            milestones=[60]
        ),
        "ReduceLROnPlateau": lambda optimizer: lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10
        ),
        "CyclicLR": lambda optimizer: lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.001,
            max_lr=0.1,
            step_size_up=2000
        ),
        "OneCycleLR": lambda optimizer: lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=100
        ),
        "CosineAnnealingWarmRestarts": lambda optimizer: lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
    }
    scheduler = scheduler_dict[args.scheduler_name](optimizer)

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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(train_loader), leave=False)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            if args.optimizer_name != "SAM":
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                if args.grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

                optimizer.step()

                train_acc += utils.accuracy(outputs, labels).item()
                train_loss += loss.item() * labels.size(0)
                train_n += labels.size(0)
                bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                    train_loss / train_n, train_acc / train_n * 100), refresh=True)
                bar.update()
            else:
                # Closure function for SAM optimizer
                def closure():
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)  # Assuming mixup is used
                    loss.backward()
                    if args.grad_clip:
                        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                    return loss

                # Perform SAM step (Sharpness-Aware Minimization)
                loss = optimizer.step(closure)
                with torch.no_grad():
                    outputs = model(images)
                    train_acc += utils.accuracy(outputs, labels).item()
                train_loss += loss.item() * labels.size(0)
                train_n += labels.size(0)
                bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                    train_loss / train_n, train_acc / train_n * 100), refresh=True)
                bar.update()

        bar.close()

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
            torch.save(model.state_dict(), f"best_{args.model}_epoch_{epoch}.pth")

    # 메모리 정리
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    best_model = model_dict[args.model]().to(device)
    best_model.load_state_dict(torch.load(f"best_{args.model}_epoch_{best_epoch}.pth"))
    print_predicted_results(best_model, test_loader, criterion, device)

    # 메모리 정리
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def add_model_specific_args(parser, model_name):
    # TODO 4: 다음은 예시 코드임. 각 모델에 대한 argument 에 대해서 추가
    if model_name == "resnet":
        group = parser.add_argument_group('ResNet')
        group.add_argument("--num_layers", type=int, default=50)
        group.add_argument("--bottleneck", action="store_true")
    elif model_name == "wide_resnet":
        group = parser.add_argument_group('Wide ResNet')
        group.add_argument("--depth", type=int, default=28)
        group.add_argument("--widen_factor", type=int, default=10)
    elif model_name == "swin":
        group = parser.add_argument_group('Swin Transformer')
        group.add_argument("--embed_dim", type=int, default=96)
        group.add_argument("--depths", nargs='+', type=int, default=[2, 2, 6, 2])
        group.add_argument("--num_heads", nargs='+', type=int, default=[3, 6, 12, 24])
    elif model_name == "efficientNet":
        group = parser.add_argument_group('EfficientNet')
        group.add_argument("--width", type=float, default=1.0)
        group.add_argument("--depth", type=float, default=1.0)
        group.add_argument("--bn_momentum", type=float, default=0.90)
        group.add_argument("--ratio", type=float, default=0.2)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO 5: argument 추가
    parser.add_argument("--label", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from. 0 starts from scratch.")
    # For Training
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--batch_size", type=int, default=128)
    temp_args, _ = parser.parse_known_args()
    # For Networks
    add_model_specific_args(parser, temp_args.model)

    args = parser.parse_args()
    main(args)


