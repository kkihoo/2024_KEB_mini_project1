import random
import os
import numpy as np
import torch
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoConfig
from tqdm import tqdm
from dataset_swin_v2_2 import BirdDataset, seed_everything, prepare_datasets


def train(model, optimizer, train_loader, val_loader, scheduler, device, num_epochs=5):
    best_model = None
    best_accuracy = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # 검증 단계
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        valid_loss /= len(val_loader)
        accuracy = correct / total

        # 학습률 조정
        scheduler.step(valid_loss)  # 여기서 valid_loss를 전달합니다.

        # 최고 성능 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Accuracy: {accuracy:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print("-" * 50)

    return best_model


def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs).logits
            preds.extend(outputs.argmax(1).cpu().numpy())
    return preds


if __name__ == "__main__":
    seed_everything(42)  # Seed 고정

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 데이터 불러오기 및 준비
    train_df, valid_df, label_to_idx, idx_to_label = prepare_datasets("./train.csv")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 데이터셋 생성
    train_dataset = BirdDataset(
        train_df,
        transform=train_transform,
        use_upscaled=False,
        label_to_idx=label_to_idx,
    )
    valid_dataset = BirdDataset(
        valid_df,
        transform=valid_transform,
        use_upscaled=False,
        label_to_idx=label_to_idx,
    )

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    # 모델 정의 부분
    model_name = "microsoft/swinv2-base-patch4-window8-256"

    # 설정 로드
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(label_to_idx)
    config.id2label = idx_to_label
    config.label2id = label_to_idx

    # 모델 로드 및 분류기 교체
    model = AutoModelForImageClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )

    # 새로운 분류기 초기화
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(label_to_idx))

    num_epochs = 5
    model.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    trained_model = train(
        model, optimizer, train_loader, valid_loader, scheduler, device, num_epochs
    )

    # 모델 저장
    torch.save(trained_model.state_dict(), "bird_classifier_2.pth")

    test = pd.read_csv("./test.csv")

    test_dataset = BirdDataset(
        test, transform=valid_transform, use_upscaled=False, is_test=True
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load("bird_classifier_2.pth"))
    model.to(device)

    preds = inference(model, test_loader, device)
    preds = [idx_to_label[pred] for pred in preds]

    submit = pd.read_csv("./sample_submission.csv")
    submission = pd.DataFrame({"id": test["id"], "label": preds})
    submission.to_csv("swinv2_2.csv", index=False)
