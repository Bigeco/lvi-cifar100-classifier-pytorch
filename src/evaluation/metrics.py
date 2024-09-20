import torch

def top_k_accuracy(true_labels, preds, k=1):
    _, top_k_preds = preds.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_preds.eq(true_labels.view(-1, 1).expand_as(top_k_preds))
    accuracy = correct.float().sum(1).mean()
    return accuracy

# Example usage
if __name__ == "__main__":
    true_labels = torch.tensor([2, 1, 0, 4, 2])
    preds = torch.tensor([
        [0.1, 0.2, 0.5, 0.1, 0.1],  # Prediction Result Classes: 2 (가장 높은 점수)
        [0.6, 0.1, 0.1, 0.1, 0.1],  # Prediction Result Classes: 0
        [0.4, 0.2, 0.1, 0.1, 0.2],  # Prediction Result Classes: 0
        [0.1, 0.2, 0.5, 0.1, 0.1],  # Prediction Result Classes: 2
        [0.2, 0.2, 0.6, 0.0, 0.0]   # Prediction Result Classes: 2
    ])
    
    # Calculate Top-1 accuracy
    top_1_accuracy = top_k_accuracy(true_labels, preds, k=1)
    print(f'Top-1 Accuracy: {top_1_accuracy.item() * 100:.2f}%')
    
    # Calculate Top-5 accuracy
    top_5_accuracy = top_k_accuracy(true_labels, preds, k=5)  # 이 예제에서는 k=5는 의미가 없으나, 큰 데이터셋에 적용할 때 사용
    print(f'Top-5 Accuracy: {top_5_accuracy.item() * 100:.2f}%')