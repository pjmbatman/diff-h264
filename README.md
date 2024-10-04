## H264CompressorNet 개요

H264CompressorNet은 Conv3d 레이어 3개로 이루어진 encoder와 ConvTranspose3d 레이어 3개로 이루어진 decoder로 구성된 autoencoder 구조를 가진 영상 압축 신경망입니다.

- **H264CompressorNet**: Conv3d 레이어 기반의 encoder와 ConvTranspose3d 레이어 기반의 decoder로 구성된 신경망
- **H264Loss**: mse + lpips
- **VideoDataset**: 데이터로더 클래스

## H264 Dataset Preparation

- **Input dataset**: webvid
- **Target dataset**: webvid를 ffmpeg로 압축한 데이터

```bash
python original_h264.py
```

- 실행하면 `webvid/train` 경로의 비디오를 ffmpeg로 h264 압축시켜 `webvid/target` 경로에 타겟 데이터셋을 저장합니다.

## H264 Training

```bash
python h264_train.py
```

### Training 예시 결과:

|  | **batch size** | **frame_size** | **train dataset** | **training time per epoch** | **total loss (lpips + mse)** | **vram** |
| --- | --- | --- | --- | --- | --- | --- |
| Ours | 8 | 8 | webvid 10000 | 1h | 0.01 (took only 10min) | 22GB taken |

### Training 설정:

```python
# Default 설정 예시
input_dir = 'dataset/webvid/train'
gt_dir = 'dataset/webvid/target'
criterion = H264Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 8
epochs = 20 
```

- **input_dir**: 입력 비디오 경로
- **gt_dir**: ffmpeg로 압축된 타겟 비디오 경로
- **loss**: mse + lpips
- **batch_size**: 배치 사이즈
- **epochs**: 에포크
- **resize**: 해상도 (diff_h264.py의 `VideoDataset` 클래스에서 수정 가능)

## H264 Inference

```bash
python h264_inference.py
```

### Inference 설정:

```python
input_dir = 'input_video'
output_dir = 'compressed_video'
gt_dir = "uncompressed_video"
compressed_out = 'compressed_output'

batch_size = 1
max_frames = 16
resize = 256
```

- **input_dir**: 입력 비디오 경로
- **output_dir**: 압축된 출력 비디오 경로
- **gt_dir**: Ground Truth (원본) 비디오 경로

## Evaluation

```bash
python eval.py
```

### Evaluation 설정:

```python
gt_video_dir = "compressed_output/uncompressed_video"
compare_video_dir = "compressed_output/compressed_video"
get_video_metric(gt_video_dir, compare_video_dir)
```

- **gt_video_dir**: 원본 비디오 경로
- **compare_video_dir**: 비교할 비디오 경로

## Attack Types

```bash
python attack.py
```

- **ImageAugmentor** 클래스의 augmentations 파이프라인에서 공격(attack)들을 추가/제거/수정할 수 있습니다.
- 학습된 H264 모델도 파이프라인에 추가 가능합니다.

### Attack 설정 예시:

```python
K.ColorJiggle(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, same_on_batch=False, p=1.0, keepdim=False),
K.RandomGaussianNoise(mean=0.0, std=1.0, same_on_batch=False, p=0.5, keepdim=False),
K.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False),
H264CompressorNet()
```

- **self.apply_augmentation**: 모든 공격을 적용
- **self.apply_random_augmentation**: 랜덤으로 하나의 공격 적용
- **self.apply_all_augmentations**: 모든 공격을 개별적으로 적용 후 저장
- **self.check_differentitable**: 공격 적용 후에도 그라디언트가 흐르는지 확인

## Attack Inference

```bash
python attack_inference.py
```

### Attack Inference 설정:

```python
# Default 설정 예시
input_dir = 'input_video'
output_dir = 'attacked_video'
gt_dir = "gt_video"
attacked_out = "attacked_output"

batch_size = 1
max_frames = 16
resize = 256
```

- **input_dir**: 입력 비디오 경로
- **output_dir**: 공격된 출력 비디오 경로
- **gt_dir**: Ground Truth 비디오 경로
```
