# 3DGS 텍스처 복원 모듈

기존 `RealDataRobustTextureLoss`(Mode I)를 분리했습니다. 다중 스케일 고주파 손실,
국소 대비 정규화, 텍스처 신뢰도, 광도 신뢰도, 약한 경계 보존 손실을 사용합니다.
SH 계수만 학습하며 Gaussian의 기하와 불투명도를 유지합니다.
패키지를 다른 프로젝트로 복사해 사용할 수 있으며 기존 학습/평가 스크립트에 의존하지 않습니다.

## 설치 및 실행

CUDA GPU, 해당 CUDA 환경에 맞는 PyTorch/torchvision 및 gsplat 빌드 환경이 필요합니다.
저장소 루트에서 실행합니다.

```bash
pip install -r texture_restoration/requirements.txt
python -m texture_restoration --ply input.ply --views views.json --output restored.ply --steps 200
```

```python
from texture_restoration import restore_texture
restore_texture("input.ply", "views.json", "restored.ply", steps=200)
```

`views.json`은 하나 이상의 이미지 쌍을 담은 목록입니다. 이미지 경로는 JSON 파일의
위치를 기준으로 해석합니다. 아래 행을 실제 카메라 값으로 바꾸어 사용합니다.

```json
[
  {
    "rendered": "renders/view_000.png",
    "original": "originals/view_000.png",
    "K": [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]],
    "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]
  }
]
```

- PLY는 Graphdeco 3DGS 형식이어야 합니다. `scale_*`은 log scale,
  `opacity`는 logit, `rot_*`은 wxyz 순서입니다. SH degree 0~3을 지원합니다.
  일반 RGB 포인트 클라우드나 UV 메시는 지원하지 않습니다.
- `K`는 입력 이미지 해상도 기준 내부 행렬이며 `w2c`는 OpenCV 좌표계
  (오른쪽 +X, 아래 +Y, 전방 +Z)의 world-to-camera 행렬입니다.
  Blender/NeRF c2w는 좌표축 변환 및 역행렬 처리가 필요합니다.
- 두 이미지는 같은 시점·해상도·크롭으로 정렬되어야 하며 최소 24×24입니다.
  투명 이미지는 미리 합성하고 `--background 1 1 1` 등과 배경을 일치시키세요.
- PLY와 두 이미지만으로는 픽셀의 3D 대응을 알 수 없어 카메라 정보가 추가로 필요합니다.
- 제공한 렌더링 이미지는 초기 L1 오차를 측정하는 데 사용합니다. 학습 중에는
  현재 SH에서 매번 미분 가능한 이미지를 렌더링해 원본과 비교합니다.
  출력 로그에서 제공 렌더, 초기 재렌더, 최종 재렌더의 L1을 확인할 수 있습니다.
- 출력은 SH가 수정된 PLY입니다. 기존 출력 파일 덮어쓰기를 거부합니다.
  한 시점만 학습하면 다른 시점의 품질이 저하될 수 있어 여러 시점을 권장합니다.

기존 `evaluation/export_ply.py`는 scale/opacity를 선형값으로 내보내므로,
그 출력은 이 모듈이 기대하는 표준 Graphdeco 형식과 다릅니다.
입력 PLY의 저장 규약을 반드시 확인하세요.
