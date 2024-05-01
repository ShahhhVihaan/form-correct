# form-correct


## 🏁 Installation
In a python 3.8 conda environment:

- Install `Pytorch`:

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch
```

- Install pytorchvideo and transformers from main branch:

```bash
pip install git+https://github.com/facebookresearch/pytorchvideo.git
pip install git+https://github.com/huggingface/transformers.git
```

- Install `video-transformers`:

```bash
pip install video-transformers
```
- To fix dependency issue
```bash
pip install transformers==4.28.0
```
- Use editable video-transformers version
```bash
pip install -e video-transformers/
```
## Model Weights and Checkpoints
[Click here](https://drive.google.com/drive/folders/1g7yW-UVDK4AcdcacucArsPoVLQnqh_lP?usp=sharing) for the model weights and checkpoints.
1. exp8 - TimeSformer
2. exp11 - ConvNext + Transformer
3. exp12 - EfficientNet + GRU


## 🔥 Usage for Toy Dataset

- Fine-tune Timesformer (from HuggingFace) video classifier:

```python
from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)

datamodule = VideoDataModule(
    train_root="toy_dataset/train",
    val_root="toy_dataset/val",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head)

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=8)

trainer.fit()

```

- Fine-tune ConvNeXT (from HuggingFace) + Transformer based video classifier:

```python
from torch.optim import AdamW
from video_transformers import TimeDistributed, VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.necks import TransformerNeck
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TimeDistributed(TransformersBackbone("facebook/convnext-small-224", num_unfrozen_stages=1))
neck = TransformerNeck(
    num_features=backbone.num_features,
    num_timesteps=8,
    transformer_enc_num_heads=4,
    transformer_enc_num_layers=2,
    dropout_p=0.1,
)

datamodule = VideoDataModule(
    train_root="toy_dataset/train",
    val_root="toy_dataset/val",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=neck.num_features, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head, neck)

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(
    datamodule,
    model,
    optimizer=optimizer,
    max_epochs=8
)

trainer.fit()

```

- Fine-tune EfficientNet (from HuggingFace) + GRU based video classifier:

```python
from video_transformers import TimeDistributed, VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.necks import GRUNeck
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TimeDistributed(TransformersBackbone("microsoft/efficientnet-v6", num_unfrozen_stages=1))
neck = GRUNeck(num_features=backbone.num_features, hidden_size=128, num_layers=2, return_last=True)

datamodule = VideoDataModule(
    train_root="toy_dataset/train",
    val_root="toy_dataset/val",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=neck.hidden_size, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head, neck)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(
    datamodule,
    model,
    max_epochs=8
)

trainer.fit()

```