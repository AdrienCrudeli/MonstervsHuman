# ZombieNet

A four-class image classifier that sorts pictures into `human`, `zombie`, `vampire` and `skeleton`.
Built as a computer vision coursework project, with one constraint that shaped most of the work:
there is no public dataset for this, so the dataset had to be built first.

Final result: **88.8% validation accuracy**, macro F1 0.885, on 143 validation images.

## How it works

The classifier uses MobileNetV2 pretrained on ImageNet, with the convolutional base frozen and a
small head on top (global average pooling, a 128-unit dense layer, dropout at 0.3, softmax over
four classes). Only the head is trained, 2.4M total parameters, of which very few are trainable.

Transfer learning is the only reasonable choice here. With 253 training images, training a
convolutional network from scratch would overfit immediately, whereas MobileNetV2 already knows
about edges, textures and faces and only needs to learn the four-way decision on top.

Training runs for up to 20 epochs with Adam at 1e-4, early stopping on validation loss with a
patience of 5, and a checkpoint that keeps the best weights rather than the last ones.

## The dataset

`crawler.py` builds it. It queries Bing Images through `icrawler` with one search string per class
and downloads 150 training and 40 validation images each. The raw download is noisy — wrong
subjects, duplicates, broken files, so everything was reviewed and cleaned by hand afterwards.

What survives: **396 images**, 253 for training and 143 for validation, roughly balanced across
the four classes.

```bash
python crawler.py      # writes into dataset/train and dataset/val
```

Re-running the crawler will not reproduce the exact dataset — Bing results shift over time, and
the manual cleaning pass is not scripted.

## Running it

```bash
pip install tensorflow scikit-learn matplotlib seaborn pillow icrawler
jupyter notebook MonsterVsHuman.ipynb
```

Run the notebook top to bottom. It expects `dataset/train` and `dataset/val` next to it, each with
one subdirectory per class. Training saves the best checkpoint to `zombienet_best.keras`, then the
evaluation cells print the per-class precision/recall/F1 table and draw the confusion matrix.

## Live webcam demo

`webcam_demo.py` runs the trained model against a webcam feed. It reads frames through `ffmpeg`,
displays them in a Tkinter window with the predicted class and confidence, and runs inference at
5 fps on a 30 fps capture so the preview stays smooth.

```bash
python webcam_demo.py
```

It needs `ffmpeg` on the PATH and `zombienet_best.keras` in the project directory. The capture
pipeline is written for macOS (AVFoundation input); other platforms need a different ffmpeg input
device.

## Files

|           File         |                       Role                       |
| -----------------------| ------------------------------------------------ |
| `MonsterVsHuman.ipynb` | Dataset exploration, model, training, evaluation |
| `crawler.py`           | Builds the dataset from Bing Images              |
| `webcam_demo.py`       | Live inference on a webcam feed                  |
| `zombienet_best.keras` | Best checkpoint from training                    |

## Notes

The confusion matrix shows most remaining errors between `zombie` and `vampire`, which is
unsurprising: both classes are mostly costume and makeup photography, and the visual boundary
between them is genuinely thin. More data would help more than a bigger model.
