# Continuous Action Detection Based on Inertial Sensors

Source code for paper "Multistream Temporal Convolutional Network for Correct/Incorrect Patient Transfer Action
Detection Using Body Sensor Network" on public action detection dataset C-MHAD

To give multimodal signals full play in fine-grained action detection,
we propose a novel temporal convolutional network by designing a channel attention-based multistream structure.
Compared to the state-of-the-art methods,
our method achieves the best performance, **using only inertial data**,
with F1 score of **95.3% for the action set of transition movements** and **98.5% for the action set of smart TV gestures**.
We surpass the benchmark performance (F1 score of 78.8% for transition movements and 81.8% for smart TV gestures) using fused data, i.e., both inertial and video signals, by a large margin.
## Prerequisites

Install the dependent packages:

```bash
conda create -n cad python=3.8
conda activate cad
pip install -r requirements.txt
```

Download inertial data of C-MHAD
directly from [here](https://drive.google.com/file/d/1nXnlT0U68v-1OOPdjke3bMfIbL5CSPaq/view?usp=sharing) (or
from [C-MHAD project page](https://personal.utdallas.edu/~kehtar/C-MHAD.html)).

## Training

Train on Transition Movements part of C-MHAD:

```bash
python main.py --dataset CMHAD_Transition
```

Train on Smart TV Gesture part of C-MHAD:

```bash
python main.py --dataset CMHAD_Gesture
```

## Testing

Please download [checkpoints](https://drive.google.com/file/d/1WoMnPo5lmWBlIHgm-I20aqsEqnLuHGaP/view?usp=sharing) and unzip it under the main directory.

Run the pretrained model on Transition Movements part of C-MHAD:

```bash
python main.py --dataset CMHAD_Transition --test_only --test_checkpoint ./checkpoints/MSSTCN_CMHAD_Transition.tar
```

Run the pretrained model on Smart TV Gesture part part of C-MHAD:

```bash
python main.py --dataset CMHAD_Gesture --test_only --test_checkpoint ./checkpoints/MSSTCN_CMHAD_Gesture.tar
```

## Citing

If you find our code is useful for you, please consider citing:

```bibtex
@article{zhong2021multistream,
  title={Multistream Temporal Convolutional Network for Correct/Incorrect Patient Transfer Action Detection Using Body Sensor Network},
  author={Zhong, Zhihang and Lin, Chingszu and Kanai-Pak, Masako and Maeda, Jukai and Kitajima, Yasuko and Nakamura, Mitsuhiro and Kuwahara, Noriaki and Ogata, Taiki and Ota, Jun},
  journal={IEEE Internet of Things Journal},
  year={2021},
  publisher={IEEE}
}
```
