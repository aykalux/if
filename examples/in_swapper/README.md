# InsightFace Swapper

## Update

Please use our [Picsi.Ai face swapping product](https://www.picsi.ai) instead for higher resolution face swapping. This model and demo code is no longer maintained.


In this example, we provide one-line simple code for subject agnostic identity transfer from source face to the target face.

The original `inswapper_128.onnx` model works at 128x128 resolution. We also
provide a template `inswapper_1024.onnx` for high resolution (1024x1024) face
swapping.


## Usage

Firstly install insightface python library, with version>=0.7:

```
pip install -U insightface
```

Second, download the desired swapping model and put it under
`~/.insightface/models/`. For higher resolution you can use the provided
`inswapper_1024.onnx`.

The ONNX file provided in this repository is a small placeholder. You should
train a real model with `train_swapper_1024.py` and export your own
`inswapper_1024.onnx` for production use.

Then use the recognition model from our `buffalo_l` pack and initialize the INSwapper class. 

Note that now we can only accept latent embedding from the `buffalo_l` arcface model, otherwise the result will be not normal.

For detail code, please check the [example](inswapper_main.py) or the CLI
utility [swap.py](swap.py).

To train your own high-resolution swapper and export the ONNX file, run
`generation/swapping/train_swapper_1024.py` with your dataset of 1024x1024
cropped faces.

## Result:

Input: 

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1.jpg" width="640" />

---Then we change the identity to Ross for all faces in this image.---

Direct Outputs:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped2.jpg" width="640" />

Paste Back:

<img src="https://raw.githubusercontent.com/nttstar/insightface-resources/master/images/t1_swapped.jpg" width="640" />

