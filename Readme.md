# Img2text


## Execute command
```python
from img2text import *
# test script
blip_model, clip_model, clip_preprocess, dtype = load()
import glob
img_path = "./test.jpg"
pil_image = Image.open(img_path).convert("RGB")
res = img2text(pil_image, clip_model, clip_preprocess, blip_model, dtype)
print(res)
```
## Demo
![test img](./test.jpg)

>Description: `a woman with a ponytail and a white shirt on posing for a picture with a white background and a white background, detailed face, Adrian Zingg, classical realism, a photorealistic painting.`
