# Img2text

```python
# test script
blip_model, clip_model, clip_preprocess, dtype = load()
import glob
img_path = "./test.jpg"
pil_image = Image.open(img_path).convert("RGB")
res = img2text(pil_image, clip_model, clip_preprocess, blip_model, dtype)
print(res)
```