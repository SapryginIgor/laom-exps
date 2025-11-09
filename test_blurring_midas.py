# depth_blur_midas.py
import cv2, torch, numpy as np

# 1) Load model + transform
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()  # DPT_Large = slower, a bit sharper
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform  # returns [1,3,H,W] tensor already

# 2) Read image
img_bgr = cv2.imread("images/example.jpg")
assert img_bgr is not None, "Failed to read input.jpg"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3) Preprocess (NO extra unsqueeze)
inp = transform(img_rgb).to(device)  # shape: [1,3,384,384] typically

# 4) Predict relative depth
with torch.no_grad():
    pred = midas(inp)                       # [1, Hm, Wm]
    pred = torch.nn.functional.interpolate( # upsample to original size
        pred.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze(1)                            # [1, H, W]
depth = pred[0].cpu().numpy()

# the more the number, the more blurry the result
p = np.percentile(depth, 70)
mask = (depth <= p).astype(np.uint8)
# No mask smoothing - keep sharp edges to preserve foreground details
# mask = cv2.GaussianBlur(mask, (5,5), 1)
mask3 = mask[..., None]

# 6) Composite: blur background
blurred = cv2.GaussianBlur(img_bgr, (31,31), 11)
out = (img_bgr*(1-mask3) + blurred*(mask3)).astype(np.uint8)
cv2.imwrite("output.jpg", out)
print("Saved: output.jpg")
