import glob, statistics as st, collections, sys
from PIL import Image
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(sys.argv[1], trust_remote_code=True)
vt=[];cr=[];px=[]
files = sorted(glob.glob(sys.argv[2]+"/**/*.*", recursive=True))
files=[f for f in files if f.lower().endswith(('.jpg','.png','.jpeg'))]
for f in files:
    im = Image.open(f).convert("RGB")
    o = p(["x(<image>./</image>)"],[[im]],max_slice_nums=9,return_tensors="pt")
    b = o["image_bound"][0]
    vt.append(int(sum(int(e-s) for s,e in b.tolist()))); cr.append(len(o["pixel_values"][0])); px.append(im.size[0]*im.size[1])
print(len(files),"images; visual tokens min/mean/median/max",min(vt),round(st.mean(vt),1),st.median(vt),max(vt))
print("crops (incl. overview)", collections.Counter(cr))
print("source pixels min/median/max", min(px), st.median(px), max(px), "frac > 1.8M px", sum(x>9*448*448 for x in px)/len(px), "frac>2.1M", sum(x>2097152 for x in px)/len(px))
