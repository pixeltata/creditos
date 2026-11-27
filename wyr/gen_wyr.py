import torch, time, os, json
from diffusers import ZImagePipeline
t0 = time.time()
pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', torch_dtype=torch.bfloat16).to('cuda')
print('carga', round(time.time() - t0), 's', flush=True)
OUT = '/workspace/acestep/salida/wyr_img'; os.makedirs(OUT, exist_ok=True)
P = json.load(open('/workspace/acestep/wyr_prompts.json'))
for i, (name, pr) in enumerate(P.items()):
    if os.path.exists(f'{OUT}/{name}.jpg'): continue
    im = pipe(prompt=pr, height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0, generator=torch.Generator('cuda').manual_seed(7 + i)).images[0]
    im.convert('RGB').save(f'{OUT}/{name}.jpg', quality=92); print(i, name, flush=True)
print('GEN_DONE', round(time.time() - t0), 's', flush=True)
