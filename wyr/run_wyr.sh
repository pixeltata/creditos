export HF_HOME=/root/zi/hf HF_HUB_ENABLE_HF_TRANSFER=0 UV_CACHE_DIR=/root/zi/uvcache
mkdir -p /root/zi && cd /root/zi
pip install -q uv
[ -d venv ] || uv venv -q venv --python 3.11
uv pip install -q --python venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128 && echo TORCH_OK
uv pip install -q --python venv/bin/python "git+https://github.com/huggingface/diffusers" transformers accelerate sentencepiece safetensors && echo LIBS_OK
venv/bin/python /workspace/acestep/gen_wyr.py
echo ALL_DONE
cd /workspace/acestep/salida && rm -f wyr_img.zip && /root/zi/venv/bin/python -c "import shutil;shutil.make_archive('wyr_img','zip','wyr_img')" && ls -la wyr_img.zip && echo ZIP_DONE
