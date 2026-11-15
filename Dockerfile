# Hugging Face Spaces (SDK docker, CPU basic). Receta oficial: usuario uid 1000.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements-docker.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-docker.txt

# CLIP en build: la primera foto no descarga nada
RUN python -c "from transformers import CLIPModel, CLIPProcessor; m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)"

# runtime sin red: CLIP ya está en el cache
ENV HF_HUB_OFFLINE=1

COPY --chown=user src/ src/
COPY --chown=user web/ web/
COPY --chown=user data/processed/catalogo.csv data/processed/transacciones.csv data/processed/
COPY --chown=user data/embeddings/*.npz data/embeddings/
COPY --chown=user data/modelos/*.pt data/modelos/

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
