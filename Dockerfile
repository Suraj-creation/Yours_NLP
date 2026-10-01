# One container runs the whole site: FastAPI serves the API and the built React app.
#   docker build -t learner-language-lab .
#   docker run -p 8000:8000 learner-language-lab      -> http://localhost:8000
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'treebank', 'universal_tagset', 'words')]"
COPY nlp_core/ nlp_core/
COPY api/ api/
COPY config/ config/
COPY data/ data/
COPY data_scale/ data_scale/
COPY gold/ gold/
COPY results/ results/
COPY scripts/ scripts/
COPY --from=web /web/dist web/dist
EXPOSE 8000
# Pipeline indexes are rebuilt into .cache on first use (a few seconds each); the heavy-dataset
# indexes need `python scripts/run_scale.py` once (about eight minutes) for MathDial/TalkMoves search.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
