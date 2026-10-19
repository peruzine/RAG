# 多阶段构建：核心零第三方依赖，所以这个镜像非常小
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGKB_API_HOST=0.0.0.0 \
    RAGKB_API_PORT=8000 \
    RAGKB_DATA_DIR=/app/data

WORKDIR /app

# ---- 依赖层 ---------------------------------------------------------------
# 核心无需任何依赖；下面这行按需打开以支持 Word/Excel/PDF 解析与 numpy 加速。
# 为了减小镜像体积，默认只装解析类依赖（不含 torch）。
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install ".[parse,speed]"

# ---- 应用层 ---------------------------------------------------------------
COPY ragkb ./ragkb
COPY scripts ./scripts
COPY eval ./eval
COPY data/corpus ./data/corpus
COPY tests ./tests

# 以非 root 运行（生产基本要求）
RUN useradd --create-home --uid 10001 ragkb \
    && mkdir -p /app/data \
    && chown -R ragkb:ragkb /app
USER ragkb

EXPOSE 8000

# 容器健康检查直接用 /health，不依赖 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# 首次启动自动入库示例语料（幂等，已入库会全部 skipped）；随后挂起服务
CMD ["sh", "-c", "python -m ragkb.cli demo && python -m ragkb.cli serve --host 0.0.0.0 --port 8000"]
