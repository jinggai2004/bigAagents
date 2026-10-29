# bigAagents — A股多Agent分析系统
# 基础镜像含 Python 3.12 + Node.js（pywencai 问财令牌需要 Node 运行时）+ 上海时区
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# Node.js（PyExecJS 依赖）+ 时区数据 + curl（健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs \
        tzdata \
        curl \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目代码（data/ 不入镜像：运行时由卷挂载提供，避免用户数据打进镜像）
COPY core/ ./core/
COPY db/ ./db/
COPY llm/ ./llm/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY ui/ ./ui/
COPY .env.example ./

# 运行时产出与数据通过卷挂载（见 docker-compose.yml）
RUN mkdir -p data output logs temp

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

# 默认启动 web（调度器由 compose 以同镜像另起服务）
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
