# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/torch:2.8.0-cu128-amd64

# 设置工作目录
WORKDIR /app

# 安装依赖
RUN apt update && \
    apt install -y libgl1 && \
    /app/.venv/bin/pip install ultralytics --no-cache-dir && \
    /app/.venv/bin/pip install git+https://github.com/ultralytics/CLIP.git --no-cache-dir && \
    /app/.venv/bin/pip cache purge && \
    rm -rf wheels

ENV ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./
RUN mkdir -p src/artemis_cve && \
    touch src/artemis_cve/__init__.py && \
    poetry install --no-root

# 拷贝源代码文件
COPY . .

# 安装当前包
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 50051

# 默认入口
CMD ["poetry", "run", "python", "-m", "artemis_cve.commands.app"]
