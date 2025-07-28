FROM python:3.13-slim

# Build arguments for image metadata
ARG NAME
ARG BRANCH
ARG HASH
ARG DOCKERTAG
ARG CREATED

# Set image metadata labels following OCI standards
LABEL org.opencontainers.image.vendor="Benchmarker"
LABEL org.opencontainers.image.title=${NAME}
LABEL org.opencontainers.image.version=${DOCKERTAG}
LABEL org.opencontainers.image.created=${CREATED}
LABEL org.opencontainers.image.revision=${HASH}
LABEL org.opencontainers.image.ref.name=${BRANCH}

# Set environment variables for application versioning and tracking
ENV NAME=${NAME}
ENV BRANCH=${BRANCH}
ENV HASH=${HASH}
ENV DOCKERTAG=${DOCKERTAG}
ENV CREATED=${CREATED}

# Configure timezone settings
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies including poppler for PDF processing
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    curl \
    git \
    # PDF processing dependencies
    poppler-utils \
    # Image processing dependencies
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    # Additional utilities
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /opt/benchmarker

# Copy application code and Install Python dependencies
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
