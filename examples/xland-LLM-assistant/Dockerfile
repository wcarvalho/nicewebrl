# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser


# Copy application code
COPY . .

RUN pip install -e ".[xland-assistant]"

RUN cd examples/xland-LLM-assistant

# Create data directory and set permissions
RUN mkdir -p data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for web server
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with error handling
CMD ["sh", "-c", "python web_app_assistant.py || echo 'Application failed to start' && exit 1"] 