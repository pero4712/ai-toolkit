ARG BASE_IMAGE=pero4712/aitoolkit-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

ARG CACHEBUST=1234
ARG GIT_COMMIT=rp
RUN echo "Cache bust: ${CACHEBUST}" && \
    git clone https://github.com/pero4712/ai-toolkit.git && \
    cd ai-toolkit && \
    git checkout ${GIT_COMMIT}

# Build UI. update_db runs first because it does `prisma generate`, which
# creates the @prisma/client types the TS build needs.
WORKDIR /app/ai-toolkit/ui
RUN npm install && \
    npm run update_db && \
    npm run build

# Expose port (assuming the application runs on port 3000)
EXPOSE 8675

WORKDIR /

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
