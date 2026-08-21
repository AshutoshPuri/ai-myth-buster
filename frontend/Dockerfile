# ========================================================
# Stage 1 — Build React application
# ========================================================

FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for dependency caching.
COPY frontend/package*.json ./

RUN npm ci

# API URL used by Vite during the build.
ARG VITE_API_BASE_URL=http://localhost:8000

ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Copy frontend source.
COPY frontend ./

# Build production React application.
RUN npm run build

# ========================================================
# Stage 2 — Serve with Nginx
# ========================================================

FROM nginx:alpine

# Remove default nginx website.
RUN rm -rf /usr/share/nginx/html/*

# Copy React production build.
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx configuration.
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
