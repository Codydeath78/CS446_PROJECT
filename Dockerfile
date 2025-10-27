#this does Build stage.
FROM node:18-alpine AS builder
WORKDIR /app

#this does install deps.
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* .npmrc* ./
RUN \
  if [ -f package-lock.json ]; then npm ci --legacy-peer-deps; \
  elif [ -f pnpm-lock.yaml ]; then npm i -g pnpm && pnpm i --frozen-lockfile; \
  elif [ -f yarn.lock ]; then yarn install --frozen-lockfile; \
  else npm i --legacy-peer-deps; fi

#this copies source and build.
COPY . .
#this ensures production build.
ENV NODE_ENV=production
RUN npm run build

#this ensures runtime stage.
FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
#this does cloud run sets PORT and next must listen on it.
ENV PORT=8080
ENV HOSTNAME=0.0.0.0

#this copies the minimal standalone output.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 8080
CMD ["node", "server.js"]