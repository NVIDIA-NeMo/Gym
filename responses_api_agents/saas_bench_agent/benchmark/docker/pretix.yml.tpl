# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/docker/pretix.yml.tpl
# Original license: Apache-2.0; Copyright: SaaS-Bench Authors.
version: "3.8"

networks:
  $prefix-net:
    driver: bridge

volumes:
  $prefix-data:
    name: $prefix-data
  $prefix-db-data:
    name: $prefix-db-data
  $prefix-redis-data:
    name: $prefix-redis-data

services:
  $prefix-db:
    image: postgres:13-alpine
    container_name: $prefix-db
    networks:
      - $prefix-net
    restart: unless-stopped
    environment:
      - POSTGRES_DB=pretix
      - POSTGRES_USER=pretix
      - POSTGRES_PASSWORD=pretix_pass
    volumes:
      - $prefix-db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pretix -d pretix"]
      interval: 10s
      retries: 10
      start_period: 15s
      timeout: 5s

  $prefix-redis:
    image: redis:7-alpine
    container_name: $prefix-redis
    networks:
      - $prefix-net
    restart: unless-stopped
    volumes:
      - $prefix-redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 5
      timeout: 5s

  $prefix:
    image: mw-pretix:latest
    container_name: $prefix
    networks:
      - $prefix-net
    restart: unless-stopped
    environment:
      - TZ=Asia/Shanghai
      - PRETIX_URL=http://$hostname:$port
    # Patch the baked /etc/pretix/pretix.cfg (which uses "localhost") to point
    # to the networked postgres / redis services before invoking the upstream
    # entrypoint.
    entrypoint:
      - /bin/bash
      - -c
      - |
        sed -i 's/^host=localhost/host=$prefix-db/' /etc/pretix/pretix.cfg
        sed -i 's|^location=redis://localhost:6379|location=redis://$prefix-redis:6379|' /etc/pretix/pretix.cfg
        sed -i 's|^backend=redis://localhost:6379|backend=redis://$prefix-redis:6379|' /etc/pretix/pretix.cfg
        sed -i 's|^broker=redis://localhost:6379|broker=redis://$prefix-redis:6379|' /etc/pretix/pretix.cfg
        # Benchmark tasks create paid events and require them to go live. Seed
        # newly-created events with Pretix's built-in manual payment provider,
        # which needs no external account or credentials.
        sed -i "/plugins = models.TextField(/,/verbose_name=_(\"Plugins\")/ s/null=False, blank=True,/null=False, blank=True, default='pretix.plugins.manualpayment',/" /pretix/src/pretix/base/models/event.py
        sed -i "s/return self.settings.get('_enabled', as_type=bool)/return self.settings.get('_enabled', default=self.identifier == 'manual', as_type=bool)/" /pretix/src/pretix/base/payment.py
        exec /entrypoint-mw.sh all
    volumes:
      - $prefix-data:/data
    ports:
      - "$port:80"
    depends_on:
      $prefix-db:
        condition: service_healthy
      $prefix-redis:
        condition: service_healthy
