# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/docker/onlyoffice.yml.tpl
# Original license: Apache-2.0; Copyright: SaaS-Bench Authors.
version: "3.8"

networks:
  $prefix-net:
    driver: bridge

services:
  $prefix-mysql:
    image: oo-bundle-mysql:latest
    container_name: $prefix-mysql
    networks:
      - $prefix-net
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: my-secret-pw
      MYSQL_DATABASE: onlyoffice
      MYSQL_USER: onlyoffice_user
      MYSQL_PASSWORD: onlyoffice_pass
    volumes:
      - ${prefix}_mysql_data:/var/lib/mysql
    command:
      - --sql_mode=
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_general_ci
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-uroot", "-pmy-secret-pw"]
      interval: 10s
      retries: 15
      start_period: 60s
      timeout: 5s

  $prefix-elasticsearch:
    image: oo-bundle-es:latest
    container_name: $prefix-elasticsearch
    networks:
      - $prefix-net
    restart: always
    environment:
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - ${prefix}_es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health | grep -qE '\"status\":\"(green|yellow)\"'"]
      interval: 15s
      retries: 10
      start_period: 60s
      timeout: 10s

  $prefix-documentserver:
    image: oo-bundle-ds:latest
    container_name: $prefix-documentserver
    networks:
      - $prefix-net
    restart: always
    environment:
      - JWT_ENABLED=false
      - ALLOW_PRIVATE_IP_ADDRESS=true
    volumes:
      - ${prefix}_ds_data:/var/www/onlyoffice/Data
      - ${prefix}_ds_logs:/var/log/onlyoffice
      - ${prefix}_ds_cache:/var/lib/onlyoffice/documentserver/App_Data/cache/files
      - ${prefix}_ds_files:/var/www/onlyoffice/documentserver-example/public/files
      - ${prefix}_ds_fonts:/usr/share/fonts
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/info/info.json"]
      interval: 30s
      retries: 5
      start_period: 60s
      timeout: 10s

  $prefix-community:
    image: oo-bundle-community2:latest
    container_name: $prefix-community
    networks:
      - $prefix-net
    restart: always
    privileged: true
    volumes:
      - /sys/fs/cgroup:/sys/fs/cgroup:rw
      - ${prefix}_community_data:/var/www/onlyoffice/Data
      - ${prefix}_community_logs:/var/log/onlyoffice
      - ${prefix}_community_letsencrypt:/etc/letsencrypt
    entrypoint:
      - /bin/bash
      - -c
      - |
        chown -R onlyoffice:onlyoffice /var/www/onlyoffice/Data /var/log/onlyoffice 2>/dev/null || true
        # cgroup v2 hosts: the image's /proc/1/cgroup container-detection fails,
        # so it sets DOCKER_ENABLED=false and never execs systemd (exits 0 ->
        # restart loop). Run the setup script, then exec systemd ourselves.
        /app/run-community-server.sh || true
        exec /lib/systemd/systemd
    environment:
      - MYSQL_SERVER_HOST=$prefix-mysql
      - MYSQL_SERVER_PORT=3306
      - MYSQL_SERVER_DB_NAME=onlyoffice
      - MYSQL_SERVER_USER=onlyoffice_user
      - MYSQL_SERVER_PASS=onlyoffice_pass
      - ELASTICSEARCH_SERVER_HOST=$prefix-elasticsearch
      - ELASTICSEARCH_SERVER_HTTPPORT=9200
      - DOCUMENT_SERVER_ENABLED=true
      - DOCUMENT_SERVER_HOST=$prefix-documentserver
      - DOCUMENT_SERVER_PROTOCOL=http
      - DOCUMENT_SERVER_API_URL=/ds-vpath
      - DOCUMENT_SERVER_JWT_ENABLED=false
    ports:
      - "$port:80"
    depends_on:
      $prefix-mysql:
        condition: service_healthy
      $prefix-elasticsearch:
        condition: service_healthy
      $prefix-documentserver:
        condition: service_healthy

volumes:
  ${prefix}_mysql_data:
  ${prefix}_es_data:
  ${prefix}_community_data:
  ${prefix}_community_logs:
  ${prefix}_community_letsencrypt:
  ${prefix}_ds_data:
  ${prefix}_ds_logs:
  ${prefix}_ds_cache:
  ${prefix}_ds_files:
  ${prefix}_ds_fonts:
