version: "3.8"

networks:
  $prefix-net:
    driver: bridge

services:
  $prefix-mariadb:
    image: mw-owncloud-mariadb:latest
    container_name: $prefix-mariadb
    networks:
      - $prefix-net
    restart: unless-stopped
    environment:
      - MYSQL_ROOT_PASSWORD=owncloud
      - MYSQL_USER=owncloud
      - MYSQL_PASSWORD=owncloud
      - MYSQL_DATABASE=owncloud
    command: ["--max-allowed-packet=128M", "--innodb-log-file-size=64M"]
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-u", "root", "--password=owncloud"]
      interval: 10s
      retries: 10
      start_period: 30s
      timeout: 5s

  $prefix-redis:
    image: redis:6
    container_name: $prefix-redis
    networks:
      - $prefix-net
    restart: unless-stopped
    command: ["--databases", "1"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 5
      timeout: 5s

  $prefix:
    image: mw-owncloud-server:latest
    container_name: $prefix
    networks:
      - $prefix-net
    restart: unless-stopped
    environment:
      - OWNCLOUD_DOMAIN=$hostname:$port
      - OWNCLOUD_TRUSTED_DOMAINS=localhost,127.0.0.1,$hostname
      - OWNCLOUD_DB_TYPE=mysql
      - OWNCLOUD_DB_NAME=owncloud
      - OWNCLOUD_DB_USERNAME=owncloud
      - OWNCLOUD_DB_PASSWORD=owncloud
      - OWNCLOUD_DB_HOST=$prefix-mariadb
      - OWNCLOUD_ADMIN_USERNAME=admin
      - OWNCLOUD_ADMIN_PASSWORD=admin
      - OWNCLOUD_MYSQL_UTF8MB4=true
      - OWNCLOUD_REDIS_ENABLED=true
      - OWNCLOUD_REDIS_HOST=$prefix-redis
    ports:
      - "$port:8080"
    depends_on:
      $prefix-mariadb:
        condition: service_healthy
      $prefix-redis:
        condition: service_healthy
