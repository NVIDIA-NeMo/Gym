version: "3.8"

networks:
  $prefix-net:
    driver: bridge

volumes:
  $prefix-config:
    name: $prefix-config
  $prefix-data:
    name: $prefix-data
  $prefix-logs:
    name: $prefix-logs
  $prefix-plugins:
    name: $prefix-plugins
  $prefix-client-plugins:
    name: $prefix-client-plugins
  $prefix-bleve-indexes:
    name: $prefix-bleve-indexes
  $prefix-pg-data:
    name: $prefix-pg-data

services:
  $prefix-postgres:
    image: mw-mattermost-postgres:latest
    container_name: $prefix-postgres
    networks:
      - $prefix-net
    restart: unless-stopped
    environment:
      - POSTGRES_USER=mmuser
      - POSTGRES_PASSWORD=mmuser_password
      - POSTGRES_DB=mattermost
      - POSTGRES_HOST_AUTH_METHOD=md5
    volumes:
      - $prefix-pg-data:/var/lib/postgresql/data
    # Ensure pg_hba.conf allows container-network hosts (baked image only allows localhost).
    command: >
      sh -c "grep -q 'host all all 0.0.0.0/0 md5' /var/lib/postgresql/data/pg_hba.conf 2>/dev/null ||
             echo 'host all all 0.0.0.0/0 md5' >> /var/lib/postgresql/data/pg_hba.conf;
             exec docker-entrypoint.sh postgres"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mmuser -d mattermost"]
      interval: 10s
      retries: 10
      start_period: 30s
      timeout: 5s

  $prefix:
    image: mw-mattermost:latest
    container_name: $prefix
    networks:
      - $prefix-net
    restart: unless-stopped
    user: root
    command: ["mattermost", "server"]
    security_opt:
      - no-new-privileges:true
    pids_limit: 200
    tmpfs:
      - /tmp
    environment:
      - MM_SQLSETTINGS_DRIVERNAME=postgres
      - MM_SQLSETTINGS_DATASOURCE=postgres://mmuser:mmuser_password@$prefix-postgres:5432/mattermost?sslmode=disable&connect_timeout=10
      - MM_BLEVESETTINGS_INDEXDIR=/mattermost/bleve-indexes
      - MM_SERVICESETTINGS_SITEURL=http://$hostname:$port
      - MM_SERVICESETTINGS_LISTENADDRESS=:8065
    volumes:
      - $prefix-config:/mattermost/config:rw
      - $prefix-data:/mattermost/data:rw
      - $prefix-logs:/mattermost/logs:rw
      - $prefix-plugins:/mattermost/plugins:rw
      - $prefix-client-plugins:/mattermost/client/plugins:rw
      - $prefix-bleve-indexes:/mattermost/bleve-indexes:rw
    ports:
      - "$port:8065"
    depends_on:
      $prefix-postgres:
        condition: service_healthy
