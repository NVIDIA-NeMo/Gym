#!/bin/bash
if ! whoami &> /dev/null 2>&1; then
  if [ -w /etc/passwd ]; then
    echo "nemo:x:$(id -u):0:nemo user:${HOME}:/sbin/nologin" >> /etc/passwd
  fi
fi
exec "$@"
