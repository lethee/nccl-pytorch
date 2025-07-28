#!/bin/bash

echo "Setting ulimit..."
ulimit -n 65535  # nofile
ulimit -l unlimited # memlock

exec "$@"
