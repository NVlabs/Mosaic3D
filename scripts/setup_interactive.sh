#!/bin/bash

PORT=$1

# Install pre-commit hooks
pre-commit install

# Install packages
apt-get update && \
apt-get install -y --no-install-recommends \
    openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Switch to root user
sudo -i

# SSH configuration
echo "Port $PORT" >> /etc/ssh/sshd_config
echo "LogLevel DEBUG3" >> /etc/ssh/sshd_config

# SSH server setup
mkdir -p /run/sshd
ssh-keygen -A
service ssh start

# Start SSH server
/usr/sbin/sshd -D -p "$PORT" -e > /dev/null 2>&1 &

# Print tunneling instructions
echo "
Running a SSH server on

    Host: $(hostname -s)
    Port: $PORT
"

# Start an interactive bash shell
exec /bin/bash
