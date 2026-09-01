#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH"

if ! python3 -m streamlit --version >/dev/null 2>&1 && ! which streamlit >/dev/null 2>&1; then
  echo "Streamlit not found. Bootstrapping Python environment..."
  if ! which pip >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user --no-setuptools --no-wheel
  fi
  ~/.local/bin/pip install --no-cache-dir streamlit scikit-learn pandas numpy matplotlib
fi

exec python3 -m streamlit run app.py \
  --server.port 3000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
