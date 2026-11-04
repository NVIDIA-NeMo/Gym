# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serve httpbin.org from a local httpbin while grading psf/requests (inserted into eval.sh by swebench_patches.py).
# Runs in its own `bash -e` so a failure stops setup without aborting eval.sh; /etc/hosts changes only once
# the local server answers, so on failure the tests use the real httpbin.org as before.
bash -e <<'LOCAL_HTTPBIN' || echo "local httpbin setup failed; tests will use the real httpbin.org"
D=/opt/local-httpbin
PY=/opt/miniconda3/bin/python3
mkdir -p $D && cd $D
curl -LsSf -m 60 https://astral.sh/uv/0.13.7/install.sh | env UV_INSTALL_DIR=$D/bin UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
$D/bin/uv pip install -q --python $PY --target $D/pkgs httpbin==0.10.4 gunicorn==23.0.0

# A throwaway CA, trusted by every bundle these requests versions read (their own cacert.pem predates today's CAs).
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 7 -subj /CN=local-httpbin-ca 2>/dev/null
openssl req -newkey rsa:2048 -nodes -keyout srv.key -out srv.csr -subj /CN=httpbin.org 2>/dev/null
printf 'subjectAltName=DNS:httpbin.org,DNS:www.httpbin.org\nbasicConstraints=CA:FALSE\n' > ext.cnf
openssl x509 -req -in srv.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out srv.crt -days 7 -extfile ext.cnf 2>/dev/null
cat srv.crt ca.crt > chain.crt
for f in /testbed/requests/cacert.pem $(python -c 'import certifi; print(certifi.where())' 2>/dev/null) \
    /etc/ssl/certs/ca-certificates.crt ${REQUESTS_CA_BUNDLE:-}; do
  if [ -f "$f" ]; then cat ca.crt >> "$f"; fi
done

# test_mixed_case_scheme_acceptable sends http:// and https:// to the same host, so serve both ports.
G="env PYTHONPATH=$D/pkgs $PY -m gunicorn -D -w 4 -k gthread --threads 8 --timeout 120 httpbin:app"
$G -b 0.0.0.0:80
$G -b 0.0.0.0:443 --certfile $D/chain.crt --keyfile $D/srv.key
for i in $(seq 300); do
  curl -sf -o /dev/null --resolve httpbin.org:443:127.0.0.1 --cacert ca.crt https://httpbin.org/get && break
  sleep 0.1
done
curl -sf -o /dev/null --resolve httpbin.org:443:127.0.0.1 --cacert ca.crt https://httpbin.org/get
curl -sf -o /dev/null --resolve httpbin.org:80:127.0.0.1 http://httpbin.org/get
echo "127.0.0.1 httpbin.org www.httpbin.org" >> /etc/hosts
echo "local httpbin serving httpbin.org"
LOCAL_HTTPBIN
