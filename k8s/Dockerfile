FROM registry.access.redhat.com/ubi9/python-312 AS base-build

USER 0
WORKDIR /opt/app-root/src

COPY pyproject.toml README.md LICENSE ./
COPY nemo_gym/ nemo_gym/

RUN mkdir -p cache && pip install --no-cache-dir -e "."

# --- Base runtime ---
FROM registry.access.redhat.com/ubi9/python-312 AS base-runtime

LABEL io.k8s.display-name="NeMo Gym" \
      io.k8s.description="NeMo-Gym server for RL training environments" \
      io.openshift.tags="python,fastapi,ray,nemo-gym" \
      io.openshift.expose-services="8080:http"

USER 0
WORKDIR /opt/app-root/src

COPY --from=base-build /opt/app-root/lib /opt/app-root/lib
COPY --from=base-build /opt/app-root/bin /opt/app-root/bin

COPY nemo_gym/ nemo_gym/
COPY pyproject.toml README.md LICENSE ./
COPY resources_servers/ resources_servers/
COPY responses_api_agents/ responses_api_agents/
COPY responses_api_models/ responses_api_models/

RUN mkdir -p cache && pip install --no-cache-dir -e . && \
    chgrp -R 0 /opt/app-root && \
    chmod -R g=u /opt/app-root && \
    chmod g=u /etc/passwd

COPY k8s/entrypoint.sh /opt/app-root/entrypoint.sh
RUN chmod +x /opt/app-root/entrypoint.sh

ENV RAY_TMPDIR=/tmp \
    PYTHONUNBUFFERED=1 \
    HOME=/opt/app-root/src

USER 1001
ENTRYPOINT ["/opt/app-root/entrypoint.sh"]

# --- Resources server (code_gen) ---
FROM base-runtime AS resources

USER 0
RUN pip install --no-cache-dir "ray[default]" numpy==2.2.6 && \
    chgrp -R 0 /opt/app-root/lib && \
    chmod -R g=u /opt/app-root/lib
USER 1001

EXPOSE 9080
CMD ["python", "resources_servers/code_gen/app.py"]

# --- Agent server (simple_agent) ---
FROM base-runtime AS agent

EXPOSE 8080
CMD ["python", "responses_api_agents/simple_agent/app.py"]

# --- Model proxy (vllm_model) ---
FROM base-runtime AS model

EXPOSE 8080
CMD ["python", "responses_api_models/vllm_model/app.py"]

# --- All-in-one (testing) ---
FROM base-runtime AS all-in-one

USER 0
RUN pip install --no-cache-dir "ray[default]" numpy==2.2.6 && \
    chgrp -R 0 /opt/app-root/lib && \
    chmod -R g=u /opt/app-root/lib
USER 1001

EXPOSE 8080
