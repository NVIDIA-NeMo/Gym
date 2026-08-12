# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM mambaorg/micromamba:2.3.2

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates coreutils curl file git jq less procps unzip vim wget \
        libcairo2 libgl1 libglib2.0-0 libpango-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.11 pip \
        numpy pandas scipy matplotlib seaborn scikit-learn scikit-image \
        biopython pysam pyvcf3 plotly statsmodels networkx h5py jupyterlab \
        r-base=4.3 r-data.table r-tidyverse \
        bioconductor-deseq2 bioconductor-edger bioconductor-limma \
        bioconductor-genomicranges bioconductor-clusterprofiler bioconductor-enrichplot \
        bioconductor-go.db bioconductor-tximport bioconductor-biostrings \
        bioconductor-org.hs.eg.db bioconductor-org.mm.eg.db bioconductor-lionessr \
        samtools bcftools bedtools bwa bowtie2 minimap2 blast seqtk seqkit fastqc nextflow \
    && micromamba clean --all --yes

# Keep h5py and its runtime ABI aligned, and repair the two large Bioconductor
# annotation packages whose conda post-link downloads can be flaky. The final
# load check turns any incomplete annotation-data install into a build failure.
RUN micromamba install -y -n base -c conda-forge 'hdf5<2.2' \
    && micromamba clean --all --yes
RUN set -eux; \
    bio_tmp="$(mktemp -d)"; \
    curl -fL --retry 12 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 900 \
        -o "$bio_tmp/HDO.db.tar.gz" \
        https://bioconductor.org/packages/3.18/data/annotation/src/contrib/HDO.db_0.99.1.tar.gz; \
    curl -fL --retry 12 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 900 \
        -o "$bio_tmp/org.Hs.eg.db.tar.gz" \
        https://bioconductor.org/packages/3.18/data/annotation/src/contrib/org.Hs.eg.db_3.18.0.tar.gz; \
    /opt/conda/bin/R CMD INSTALL --library=/opt/conda/lib/R/library "$bio_tmp/HDO.db.tar.gz"; \
    /opt/conda/bin/R CMD INSTALL --library=/opt/conda/lib/R/library "$bio_tmp/org.Hs.eg.db.tar.gz"; \
    rm -rf "$bio_tmp"; \
    /opt/conda/bin/Rscript -e 'pkgs <- c("DESeq2", "edgeR", "limma", "GenomicRanges", "clusterProfiler", "enrichplot", "GO.db", "tximport", "Biostrings", "org.Hs.eg.db", "org.Mm.eg.db", "lionessR", "HDO.db"); stopifnot(all(vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)))'
RUN set -eux; \
    /opt/conda/bin/python -W error -c 'import h5py'; \
    /opt/conda/bin/python -c 'import Bio, matplotlib, networkx, numpy, pandas, plotly, pysam, scipy, seaborn, sklearn, statsmodels, vcf'; \
    PATH="/opt/conda/bin:$PATH"; export PATH; \
    for tool in samtools bcftools bedtools bwa bowtie2 minimap2 blastn seqtk seqkit fastqc nextflow R; do command -v "$tool" >/dev/null; done

# These packages recur in agent-written analyses. Installing them as an additive
# layer preserves the validated v11 base and avoids repeated source builds in
# every short-lived sandbox.
USER $MAMBA_USER
RUN micromamba install -y --freeze-installed -n base -c conda-forge -c bioconda \
        'hdf5<2.2' pybigwig sourmash pydeseq2 scanpy python-igraph leidenalg plink \
    && micromamba clean --all --yes
RUN set -eux; \
    /opt/conda/bin/python -W error -c 'import h5py'; \
    /opt/conda/bin/python -c 'import igraph, leidenalg, pyBigWig, pydeseq2, scanpy, sourmash'; \
    PATH="/opt/conda/bin:$PATH"; export PATH; \
    command -v plink >/dev/null

USER root
RUN mkdir -p /data /workspace && chmod 1777 /workspace
WORKDIR /workspace

ENV MAMBA_DOCKERFILE_ACTIVATE=1
CMD ["bash"]
