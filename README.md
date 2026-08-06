# TRAJDEBUG

> [中文说明](README_zh.md)

**TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories** is an evidence-grounded framework for locating the earliest decisive error responsible for a failed LLM-agent trajectory. This repository provides the detector, unified data adapters, evaluation tools, and a local viewer.

## Method

TRAJDEBUG first builds multi-granularity trajectory views, then performs three
auditable stages:

1. **Error trigger detection:** identifies wrong commitments and requires verbatim evidence for both the commitment and violated reference.
2. **Error state classification:** clusters triggers by violated object and classifies resolution and terminal impact.
3. **Causal attribution:** selects the failure-responsible origin from terminal-relevant candidates.

![Overview of the TRAJDEBUG pipeline](assets/%20pipeline.png)

See [`detector/README.md`](detector/README.md) for implementation details and the [`paper`](main.pdf).

## Results

![Critical error detection results](assets/results.png)

The source code and associated data are undergoing internal review and approval. They will be made publicly available once the approval process is complete. Thank you for your patience and interest.
