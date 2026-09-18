# Deployment Architecture

## Architecture Evaluation Table

The following table evaluates three candidate AWS inference architectures against
the criteria relevant to the retail computer vision use case (offline batch analysis
of recorded CCTV footage).

| Criterion | SageMaker Batch Transform | SageMaker Real-Time Endpoint | Lambda + API Gateway |
|---|---|---|---|
| Cost per inference | **Suitable** — pay only while the job runs; no idle cost between jobs | **Conditional** — always-on instance accrues cost even when idle; acceptable only if the endpoint is deleted after 30 min of inactivity | **Suitable** — pay per invocation with no standing infrastructure cost |
| Cold-start latency tolerance | **Suitable** — batch jobs tolerate minutes-to-hours of startup time by design | **Conditional** — requires warm endpoint; cold-start must not exceed 30 s for dashboard use cases | **Unsuitable** — cold starts of 10–30 s make sub-second or near-real-time response unreliable |
| Input payload size | **Suitable** — input is an S3 prefix; no per-request payload limit | **Conditional** — 20 MB per request cap is sufficient for single frames but rules out multi-frame or video payloads | **Unsuitable** — 6 MB hard limit on Lambda payload prevents full-resolution frame delivery |
| GPU availability | **Suitable** — `ml.g4dn.xlarge` (NVIDIA T4) available for batch transform jobs | **Suitable** — `ml.g4dn.xlarge` available for real-time endpoints | **Unsuitable** — Lambda has no GPU execution environment |
| Operational complexity | **Suitable** — single `CreateTransformJob` API call plus status polling; no persistent resource lifecycle to manage | **Conditional** — requires endpoint creation, health monitoring, auto-scaling policy, and explicit teardown to control cost | **Unsuitable** — requires a custom container, API Gateway integration, and manual cold-start management |

### Verdict Summary

| Architecture | Overall Verdict |
|---|---|
| SageMaker Batch Transform | **Selected (primary)** |
| SageMaker Real-Time Endpoint | **Conditional (secondary — optional)** |
| Lambda + API Gateway | **Unsuitable** |

---

## Selected Architecture: SageMaker Batch Transform

**Primary architecture: SageMaker Batch Transform**

### Justification

**(a) Read-after-completion consumption pattern**
Inference results for this pipeline are consumed only after the full job completes,
not per-request. Store analysts and downstream data pipelines retrieve detection
output as a complete JSON result set for an entire footage batch. SageMaker Batch
Transform matches this read-after-completion pattern exactly: it processes all input
objects in an S3 prefix, writes one output JSON per input image to a destination
S3 prefix, and signals completion atomically. There is no requirement for
per-frame streaming or interactive querying that would justify a live endpoint.

**(b) No always-on endpoint cost between jobs**
Batch transform incurs no always-on endpoint costs between jobs. Compute resources
are provisioned only for the duration of the transform job and released immediately
upon completion. For scheduled offline workloads — such as nightly or weekly
processing of recorded store footage — this makes batch transform significantly
lower cost than maintaining a real-time endpoint that would sit idle for hours or
days between invocations.

---

## Instance Type Selection: ml.g4dn.xlarge

The minimum recommended instance type for GPU-accelerated YOLO nano inference is
`ml.g4dn.xlarge`, which provides an NVIDIA T4 GPU with 16 GB VRAM.

### GPU vs CPU Inference Time Comparison

| Instance Type | GPU | Approx. Inference Time per Frame | Source |
|---|---|---|---|
| `ml.g4dn.xlarge` | NVIDIA T4 (16 GB VRAM) | ~3–8 ms | [Ultralytics benchmark docs](https://docs.ultralytics.com/modes/benchmark/); community T4 measurements |
| `ml.m5.xlarge` | None (CPU only) | ~150–350 ms | AWS instance specs + Ultralytics CPU benchmarks |

For a representative batch of 1,000 frames:

- **GPU (T4, ml.g4dn.xlarge):** ~5–8 seconds total inference time
- **CPU (ml.m5.xlarge):** ~150–350 seconds total inference time

This represents a **30–50× speedup** on GPU over a CPU-only instance for a YOLO
nano model at 416 px input resolution.

### Why ml.g4dn.xlarge and not a larger instance

`ml.g4dn.xlarge` is the entry-level GPU instance in the SageMaker catalog. The next
tier (`ml.g4dn.2xlarge`) doubles the hourly cost without proportionally improving
throughput for a single-model nano workload that fits comfortably within T4 VRAM.
Scaling horizontally (multiple parallel transform instances) is a better cost lever
if throughput needs to increase.

---

## Optional Real-Time Endpoint Configuration

WHERE a stakeholder requires on-demand single-frame inference (e.g., a store
dashboard querying one frame at a time), a SageMaker Real-Time Endpoint may be
deployed as a secondary optional configuration.

| Parameter | Value |
|---|---|
| Instance type | `ml.g4dn.xlarge` |
| Minimum instance count | 1 |
| Maximum instance count | 4 (auto-scaling) |
| Request payload format | Single JPEG or PNG frame; max 20 MB; max 1920×1080 pixels |
| Cold-start latency tolerance | **≤ 30 seconds** |
| Auto-scale out trigger | `InvocationsPerInstance` > 10 requests/min |
| Auto-scale in trigger | `InvocationsPerInstance` < 5 requests/min for 5 consecutive minutes |
| Idle shutdown | Endpoint **must** be deleted after 30 minutes of zero invocations to avoid unnecessary cost |

> **Note:** The real-time endpoint is an optional secondary configuration only.
> It should not be deployed alongside batch transform for the same workload.
> Deploy it only when interactive, per-frame query latency is a hard requirement.
