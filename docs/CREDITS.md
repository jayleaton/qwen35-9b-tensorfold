# Credits

- **[Qwen team](https://huggingface.co/Qwen): [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)**, the model, its
  hybrid Gated DeltaNet decoder, vision tower and MTP head, at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
  Apache License 2.0.
- **[Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold)** and the TensorFold contributors: the
  engine and server. The `qwen3_5` CUDA family and the dense `qwen27` Metal family that runs Qwen3.5-9B, the Gated
  DeltaNet and hybrid-attention kernels, the MLX affine readers, the EXL3 loader, copy drafts, the prompt cache and its
  disk tier, and the kernel-contract harness are upstream work. Apache License 2.0; code written before 0.6.0 keeps
  its MIT notice.
- **[SenseNova-Loop-9B recipe](https://github.com/jayleaton/sensenova-loop9b-tensorfold)**: the patch series, build
  and serve scripts reused here unchanged.
- **Shared-round drafter pricing: upstream TensorFold PR [#2475](https://github.com/ashhart/TensorFold/pull/2475) by
  [jrb-ai](https://github.com/jrb-ai)**, which the series' copy-draft work builds on.
- **[exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp**: the EXL3 format and its quantizer, which
  the RTX packs are made with, and **[QTIP](https://github.com/Cornell-RelaxML/qtip)** (Tseng, Sun, Hou and De Sa,
  Cornell RelaxML), the trellis-coded quantization EXL3 is built on. MIT and GPL-3.0 respectively; neither is included
  here.
- [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (Apache License 2.0): the grammar engine behind
  `libtfgrammar`, which `scripts/build-engine.sh` builds for required and named tool calls.
- [EvalPlus](https://github.com/evalplus/evalplus) (HumanEval+) and the
  [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla) (BFCL v4): the quality checks.
