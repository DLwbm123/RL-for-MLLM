# RSNA v3 实际执行入口

> 后续授权更新（2026-10-06）：用户已授权 GPU 0/1、发布，并指定 GPT Astra 替代本轮人工审核。原 CPU 重放配置/协议锁保留为历史记录；当前权限与状态以 `rsna_v3_execution_amendment.json`、`rsna_v3_stage_status.json` 为准。AI 审核不改称真实专家意见；历史未授权描述仅对应初始阶段。

使用已配置的private/runtime.json和既有远端CPU环境，所有远端Python入口通过stdin传递，CUDA_VISIBLE_DEVICES为空。本轮已完成prepare/replay/review，重复执行会拒绝覆盖协议、重放目录或真实审核返回文件。

CPU状态收集（可重复，只读远端）：
```sh
V3_OPERATION=collect python3 - <<'PY'
import sys
sys.path.insert(0,'scripts')
from dispatch_rsna_v3 import main
main()
PY
```

同一入口支持prepare（身份与协议锁）、replay（已有缓存重放）、review（旧区域盲审包）、numerics（已存重放残差聚合）。前三项本轮已经执行，不需要重跑；numerics不加载模型、不采样。远端代码和数据部署位置由private/runtime.json给定，不将患者产物放入公开报告目录。

合成R1检查，在已具备torch的CPU环境执行：
```sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import pytest
raise SystemExit(pytest.main(['-q','tests/test_v3.py']))
PY
```

作者接口CPU检查（本地已有固定作者私有副本）：
```sh
python3 - <<'PY'
import unittest
suite=unittest.defaultTestLoader.discover('tests',pattern='test_v3_interfaces.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

GPU就绪检查：同样入口将V3_OPERATION设为G1或G2，返回最新阶段状态、gpu_authorized=true及launched=false。此入口只检查就绪状态，不是模型启动器；授权修订后仍须满足AI审核/来源recipe等前提，不能通过改环境变量绕过。

原始case_decomposition.json、group_advantage_replay.json、blind index/review.csv与身份映射均在独立v3私有outputs内。本轮已保存独立修订提议和最终排除manifest（0 selected、32 excluded）；旧区域只读保留，原人工pending表与Astra返回表分开。

AI审核完成后的CPU合并入口：
```sh
python3 - <<'PY'
import sys
sys.path.insert(0,'scripts')
from summarize_rsna_v3_ai_review import main
main()
PY
```
该命令要求93例930行全部有实际AI审核记录，拒绝缺失或pending；不自动生成判定、不覆盖原表。


单次语义提议的确定性生成入口（需要Pillow；本轮已运行，重复运行会拒绝覆盖）：
```sh
python3 - <<'PY'
from scripts.prepare_rsna_v3_ai_regions import main
main()
PY
```

G1实际GPU启动器在远端部署环境中通过中性stdin调用；需预先设置既有`OUTPUT_ROOT`、`V1_OUTPUT_ROOT`、`V2_OUTPUT_ROOT`、`DATA_ROOT`、`MODEL_ROOT`、`DATASET_NAME=rsna`、`PYTHONPATH`和私有冻结manifest的`V3_G1_MANIFEST_SHA256`：
```sh
python -u - <<'PY'
from scripts.launch_rsna_v3_g1 import main
main()
PY
```
该启动器核验授权、预算、空清单、显存和单次启动收据，以中性stdin创建两个固定分片；工作进程核验完整冻结身份后才加载B1。累计1 GPU小时包含加载、推理和失败，硬超时/任一进程失败会停止本次进程组；没有训练或重新采样入口。本轮selected为空，**没有执行这个GPU命令**，它会拒绝启动；不能通过修改manifest来绕过科学资格。

G1工作进程门禁和启动器CPU检查：
```sh
python3 - <<'PY'
import unittest
suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p)
    for p in ['test_v3_g1.py','test_v3_g1_launcher.py'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```
本轮G1工作进程4项和启动器1项均通过；总计26项检查通过，具体日志摘要见provenance。G2无满足来源条件的模型入口；只交付已测CPU接口合同和明确阻塞，不提供冒充author_exact的运行命令。
