# MedEvidence pilot-3 工程检查

CPU身份/GT/排程/封存检查、GPU无更新梯度与恢复、初始状态比较见相邻JSON。

GPU预检状态：completed；两分支实际初始状态相等：True。浮点容差预定2个bf16 epsilon=.015625；源码和adapter/RNG身份使用精确比较。无optimizer更新式smoke。
