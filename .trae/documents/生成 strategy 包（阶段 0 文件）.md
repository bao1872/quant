## 要创建的结构
- 在项目根目录 `d:\quant` 下创建目录 `strategy/`
- 创建文件：
  - `strategy/base.py`（按明细中的完整内容）
  - `strategy/registry.py`（按明细中的完整内容）
  - `strategy/__init__.py`（按明细中的完整内容）

## 具体实现点
- 逐字同步明细中的类与数据结构：`Bar`、`Tick`、`PriceLevel`、`MicrostructureFeatures`、`TradeSignal`、`StrategyContext`、`PriceLevelProvider`、`MicrostructureProvider`、`BaseStrategy`
- 包含示例实现：`DummyPriceLevelProvider`、`DummyMicrostructureProvider`、`DummyContext`、`DummyStrategy`
- 在 `registry.py` 中实现：`StrategyBundle`、注册表字典、`register_strategy`、`get_strategy_bundle`、`create_strategy_instances`，并默认注册 `dummy`
- 在 `__init__.py` 中导出常用抽象与注册函数，含自测入口
- 保留每个文件底部的 `__main__` 自测片段（与明细一致）

## 验证步骤
- 在 `d:\quant` 打开终端，依次运行：
  - `python -m strategy.base`
  - `python -m strategy.registry`
  - `python -m strategy`
- 期望日志：
  - 能看到 `dummy` 策略处理一根 `Bar` 并产生 `BUY` 信号的打印
  - 能看到注册的策略列表包含 `dummy`
  - `on_tick` 自测仅打印，不报错

## 兼容与约束
- 无第三方依赖；仅用到标准库 `abc`、`dataclasses`、`datetime`、`typing`
- 不引入新文件/目录，严格按明细创建 `strategy` 包及 3 个文件
- 不触碰数据库或外部服务；仅自测逻辑

## 下一步（确认后执行）
- 创建目录与文件并写入明细中的代码
- 运行上述三个自测命令确保无错误日志