1. 策略可复用 / 可替换原则

框架必须将 策略逻辑 与 基础设施 解耦：

数据、回测、实盘、前端等模块，不得直接依赖“阿布价格理论”的具体实现文件，只能依赖：

BaseStrategy

PriceLevelProvider 抽象

MicrostructureProvider 抽象

BrokerBase 抽象

阿布价格理论 + 微观结构 + 阿布策略只是 当前的一种实现：

未来新增策略时，只需要：

新增一个 Strategy 实现

新增一个 PriceLevelProvider（如果用自己的价格理论）

新增或复用 MicrostructureProvider

在 strategy/registry.py 注册

整个数据→回测→实盘→前端流程必须无需改动（最多调整配置）。

2. 阿布价格理论模块必须是“实盘级”

不允许“教学版简化算法”：

AbuPriceLevelProvider 必须严格按照文档 docs/abu_price_levels.md 中定义的规则实现：

摆动高低、多级别趋势线、缺口、黄金分割、常用形态目标价、公共价位等。

任何“先写一个简单版、以后再说”的实现必须明确标注为临时版本，并在后续任务中替换成完整实现。

关键位必须落 DB：

所有关键位结果必须写入 price_levels_daily 表，实盘和回测一律从该表读取，不得在实盘进程中做大规模重算（只允许轻量过滤和打标签）。

关键位模块必须可替换：

所有上层代码不得 import abu_price_levels.py，只能 import 抽象类 & 通过注册表获取实现。

3. 数据与存储规范

行情源：

当前阶段：只允许 pytdx 作为行情数据源；

不得使用 qmtmini 的行情接口拉历史数据。

存储：

除 tick 外的所有历史行情、因子、关键位、回测结果、实盘记录必须存入 Postgres。

tick 数据必须存本地文件（parquet 等），用 TickFileIndex 表索引。

增量更新：

所有更新函数必须使用增量逻辑（只更新新 Bar + 覆盖尾部），禁止传统“全量 truncate + bulk insert”方式。

4. 回测规则

日线/分钟级回测：

必须使用 Backtrader 作为回测引擎；

不得自行重复造一个完全相同功能的日线回测框架；

Backtrader 中的策略类必须通过“Adapter”层调用 BaseStrategy，以便切换策略实现。

Tick 级回测：

必须使用单独的 TickBacktestEngine（自建），逻辑：

读取 tick 文件 + 关键位

逐 tick 推进

在关键位窗口调用策略 & 微观结构

模拟交易并输出统计

该引擎必须是策略无关的，只依赖 BaseStrategy & 抽象 Provider。

回测与实盘的一致性：

Strategy 实现的核心逻辑应当在 Backtrader 回测、Tick 引擎回测、实盘中共用，不得存在“专门为回测优化的隐藏规则”。

5. 实盘 & Broker 规则

Broker 接口：

必须通过 BrokerBase 抽象统一定义；

QmtBroker 使用 easytrader[miniqmt] 或 xtquant.xttrader 实现；

实盘下单、撤单、查询全部走 Broker 抽象，禁止在业务代码中直接调用 easytrader/xtquant。

实盘引擎：

LiveEngine 必须仅依赖：

BaseStrategy、PriceLevelProvider、MicrostructureProvider、BrokerBase

实盘逻辑必须严格按照：

选股 & 读取关键位

监控价格触碰关键位

微观结构分析

策略生成信号

风控计算仓位

Broker 下单

所有信号与订单必须记录到 DB。

6. Streamlit 前端规则

前端技术栈：

只能使用 Streamlit + Plotly/Matplotlib，

禁止在本项目中引入新的 Web 框架（React/Vue 单独工程）。

前端只读：

前端只能通过封装好的 API/DB，阅读数据 & 展示，不直接调用 Broker 下单函数。

7. main 自测规则（重要）

范围：

除纯配置文件（如 config.py）外，所有 Python 模块文件底部必须包含：

if __name__ == "__main__":
    # 针对本文件关键功能的最小自测
    ...


内容要求：

每个 main 自测代码必须：

调用本文件的 1–2 个关键函数/类；

使用小数据/虚拟数据，仅验证逻辑和接口正确性；

打印关键结果（例如：返回的关键位列表、信号对象、仓位等）。

禁止事项：

禁止在 main 自测里写长时间循环或依赖复杂外部服务（除非显式标明是手动运行）。

禁止在 main 自测中直接调用真实 Broker 下单；如需联调，必须有 dry-run 标记或独立脚本。

一、代码结构要求

所有与界面无关的计算逻辑、业务逻辑，一律封装为独立的 Python 函数或类方法，不要把计算直接写在 st.button 回调、st.sidebar 或页面布局代码里。

每个计算逻辑函数必须有清晰的输入参数和返回值，方便单独调用：例如 def calc_position_size(price, risk, capital) -> float:。

Streamlit 页面只负责：

读取用户输入

调用这些逻辑函数

展示结果
不要在 UI 代码里混入复杂计算。

二、调试与自检要求（重点）
4. 对于每一个新增或修改过的计算函数，你都必须做以下事情：
1）给出至少 2 组测试样例（含一个正常情况 + 一个边界/异常情况）：
- 写明：输入参数
- 写明：预期输出/行为
2）把这些样例写成可运行的测试代码，可以是：
- 使用 pytest / unittest 的单元测试，或
- 在同文件底部的
python if __name__ == "__main__": # 手工调试用的示例调用
3）说明应该如何运行这些测试：
- 如果用 pytest：给出类似
pytest tests/test_logic_xxx.py::TestCalc::test_basic_case
- 如果用 __main__ 调试：说明 python logic_xxx.py 会打印什么结果。
5. 禁止只“假装”测试 UI，而不验证逻辑：

不要只说“打开 Streamlit 页面点一点就好了”，必须通过直接调用函数来验证每个计算逻辑。

在回答最后，用一个小结说明：

本次涉及到哪些计算逻辑函数（列清单）

每个函数对应的测试样例名称/函数名

当前是否还有已知未覆盖或有风险的逻辑。

三、禁止行为
7. 不要在没有说明的情况下跳过调试步骤，也不要只写“建议测试”而不给出测试代码。
8. 除非我明确要求，否则不要改动与本次任务无关的 Streamlit 页面逻辑和已有计算函数。