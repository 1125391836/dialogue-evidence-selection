# 分阶段推进与运行手册

首次运行先看 [运行与验证入门](quickstart_zh.md)。`project.py verify` 是零调用的一键验收；`project.py next` 是第二阶段的离线采样计划入口。当前第二阶段计划 1346 次调用，未发送；执行和重复聚合训练已经通过模拟集成验证，后续真实采集需单独确定预算。

## S0 数据和输入协议

目的：先确定模型能看到什么、答案在哪里、预算怎么算，防止方法收益其实来自信息泄漏。

本机数据已经齐备，无需重下。迁移到新环境时可重建：

```powershell
python scripts/fetch_mem2act.py --variant full --ref c935c65be9aa19c930b60657c0c4b2b7ea13a34a --out data/raw/mem2act-full
python -m context_bundles prepare --qa data/raw/mem2act-full/qa_dataset.jsonl --conversations data/raw/mem2act-full/Mem2ACT_conversation.jsonl --out data/processed/mem2act-v1 --seed 42 --quarantine-empty-sources --quarantine-missing-sources
python scripts/fetch_deepseek_tokenizer.py
python scripts/project.py audit
```

以上 `python` 指项目虚拟环境解释器。续传下载加 `--resume`。不覆盖已冻结实验的数据；改变预处理应输出新的数据目录。

当前 533 条可用任务中，训练/开发/测试分别 376/58/99。仅本轮使用的 5 个训练组产生了 25 条效用观测，并非 376 道题已全部训练。上下文来自提供来源会话集合，不检索整库。历史预算 2048，完整历史参考不受此限制。

## S1 模型接口和基线

`campaign --stage health` 做一次连通性检查；`baseline` 固定随机种子预抽取互不共享历史的开发组，运行 recent、BM25、规则配对、full。提示过长的排除在 cohort 文件中记录，不能按答对与否挑题。

检查响应类型、结束原因、用量和拒答，再看效果。完整历史本轮也很差，因此不能把所有错误都归因于压缩。

## S2 从真实执行采集学习信号

`campaign --stage diagnostics` 在固定背景 S 上运行 S、S+a、S+b、S+a+b、S+随机组。词面候选不读 gold；标准答案仅在结果出来后评分。随机组只是近似长度匹配，长度差保存在日志。

当前 5 个训练组用于学习，另 2 个开发组用于诊断。每条件只有 1 次，不能把单次差值称为稳定因果效应。下一轮需人工核验参数依赖、长度/位置/替代证据，并重复运行。

## S3 学习效用

`campaign --stage train` 只接收 train 观测；14 个特征来自可见问题、schema 和所选原文。用 StandardScaler + Ridge，记录按历史组交叉验证与常数预测基线。模型系数、数据哈希和训练任务 ID 可审计。

当前训练范围没有空集和足够小的集合，模型出现外推错误。下一轮必须扩充采样支持范围，而不是只禁掉空集再宣布问题解决。结构特征也不保证理解实体指代或参数绑定。

## S4 选择与回退

`campaign --stage selection` 使用冻结预测器，以同一候选集合比较逐项贪心和组合束搜索。新计划中，效用查询超训练坐标范围则不给可用预测；最终方案还须通过训练门槛、CV 收益、非空和预算检查，否则回退 BM25，并记录 `guard` 和 `effective_selected_ids`。

30 个训练组只是保守工程默认下限，当前 gate 允许范围内仍需正式验证。范围检查无法检测所有分布偏移。回退结果必须标明，不能算作纯学习方法的效果。

本轮 6 次旧结果保留原始无防护行为；恢复旧计划不会改写历史结果。`project.py review` 离线展示新防护会如何处理，生成 `guarded_proposals_not_executed.jsonl`；其中 `executed=false`、`performance=null`，不编造修复后的准确率。

## S5 公平对照与报告

`summary` 先生成摘要，再用同一执行提示回答；生成和执行两部分成本都记录。摘要输出截断/无效会留审计，不隐藏失败。当前只做两题，与其他方法比较时取共同题目。

`report` 产生 JSON、CSV、Markdown 和 HTML；按共享任务和历史组做配对 bootstrap。极少组数下区间不具备论文主结论的可靠性。

`review` 不联网、不修改严格评分；事后默认值敏感性只处理文档明确的可选字面默认值。它用于发现评分与工具协议的差异，正式实验前应预注册规范并用实际工具验证。

## S6 下一步真正的研究扩展

1. 人工审核预先选定的 train/dev 任务，记录历史依据、实体绑定、显式默认值及错误类别。
2. 先采集能覆盖选择器搜索范围的训练集合；重新设计候选的参数依赖表达，对比词面、实体链、随机候选。
3. 多预算、多次重复，增加强检索、LLM 一次性集合选择、条件边际选择和适用文献方法；区分复现与改编。
4. 对同一预测器比较逐项、配对种子、组合束搜索，报告完整成本，拆开候选漏召回、预测误差和搜索误差。
5. 有证据后接入 AppWorld：相同环境快照继续执行、同恢复权限、最终状态自动评分。此模块尚未实现。

“带训练支持约束的预算选择”可以作为工程稳健性设计；目前不能直接把它包装成新创新点。论文仍需要最近邻强基线、稳定现象、机制解释与跨环境收益，不能通过项目复杂度代替这些证据。

## 续跑、费用和故障

本轮 ledger 上限 60、使用 58；不要换目录绕过本轮上限。新活动是新的预算决策。当前完成状态可用 `project.py campaign --stage status` 查看。

调用前预留额度，异常中断后 `running` 表示服务端是否完成不确定，程序不会自动重发。完成任务按内容哈希去重，错误也保留，不选择性重试。若要补测，先核对日志并显式设计新请求及预算。

旧通用 `run-api` 是早期接口入口，不与 campaign 共用 SQLite 总额度；本项目后续真实 DeepSeek 实验统一走 campaign，避免分散计数。不要把“60 次调用”理解为费用封顶，长上下文与摘要输出都会影响 token 用量。
