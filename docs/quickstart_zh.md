# 如何运行和验证项目

更新：2026-11-18。你目前最应该运行的是 **离线验收**。它验证程序、数据和恢复流程是否正常，不产生 DeepSeek 调用，也不代表研究方法已经有效。

## 直接在编辑器运行

项目根目录已提供 `run.py`。无参数运行它就会执行完整离线验收，不需要手动输入命令。入口自动定位项目目录；如果编辑器选中了系统 Python，会显示 `Switching to project interpreter` 并通过项目 `.venv` 重新启动，保留原参数和退出码。项目环境缺失时会给出明确提示，不自动修改系统 Python。

2026-11-20 修复：系统解释器缺少 `tokenizers`，其子进程也找不到 `context_bundles`，曾导致五项检查连锁失败。现在普通 Run 可自动切换；使用断点 Debug 时仍应手动选择 `.venv\Scripts\python.exe`，以便调试器直接附着到项目进程。

**VS Code：**打开 `context-bundles` 文件夹本身（不要只打开父目录）。安装推荐的 Microsoft Python 和 Python Debugger 扩展。打开 `run.py`，选择项目 `.venv\Scripts\python.exe` 作为解释器，点击右上角“运行 Python 文件”。也可以在“运行和调试”中选择“项目：完整离线验收（无 API 调用）”，按 F5 调试或 Ctrl+F5 运行。已提供自动测试、第二阶段离线计划和余额查询三个其他配置。

如果 VS Code 已经记住别的解释器，配置默认路径不会强制替换它；执行 “Python: Select Interpreter” 手动选本项目 `.venv\Scripts\python.exe`。F5 的项目配置则明确指定了该解释器。

**PyCharm：**打开 `context-bundles` 项目，在设置中把 Python Interpreter 指向已有 `.venv\Scripts\python.exe`，右键 `run.py` → Run。无需添加启动参数；要只测单元测试，可以在运行配置的 Parameters 中填写 `test`。

正常结束显示 `PASS: 5/5 checks; 0 model calls`。上述项目配置均不发送 DeepSeek 请求。下面的 PowerShell 操作仍可作为命令行备选。

## 1. 打开终端，进入项目

在 Windows 打开 PowerShell，复制执行：

```powershell
Set-Location -LiteralPath 'C:\Users\张恒\Desktop\开题报告\第二个创新点\context-bundles'
.\.venv\Scripts\python.exe scripts/project.py verify
```

不需要先安装 Python，不需要激活虚拟环境，也不需要再次填写密钥。本机已经配置好项目环境。首次读取 tokenizer 和检查计划会花一点时间，看到 PASS 行后继续等待最终汇总即可。

正常会看到：

```text
[PASS] Environment and imports
[PASS] Raw data and history split audit
[PASS] Unit and integration tests
[PASS] Frozen campaign replay without model access
[PASS] Next-stage frozen plan
PASS: 5/5 checks; 0 model calls; runs\verification\latest.json
```

当前测试总数为 31。日志在 `runs/verification/latest.json`，便于阅读的汇总在同目录 `latest.md`。如果输出 FAIL，打开 JSON 看对应检查项的 details；不要为了通过验收删除失败结果、改原数据或改评分。

## 2. 每项 PASS 到底验证什么

| 检查 | 说明 | 通过不能证明什么 |
|---|---|---|
| 环境 | 项目解释器能导入 NumPy、sklearn、tokenizer 等依赖 | 不证明模型会答题 |
| 数据 | 原始文件校验通过；533 任务的来源和划分符合规则 | 不证明每条标准答案语义都无误 |
| 测试 | 单元测试与模拟接口集成测试通过 | 模拟测试的成绩不是 DeepSeek 成绩 |
| 冻结实验恢复 | 在临时副本重放原阶段；禁止新增模型请求；原文件哈希不变 | 不重新测试在线服务可用性 |
| 新计划 | 检查请求哈希、预算、重复条件、历史来源、train/dev 隔离、固定背景 | 不证明词面候选真有互补关系 |

最终测试集 99 题未用于模型效果评估。验收允许检查其划分结构，不把答案发送给模型。

## 3. 查看首轮真实运行

```powershell
.\.venv\Scripts\python.exe scripts/project.py campaign --stage status
Start-Process -FilePath '.\runs\deepseek-progressive-60\report\index.html'
```

状态应为：`reserved_calls: 58`、`remaining_calls: 2`、`jobs: {ok: 58}`。这里 ok 指接口成功返回，不是 58 道题答对。

原始实验报告在 `runs/deepseek-progressive-60/report/`。失败分析在 `runs/deepseek-progressive-60/postmortem/review.md`。严格完整调用正确率在这轮极少量开发题上为 0，不能把工程 PASS 当成论文实验成功。

只重做离线诊断：

```powershell
.\.venv\Scripts\python.exe scripts/project.py review
```

此命令不调用模型，只更新独立 postmortem 分析，不覆盖原始模型和严格分数。

## 4. 运行第二阶段的离线计划

```powershell
.\.venv\Scripts\python.exe scripts/project.py next
.\.venv\Scripts\python.exe scripts/project.py next --action status
Start-Process -FilePath '.\runs\curriculum-v2-plan\audit.html'
```

`next` 默认是 plan，不是付费执行。第一次生成计划，再次运行校验冻结计划，不随机换一批题。当前结果：

- 32 个独立训练组 + 8 个独立开发组，最终 test 不参与。
- 673 种不同历史集合，包括 40 个空历史锚点、单条信息、512/1024/2048 token 采样、组合干预和完整历史参考。
- 每个集合重复 2 次，所以计划 1346 次模型调用；当前实际新增 0 次。
- 相同题目、相同历史和相同提示只采集一组观测，多种采样角色共享它，不重复花费，也不重复充当独立样本。
- 40 个词面候选对中，26 个找到历史长度差不超过 64 tokens 的随机对照；其余明确记为缺失。长度接近不等于对照内容无关，仍需审核。

网页中可展开任务、工具、原始历史和候选组合。它不展示隐藏答案，也没有把词面重合自动标为真实依赖。`annotations-template.jsonl` 是未完成的人工审核模板；复制后填写，不能把模板当成已审核证据。

这个规模是下一轮开发候选，**并非已经确定必须支付 1346 次调用**，也不是论文统计充分性的保证。可以先通过离线审核修订配置并生成新计划，不能看到成绩后挑选题目。

## 5. 哪些命令会调用模型

| 命令 | 是否有真实 API 调用 |
|---|---|
| `verify`、`test`、`audit`、`review` | 无 |
| `next`、`next --action validate`、`next --action status` | 无 |
| `next --action train` | 无；但必须先完成对应的真实观测 |
| `campaign --stage ...` | 未完成的请求可能调用；默认旧活动已经完成 |
| `next --action run --max-new-calls N` | 可能；还必须有足够的既有账本额度 |

第二阶段执行器不会创建新账本或提高额度，不会先跑两次再因额度不足留下偏置子集。当前 1346 次计划超过剩余 2 次，所以执行会在发送之前拒绝；真实调用总数继续保持 58。后续要实际采集新一轮数据，应先确定独立调用预算和冻结协议。

代码已经实现真实执行后的训练路径：同一个历史集合的重复结果先求平均，只有完整的重复条件才进入拟合；API 错误和截断记为排除，全部重复均失败的集合也会显式列出。仅 train 的观测拟合 Ridge，dev 只用于样本外诊断。纯完整历史参考不进入预算预测器拟合；若同一集合同时满足某个预算采样角色，可以按那个预算角色使用。

目前第二阶段只有真实数据上的计划、模拟服务集成验证，尚未采集 DeepSeek 新结果，也没有第二阶段训练模型或准确率。

## 6. 常见问题

**提示找不到路径：**先执行上面的 Set-Location；不要在父目录直接使用 `.venv` 相对路径。

**提示 ModuleNotFoundError：**确认用的是 `.\.venv\Scripts\python.exe`。只有迁移或重建环境时才需要参考 README 安装依赖。

**计划冻结检查失败：**检查是否改了数据、配置或 requests 文件。新方案用新的输出目录保存，不覆盖旧计划。

**完整验收通过，但效果很差：**这是当前真实状态。工程验收回答“实验是否可靠运行”，科学实验回答“方法是否改善任务完成”。下一步是依据审核和采样修正，然后才是新的付费数据采集。
