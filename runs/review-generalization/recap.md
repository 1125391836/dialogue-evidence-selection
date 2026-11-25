# 原生采集场景聚类和分组误差

冻结原实验阈值 0.5 和预测。所有差值均为字段选择减最近消息，单位为百分点。按每条对话在原始 JSON 的完整 scenario 字符串聚类。

|批次|对话 / 场景|指标|差值|场景重采样 95% 区间|分别删除一个场景后的差值范围|场景符号检验 Holm p|
|---|---:|---|---:|---:|---:|---:|
|test|470 / 37|correct|+2.77|[-0.79, +10.03]|[+0.00, +3.28]|1|
|test|470 / 37|wrong|+2.98|[+1.42, +4.85]|[+2.64, +3.54]|0.008057|
|replication|967 / 38|correct|+2.28|[-1.43, +9.57]|[-0.64, +2.65]|1|
|replication|967 / 38|wrong|+2.79|[+1.45, +4.50]|[+2.55, +3.49]|0.0001831|

20,000 次场景整组有放回抽样，每次以抽到的全部对话为分母。区间是单项百分位区间。另在每个场景内整体交换方法标签，精确计算双侧 p 值，以两个批次的两项结局组成四项 Holm 检验族。该检验依赖场景间独立性和零假设下方法标签在场景内整体可交换；它是事后敏感性分析。少数场景贡献差异时有效信息有限。逐一删除场景考察已训练模型的评测构成敏感性；模型训练时未按场景隔离。

## 对话长度

输入长度计入当前用户消息及之前所有公开的用户、助手消息。分箱用于描述。表中为各箱内的正确 / 错单数，以该箱全部对话为分母。

|批次|公开消息数|对话|最近消息正确 / 错单|字段选择正确 / 错单|
|---|---|---:|---:|---:|
|test|1-12|8|0 / 0|0 / 0|
|test|13-20|377|51 / 6|66 / 15|
|test|21+|85|42 / 1|40 / 6|
|replication|1-12|18|0 / 1|0 / 1|
|replication|13-20|797|105 / 10|131 / 32|
|replication|21+|152|81 / 0|77 / 5|

## 已提交订单的字段错误

漏填表示参考有值但预测缺失，多填表示参考缺失但预测有值，错值表示双方都有值且不相等。同一订单可以有多处字段错误。只列出现错误的字段。分母为该方法在该批次的提交数。

|批次|方法|提交数|字段|漏填|多填|错值|
|---|---|---:|---|---:|---:|---:|
|test|recent|100|date.showing|4|0|1|
|test|recent|100|name.theater|0|0|4|
|test|recent|100|num.tickets|1|0|1|
|test|recent|100|seating|5|0|0|
|test|field_support|127|date.showing|2|0|2|
|test|field_support|127|name.movie|0|0|5|
|test|field_support|127|name.theater|0|0|6|
|test|field_support|127|num.tickets|1|0|3|
|test|field_support|127|seating|11|5|1|
|test|field_support|127|time.showing|0|0|1|
|replication|recent|197|date.showing|0|0|7|
|replication|recent|197|name.theater|0|0|1|
|replication|recent|197|num.tickets|1|0|1|
|replication|recent|197|seating|8|0|0|
|replication|recent|197|time.showing|0|0|1|
|replication|field_support|246|date.showing|2|0|12|
|replication|field_support|246|name.movie|0|0|4|
|replication|field_support|246|name.theater|0|0|9|
|replication|field_support|246|num.tickets|5|0|4|
|replication|field_support|246|seating|19|12|1|
|replication|field_support|246|time.showing|0|0|5|

## 正确率差受哪些场景影响

test 中，去除 Auto template 1 with theater name error 后正确率差降幅最大。该场景有 14 段，字段选择多匹配 13 单；其余 456 段的正确率差为 +0.00 个百分点。

replication 中，去除 Auto template 1 with theater name error 后正确率差降幅最大。该场景有 29 段，字段选择多匹配 28 单；其余 938 段的正确率差为 -0.64 个百分点。

逐场景结局、删组结果、普通 McNemar 对照、文件哈希见 [results.json](results.json)。逐对话分组见 [case-groups.jsonl](case-groups.jsonl)。按场景隔离并重新训练模型的实验见 [五折重训](scenario-refit.md)，模式规则跨服务结果见 [SGD 新对话](sgd-recap.md)。
