"""Produce a readable report from saved native metrics; never call a model."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    model = 'hermes_mixed' if meta.get('mixed_models') else meta['model']
    raw_path = root/'selected_results.json'
    if not raw_path.exists():
        raw_path = root/'results'/model/'raw_results.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    source_counts = {name: sum(r['model_name'] == name for r in raw) for name in sorted({r['model_name'] for r in raw})}
    evaluation = json.loads((root/'evaluation/evaluation_results.json').read_text(encoding='utf-8'))
    dataset = pd.read_csv(root/'dataset.csv')
    base = dataset[['case_id', 'query', 'task_scenario', 'intent_family']].copy()
    records = []
    for row in raw:
        records.append({'case_id': row['case_id'], 'model_name': row['model_name'], 'execution_ok': not bool(row.get('error')),
            'execution_seconds': round(row.get('execution_time', 0), 2),
            'tool_calls': sum(len(w.get('tool_calls', [])) for w in row.get('worker_output', [])),
            'exported_tokens': sum(u.get('total_tokens', 0) for u in row.get('token_usage', {}).values() if isinstance(u, dict)),
            'error': row.get('error', ''), 'answer': row.get('reporter_output', '')})
    base = base.merge(pd.DataFrame(records), on='case_id', how='left', validate='one_to_one')
    metric_keys = {'tool': ['TS_P', 'TS_R', 'SC'], 'planning': ['DEC_P', 'DEC_R'],
        'instruction': ['ID', 'IE'], 'answer': ['DR', 'FPR'], 'efficiency': ['IT', 'OT']}
    for metric, keys in metric_keys.items():
        detail = pd.read_csv(root/'evaluation'/model/f'{metric}_evaluation.csv')
        cols = [key for key in keys if key in detail]
        base = base.merge(detail[['case_id']+cols], on='case_id', how='left', validate='one_to_one')
    base.to_csv(root/'case_summary.csv', index=False, encoding='utf-8-sig')
    failures = sum(bool(row.get('error')) for row in raw)
    lines = ['# Hermes sample10 基线结果', '',
        f'结果分组：`{model}`；并发：{meta["workers"]}；数据：sample_10.csv，共 {len(dataset)} 条。', '',
        '实际模型分布：'+ '；'.join(f'`{name}` {count} 条' for name,count in source_counts.items())+'。', '',
        f'已运行 {len(raw)} 条，程序正常结束 {len(raw)-failures} 条，执行异常 {failures} 条。程序完成不等于答案通过。', '',
        '本轮保持原来的 Hermes 提示词、Agent 循环、工具与结果映射，使用 MobilityBench 原生评估器。共享沙箱只改变进程和 MCP 传输，原有工具与评分代码没有修改。', '',
        '**相似度使用原生字符规则回退**，未加载 Embedding 权重。后续对照实验应使用相同评分后端。', '',
        '## 原生总分项', '', '| 类别 | 分项 | 结果 | 参与评分题数 | 评分异常题数 |', '| --- | --- | ---: | ---: | ---: |']
    for metric, keys in metric_keys.items():
        summary = evaluation[metric][model]['summary']
        for key in keys:
            if key in summary['sub_scores']:
                value = summary['sub_scores'][key]
                shown = f'{value:.2f}' if metric == 'efficiency' else f'{value*100:.2f}%'
                lines.append(f'| {metric} | {key} | {shown} | {summary["total_cases"]} | {summary["failed_cases"]} |')
    lines += ['', 'TS_P/TS_R/SC 为工具指标，DEC_P/DEC_R 为规划指标，ID/IE 为理解指标，DR/FPR 为答案交付/通过指标，IT/OT 为平均输入/输出 Token。原实现没有统一的加权总成绩，未另造总分。', '',
        '## 分场景结果', '', '| 场景 | 样本数 | 程序完成 | 答案评分覆盖 | DR | FPR |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scenario, group in base.groupby('task_scenario', sort=False):
        dr, fpr = group['DR'].mean(), group['FPR'].mean()
        lines.append(f'| {scenario} | {len(group)} | {int(group["execution_ok"].fillna(False).sum())} | {group["FPR"].count()} | {dr*100:.2f}% | {fpr*100:.2f}% |')
    diagnosis_path = root/'diagnosis.json'
    if diagnosis_path.exists():
        diagnosis = json.loads(diagnosis_path.read_text(encoding='utf-8'))
        budget_cases = [r for r in diagnosis['execution_failures'] if r['tool_rounds'] == meta['harness']['max_steps']]
        lines += ['', '## 本轮低分与异常的核查', '',
            f'- 路线输出格式：{diagnosis["route_outputs"]} 条正常导出的路线规划答案中，{diagnosis["missing_recognizable_route_index"]} 条没有原解析器可识别的 route_index。仅 case_37、case_56 识别到索引；前者约束方案索引不匹配，后者通过。没有据此断言所有路线内容错误。',
            '- 天气字符回退：10 条天气答案仅 1 条通过。case_71 的天气、气温、湿度、风力和更新时间与标签一致，但改了标点并增加保暖提醒后，原字符相似度为 0.7708，未超过 0.8，因而被判 0。这说明本轮天气低分包含表达格式带来的影响。',
            f'- 轮次上限：{len(budget_cases)} 条执行异常都保存了 12 轮工具调用以及最后的总结。Hermes 在达到 max_iterations 后把 completed 标为 false，当前适配器据此抛错，未向原生评分器导出这些总结和调用轨迹。它们不等于“完全没有回答”，但本轮未补改映射。',
            '- 后续优先考虑：明确最终答案的原生字段协议、让适配器保留预算耗尽时的真实输出；天气答复保留工具原文可以减少字符回退的格式损失。若要评价真实规划，再单独落实执行前计划的输出与读取。所有改造应另存对照结果，评分公式保持不变。',
            '- 以上诊断未重新调用模型、未改写答案、未改变本轮原生成绩；逐条证据见 diagnosis.json。']
    lines += ['', '## 解释范围', '',
        '- 本轮根据用户要求在原模型免费额度耗尽后换模型接着跑。组合成绩用于查看这批任务的整体效果，不是任一单模型完成 115 条的成绩；各模型覆盖的题目也不同，不据此直接排名。',
        '- 仅额度/供应商错误重试；原先正常结束及 Agent 执行未完成的样本均保留。旧的额度失败尝试仍在 results/ 和 traces/，selected_results.json 每题只选一个用于本轮评分的结果。',
        '- 所有分数按原生公式与原生样本分母展示，没有补零、修参数值评分或增加严格检查。空输出或无工具样本可能缺席部分原生指标，因此同时报告覆盖数。',
        '- 当前计划/意图仍由适配器根据工具轨迹生成，不能把这些分数单独解释成 Hermes 事前计划的质量。',
        '- 最终答案仍使用原来的自然语言提示词，没有为了 route_index 改写答案；答案格式导致的丢分也包含在本轮基线中。',
        '- 异常退出时当前适配器可能没有导出已消耗 Token；统计值不是完整供应商账单。', '',
        '## 文件', '',
        '- `case_summary.csv`：115 条题目、执行状态、原答案和原生各项分数，未参与评分的字段留空。',
        '- `evaluation/evaluation_summary.csv`：原生总体及意图族汇总。',
        '- `evaluation/<model>/*_evaluation.csv`：原生逐题详细评分。',
        '- `results/<model>/raw_results.json`：各模型直接导出的结果，包括被续跑替换的原始额度失败记录。',
        '- `selected_results.json` / `evaluation_input/`：逐题选用结果、分模型和组合的原生评分输入。',
        '- `events.jsonl` / `run.log` / `traces/`：运行进度、失败原因与真实 Hermes 会话。',
        '- `transport-parity.json`：共享 MCP 与此前 stdio 返回的比对。', '']
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(root/'report.md')
    print(base.groupby('task_scenario')[['DR', 'FPR']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
