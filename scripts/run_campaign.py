import argparse
from pathlib import Path
from context_bundles.campaign import Campaign, STAGES

p = argparse.ArgumentParser(description="Run/resume a stage of the frozen DeepSeek research campaign")
p.add_argument("--config", type=Path, default=Path("configs/deepseek-60.json"))
p.add_argument("--out", type=Path, default=Path("runs/deepseek-progressive-60"))
p.add_argument("--stage", choices=STAGES + ["all", "status"], default="status")
args = p.parse_args()
campaign = Campaign(args.config, args.out)
if args.stage == "status":
    import json
    print(json.dumps(campaign.ledger.stats(), indent=2))
else:
    campaign.run(args.stage)
