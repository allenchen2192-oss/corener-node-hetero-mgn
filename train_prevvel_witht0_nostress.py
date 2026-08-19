# w_stress=0.0 + includes t=0 in training
# 與 train_prevvel_witht0.py 完全相同，只有 config_name 不同
# 對應 conf/config_prevvel_no_nodetype_witht0_nostress.yaml

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "train_prevvel_witht0.py",
         "--config-name=config_prevvel_no_nodetype_witht0_nostress"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    sys.exit(result.returncode)
