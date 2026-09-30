"""Ejecuta todo el laboratorio en orden. Uso:  python src/run_all.py"""
import runpy
import sys
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here))
for script in ["t1_1_download", "t1_2_eda", "t1_3_coverage", "t2_1_sensitivity", "t2_2_vulnerability", "t2_3_mclp"]:
    print(f"\n######## {script} ########")
    runpy.run_path(str(here / f"{script}.py"), run_name="__main__")
