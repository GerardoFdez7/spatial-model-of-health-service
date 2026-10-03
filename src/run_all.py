# Ejecuta todos los tasks en orden y guarda la salida de cada uno en outputs/<script>_log.txt.
# Uso: python src/run_all.py           (todos)
#      python src/run_all.py t2_2 t3_1 (solo los que empiecen así)
import contextlib
import runpy
import sys
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here))
from config import OUT  # noqa: E402

SCRIPTS = ["t1_1_carga", "t1_2_estadisticas", "t1_3_buffers", "t2_1_sensibilidad", "t2_2_vulnerabilidad",
           "t2_3_mclp", "t3_1_isocronas", "t3_3_sensibilidad_pesos"]


# Escribe la salida a la vez en consola y en el archivo de log.
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


if __name__ == "__main__":
    sel = [s for s in SCRIPTS if not sys.argv[1:] or any(s.startswith(a) for a in sys.argv[1:])]
    for script in sel:
        print(f"\n# {script} #")
        with open(OUT / f"{script}_log.txt", "w", encoding="utf-8") as fh, \
                contextlib.redirect_stdout(Tee(sys.stdout, fh)):
            runpy.run_path(str(here / f"{script}.py"), run_name="__main__")
