#!/bin/zsh
# Lanza la rejilla del notebook 04 en paralelo y consolida los checkpoints.
#
#   ./scripts/lanzar_rejilla_paralela.sh [n_workers] [python]
#
# Despues de esto, reejecutar notebooks/04 y 05: encontraran todas las
# combinaciones en los checkpoints, se las saltaran, y solo generaran las
# tablas y las graficas (~2 min en vez de 2 h).
set -u
N=${1:-10}
PY=${2:-python}
cd "$(dirname "$0")/.."

# Cerrojo: dos lanzamientos simultaneos ponen el doble de procesos que
# nucleos hay y ademas se pisan los shards. Paso de verdad durante el
# desarrollo (16 workers en 10 nucleos), asi que la segunda invocacion
# aborta en vez de duplicar el trabajo.
LOCK=datos/interim/.lock_rejilla
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Ya hay una ejecucion en curso ($LOCK). Si no es asi, borra ese directorio."
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

echo "=== reparto previsto ==="
$PY scripts/rejilla_paralela.py --plan --n-workers $N

echo "\n=== lanzando $N workers  $(date +%H:%M:%S) ==="
mkdir -p datos/interim/paralelo/logs
for w in $(seq 0 $((N-1))); do
  $PY scripts/rejilla_paralela.py --worker $w --n-workers $N \
      > datos/interim/paralelo/logs/w$w.log 2>&1 &
done
wait

echo "\n=== consolidando  $(date +%H:%M:%S) ==="
$PY scripts/rejilla_paralela.py --merge --n-workers $N
echo "=== listo  $(date +%H:%M:%S) ==="
