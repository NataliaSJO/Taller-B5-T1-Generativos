#!/bin/zsh
# Lanza la comparacion de arquitecturas del notebook 04 en paralelo.
set -u
PY=${1:-python}
cd "$(dirname "$0")/.."

# Cerrojo: dos lanzamientos simultaneos ponen el doble de procesos que
# nucleos hay y ademas se pisan los shards. Paso de verdad durante el
# desarrollo (16 workers en 10 nucleos), asi que la segunda invocacion
# aborta en vez de duplicar el trabajo.
LOCK=datos/interim/.lock_arquitecturas
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Ya hay una ejecucion en curso ($LOCK). Si no es asi, borra ese directorio."
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
mkdir -p datos/interim/paralelo/arquitecturas/logs
echo "=== lanzando 8 arquitecturas  $(date +%H:%M:%S) ==="
for a in $($PY scripts/comparar_arquitecturas_paralelo.py); do
  $PY scripts/comparar_arquitecturas_paralelo.py --arquitectura $a \
      > datos/interim/paralelo/arquitecturas/logs/$a.log 2>&1 &
done
wait
echo "=== consolidando  $(date +%H:%M:%S) ==="
$PY scripts/comparar_arquitecturas_paralelo.py --merge
echo "=== listo  $(date +%H:%M:%S) ==="
