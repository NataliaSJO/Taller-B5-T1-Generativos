# Fase 2 de la busqueda, encadenada automaticamente.
#
# Espera a que terminen los 6 workers de generadores (~07:45) y entonces
# reaprovecha TODOS los cores para la busqueda del predictor con validacion
# walk-forward, en dos variantes que se compararan en el informe:
#   - 4 workers CON purga    (embargo 60d, protocolo riguroso)
#   - 4 workers SIN purga    (embargo 0d,  protocolo ingenuo)
#
# Se hace en etapa A (synth_years=0, SOLO datos reales) porque esa busqueda
# no depende de los generadores: es valida independientemente de que
# hiperparametros de generador acaben eligiendose. La etapa B (93%
# sintetica) se lanza despues, a mano, cuando los datasets sinteticos se
# hayan regenerado con los mejores generadores.

$repo = "c:\Users\1jose\Desktop\educacion\master_bme\modulo 5\Git\Taller-B5-T1-Generativos"
$py   = "C:\Users\1jose\anaconda3\envs\taller_gen\python.exe"
$logs = "$repo\reports\tables\hpsearch_logs"
Set-Location $repo

# --- 1. esperar a que acaben los generadores -------------------------------
while ($true) {
  $done = (Get-ChildItem "$logs\gen_w*.log" -ErrorAction SilentlyContinue |
           Where-Object { Select-String -Path $_.FullName -Pattern 'TERMINADO' -Quiet }).Count
  $alive = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'hp_search_generators' }).Count
  if ($done -ge 6 -or $alive -eq 0) { break }
  Start-Sleep -Seconds 120
}
"$(Get-Date -Format 'HH:mm') generadores terminados -> arrancando fase 2" |
  Out-File "$logs\fase2.log" -Append

# --- 2. liberar los workers de corte unico --------------------------------
# Ya habran acumulado configuraciones de sobra para servir como tercera
# variante de comparacion (corte unico) en el informe.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'hp_search\.py' -and $_.CommandLine -notmatch 'walk-forward' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 5

# --- 3. lanzar walk-forward, con y sin purga ------------------------------
foreach ($w in 0..3) {
  Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden `
    -ArgumentList "scripts/hp_search.py","--stage","A","--walk-forward",
                  "--embargo-days","60","--minutes","600","--worker","$w" `
    -RedirectStandardOutput "$logs\wfA_emb60_w$w.log" -RedirectStandardError "$logs\wfA_emb60_w$w.err"
}
foreach ($w in 0..3) {
  Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden `
    -ArgumentList "scripts/hp_search.py","--stage","A","--walk-forward",
                  "--embargo-days","0","--minutes","600","--worker","$w" `
    -RedirectStandardOutput "$logs\wfA_emb0_w$w.log" -RedirectStandardError "$logs\wfA_emb0_w$w.err"
}
Start-Sleep -Seconds 10
$n = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'hp_search' }).Count
"$(Get-Date -Format 'HH:mm') fase 2 en marcha: $n workers walk-forward (4 con purga + 4 sin purga)" |
  Out-File "$logs\fase2.log" -Append
