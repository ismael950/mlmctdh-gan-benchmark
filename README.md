# mlmctdh-gan-benchmark

Benchmark del algoritmo cuántico de simulación para el modelo **Generalized
Anderson–Newns (GAN)** frente a **ML-MCTDH** y a la propagación exacta, y
estimación del **error de Trotter** de dicho algoritmo.

## Motivación

- El algoritmo cuántico GAN (Lang *et al.*, [arXiv:2601.16264](https://arxiv.org/abs/2601.16264))
  evoluciona en el tiempo bajo el Hamiltoniano GAN con una **fórmula de producto
  de primer orden** (fragmentación por *matchings* + diagonalización Clifford +
  QFT para la energía cinética nuclear). El paper reporta costos para 1000 pasos
  de Trotter pero deja explícitamente fuera *"a full analysis on the number of
  required Trotter steps"*.
- El método de estimación de error de Trotter (Maxwell *et al.*,
  [arXiv:2606.30738](https://arxiv.org/abs/2606.30738)) ve la trotterización como
  evolución bajo un Hamiltoniano efectivo `H' = H + Δt^d · E`, con `E` dado por la
  expansión BCH de la fórmula de producto, y estima el error propagando bajo `H'`
  (exacto para sistemas chicos; ML-MCTDH/DMRG para sistemas grandes).

Este repo une ambas piezas: mide cuántos pasos de Trotter necesita el algoritmo
GAN para un objetivo de precisión dado en un observable físico (poblaciones
electrónicas), usando un sistema chico con referencia exacta para validar el
estimador, y transfiriéndolo luego a sistemas grandes sin referencia exacta
(NO/Au(111)).

## Los cuatro benchmarks

| Nombre | Qué es | Estado |
|---|---|---|
| `benchmark1_newns_anderson` | Convergencia histórica de ML-MCTDH (Newns–Anderson). | archivado, `run_015`, error máx. de población molecular 1.5e-5 |
| `benchmark2_two_molecular` | Convergencia histórica de ML-MCTDH, 2 orbitales moleculares. | archivado, `run_010`, error máx. 3.7e-6 |
| `benchmark3_no_au_scattering` | Benchmark grande de escalamiento: NO/Au(111), estudio de convergencia de rango con propagaciones de 50 fs. | en curso (`run_007`) |
| `small_direct_benchmark` | Comparación directa de 3 vías (exacto vs ML-MCTDH convergido vs Trotter GAN) sobre **un mismo** Hamiltoniano GAN chico + validación del estimador BCH del error de Trotter. | ver `RESULTS.md` |

Métrica de error primaria en todos los casos:
`E_X = max_t max_i |n_i^X(t) − n_i^exact(t)|` sobre las poblaciones de orbitales
moleculares.

## Estructura del repositorio

```
src/ganbench/
  exact/         Construcción del Hamiltoniano y propagación exacta (state vector).
  quantum/       Modelo GAN de juguete, fragmentos, fórmula de Trotter de 1er orden,
                 backend PennyLane (circuitos), malla nuclear periódica.
  heidelberg/    Lectura/escritura y análisis de entradas/salidas de ML-MCTDH (Heidelberg).
  metal_bath.py, model.py, nuclear_grid.py, plotting.py, results.py

configs/         Configuraciones YAML de cada benchmark.
backend_inputs/  Entradas de ML-MCTDH (Heidelberg) versionadas: benchmark.inp por run.
scripts/
  run_benchmark.py     Corredor general (exacto + Heidelberg adaptativo).
  small_benchmark/      Pipeline del small_direct_benchmark.
  no_au/               Pipeline del benchmark3 NO/Au(111).
  diagnostics/         Inspecciones y chequeos puntuales (no producen resultados reportables).
  tests/              Tests de validación (pass/fail).

results/         (IGNORADO por git) Resultados numéricos generados.
checkpoints/     (IGNORADO) Checkpoints completos de Heidelberg.
figures/         (IGNORADO) Figuras generadas.
generated/       (IGNORADO) Entradas de Heidelberg generadas al vuelo.
```

## Scripts (`scripts/`)

Los scripts se corren **desde la raíz del repo** (`python scripts/<subcarpeta>/<script>.py`).

### `scripts/small_benchmark/`

| Script | Función |
|---|---|
| `generate_small_exact_reference.py` | Referencia exacta (state vector, malla K=32). Verdad de fondo. |
| `check_small_nuclear_basis_convergence.py` | Convergencia de la malla nuclear (K = 8/16/32/64). |
| `generate_small_trotter_sweep.py` | Trotteriza numéricamente el algoritmo GAN; barrido en dt; `E_trot(dt)` vs exacto. |
| `validate_small_effective_hamiltonian.py` | Estimador BCH (PennyLane `labs.trotter_error`, orden 2): construye `H_eff = (i/dt)·Ω`, lo propaga **exacto**, y reporta `E_eff` (predicción de `E_trot`) y `E_model` (residual `H_eff` vs Trotter, ~dt²). Correr con `.venv-estimator`. |
| `build_small_direct_comparison.py` | Ensambla la comparación de 3 vías (exacto / ML-MCTDH / Trotter). |
| `plot_small_direct_benchmark.py` | Figuras de la comparación de 3 vías. |
| `plot_trotter_error_estimator.py` | Figura del escalamiento del estimador (`E_trot`, `E_eff`, residual vs dt). |
| `plot_trotter_estimator_dynamics.py` | Dinámica del error para un dt fijo (2.5 a.u.). |

### `scripts/no_au/`

| Script | Función |
|---|---|
| `generate_no_au_benchmark.py` | Genera el benchmark físico NO/Au. |
| `generate_no_au_heidelberg_operator.py` | Escribe el operador de Heidelberg (`benchmark.op`). |
| `generate_no_au_heidelberg_input.py` | Escribe la entrada de Heidelberg (`benchmark.inp`). |
| `inspect_noau_convergence.py` | Seguimiento de la convergencia de rango de las corridas de Heidelberg. |
| `inspect_no_au_bch_structure.py` | Estructura de la expansión BCH del operador NO/Au. |
| `plot_mlmctdh_convergence.py` | Figura de convergencia de ML-MCTDH (coeficientes vs error); genérica. |

### `scripts/diagnostics/`

| Script | Función |
|---|---|
| `inspect_no_au_benchmark.py` | Inspección física del modelo NO/Au (superficies diabáticas, baño metálico, paquete de onda). |
| `check_small_nuclear_boundaries.py` | Chequeo puntual: la función de onda no toca el borde de la malla periódica. |

### `scripts/tests/`

| Script | Función |
|---|---|
| `test_quantum_trotter_k32.py` | Comprueba que el Trotter GAN a nivel operador coincide con el circuito PennyLane en K=32. |

### `scripts/run_benchmark.py`

Corredor general: backend exacto y backend Heidelberg (con refinamiento adaptativo de rangos SPF).

## Entornos

- **Principal**: `numpy`, `scipy`, `pandas`, `matplotlib`, `pennylane`, `qiskit`.
- **`.venv-estimator/`** (ignorado): entorno separado con PennyLane 0.45, que
  expone la API GenericFragment de `pennylane.labs.trotter_error`
  (`te.effective_hamiltonian`, `te.ProductFormula`, `te.generic_fragments`).
  Necesario para `small_benchmark/validate_small_effective_hamiltonian.py` y
  `no_au/inspect_no_au_bch_structure.py`. Instálalo con
  `.venv-estimator/Scripts/python -m pip install -e .`.
- **ML-MCTDH**: binario externo de Heidelberg
  (`mctdh86`, ruta local en `run_benchmark.py`). Corre en Linux, típicamente en
  un servidor/cluster. Este repo sólo genera/parsea sus entradas y salidas.

## Flujo de trabajo (small_direct_benchmark)

```bash
# Correr siempre desde la raíz del repo.

# 1. Referencia exacta y convergencia de malla
python scripts/small_benchmark/generate_small_exact_reference.py
python scripts/small_benchmark/check_small_nuclear_basis_convergence.py

# 2. Barrido de Trotter numérico -> E_trot(dt)
python scripts/small_benchmark/generate_small_trotter_sweep.py

# 3. Estimador BCH del error de Trotter (propagación exacta de H_eff)
.venv-estimator/Scripts/python scripts/small_benchmark/validate_small_effective_hamiltonian.py

# 4. Convergencia de ML-MCTDH: generar entradas, correr en el host de MCTDH,
#    y analizar. Las entradas versionadas están en
#    backend_inputs/small_direct_benchmark/heidelberg/run_XXX/benchmark.inp

# 5. Ensamblar comparación y figuras
python scripts/small_benchmark/build_small_direct_comparison.py
python scripts/small_benchmark/plot_small_direct_benchmark.py
python scripts/small_benchmark/plot_trotter_error_estimator.py
python scripts/small_benchmark/plot_trotter_estimator_dynamics.py
```

## Ramas

- `main` — implementación base de ML-MCTDH.
- `adaptive-convergence` — bucle adaptativo de convergencia de Heidelberg.
- `no-au-benchmark` — benchmark físico NO/Au(111).
- `quantum-gan-implementation` — algoritmo cuántico GAN, `small_direct_benchmark`
  y estimador de error de Trotter (rama de trabajo actual).

## Qué se versiona y qué no

Se versiona: código (`src/`, `scripts/`), configs, y `backend_inputs/*/benchmark.inp`.
No se versiona (ver `.gitignore`): `results/`, `checkpoints/`, `figures/`,
`generated/`, archivos `.dat`/`.op` reutilizados de Heidelberg, y cualquier
archivo comprimido.
