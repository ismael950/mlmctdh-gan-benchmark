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
scripts/         Generadores, corredores, validadores y graficadores (ver abajo).

results/         (IGNORADO por git) Resultados numéricos generados.
checkpoints/     (IGNORADO) Checkpoints completos de Heidelberg.
figures/         (IGNORADO) Figuras generadas.
generated/       (IGNORADO) Entradas de Heidelberg generadas al vuelo.
```

## Scripts (`scripts/`)

**small_direct_benchmark**

| Script | Función |
|---|---|
| `generate_small_exact_reference.py` | Referencia exacta (state vector, malla K=32). Verdad de fondo. |
| `check_small_nuclear_basis_convergence.py` | Convergencia de la malla nuclear (K = 8/16/32/64). |
| `check_small_nuclear_boundaries.py` | Chequeo de fronteras de la malla nuclear. |
| `generate_small_trotter_sweep.py` | Trotteriza numéricamente el algoritmo GAN; barrido en dt; `E_trot(dt)` vs exacto. |
| `validate_small_effective_hamiltonian.py` | Estimador BCH (PennyLane `labs.trotter_error`, orden 2): construye `H_eff = (i/dt)·Ω`, lo propaga **exacto**, y reporta `E_eff` (predicción de `E_trot`) y `E_model` (residual `H_eff` vs Trotter, ~dt²). |
| `validate_quantum_trotter_k32.py` | Validación del circuito Trotter GAN (PennyLane) contra la trotterización numérica en K=32. |
| `build_small_direct_comparison.py` | Ensambla la comparación de 3 vías (exacto / ML-MCTDH / Trotter). |
| `plot_small_direct_benchmark.py` | Figuras de la comparación de 3 vías. |
| `plot_mlmctdh_convergence.py` | Figura de convergencia de ML-MCTDH (coeficientes vs error). |
| `plot_trotter_error_estimator.py` | Figura del escalamiento del estimador (`E_trot`, `E_eff`, residual vs dt). |
| `plot_trotter_estimator_dynamics.py` | Dinámica del error para un dt fijo (2.5 a.u.). |

**benchmark3_no_au_scattering**

| Script | Función |
|---|---|
| `generate_no_au_benchmark.py` | Genera el benchmark físico NO/Au. |
| `generate_no_au_heidelberg_operator.py` | Escribe el operador de Heidelberg (`benchmark.op`). |
| `generate_no_au_heidelberg_input.py` | Escribe la entrada de Heidelberg (`benchmark.inp`). |
| `inspect_no_au_benchmark.py`, `inspect_noau_convergence.py`, `inspect_no_au_bch_structure.py` | Inspección del benchmark, de la convergencia de rango y de la estructura de la expansión BCH. |

**General**

| Script | Función |
|---|---|
| `run_benchmark.py` | Corredor: backend exacto y backend Heidelberg (con refinamiento adaptativo de rangos SPF). |

## Entornos

- **Principal**: `numpy`, `scipy`, `pandas`, `matplotlib`, `pennylane`, `qiskit`.
- **`.venv-estimator/`** (ignorado): entorno separado con la versión de PennyLane
  que expone `pennylane.labs.trotter_error` (`te.effective_hamiltonian`,
  `te.ProductFormula`, `te.generic_fragments`). Necesario para
  `validate_small_effective_hamiltonian.py` e `inspect_no_au_bch_structure.py`.
- **ML-MCTDH**: binario externo de Heidelberg
  (`mctdh86`, ruta local en `run_benchmark.py`). Corre en Linux, típicamente en
  un servidor/cluster. Este repo sólo genera/parsea sus entradas y salidas.

## Flujo de trabajo (small_direct_benchmark)

```bash
# 1. Referencia exacta y convergencia de malla
python scripts/generate_small_exact_reference.py
python scripts/check_small_nuclear_basis_convergence.py

# 2. Barrido de Trotter numérico -> E_trot(dt)
python scripts/generate_small_trotter_sweep.py

# 3. Estimador BCH del error de Trotter (propagación exacta de H_eff)
python scripts/validate_small_effective_hamiltonian.py   # usa .venv-estimator

# 4. Convergencia de ML-MCTDH: generar entradas, correr en el host de MCTDH,
#    y analizar. Las entradas versionadas están en
#    backend_inputs/small_direct_benchmark/heidelberg/run_XXX/benchmark.inp

# 5. Ensamblar comparación y figuras
python scripts/build_small_direct_comparison.py
python scripts/plot_small_direct_benchmark.py
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
