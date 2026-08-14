import sys

from scipy import sparse

from ganbench.model import load_config
from ganbench.exact.hamiltonian import build_exact_hamiltonian


def main() -> None:
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "configs/validation_small.yaml"
    )

    config = load_config(config_file)
    system = build_exact_hamiltonian(config)

    hermiticity_error = sparse.linalg.norm(
        system.hamiltonian - system.hamiltonian.getH()
    )

    initial_norm = (
        system.initial_state.conj()
        @ system.initial_state
    ).real

    total_number_el = sum(
        system.electronic_number_operators,
        sparse.csr_matrix(
            (
                system.electronic_dimension,
                system.electronic_dimension,
            ),
            dtype=complex,
        ),
    )

    total_number = sparse.kron(
        total_number_el,
        sparse.identity(
            system.vibrational_dimension,
            format="csr",
        ),
        format="csr",
    )

    number_commutator = (
        system.hamiltonian @ total_number
        - total_number @ system.hamiltonian
    )

    number_conservation_error = sparse.linalg.norm(
        number_commutator
    )

    initial_particle_number = (
        system.initial_state.conj()
        @ (
            total_number
            @ system.initial_state
        )
    ).real

    print("Config:", config_file)
    print("Electronic dimension:", system.electronic_dimension)
    print("Vibrational dimension:", system.vibrational_dimension)
    print("Total dimension:", system.total_dimension)
    print("Hamiltonian shape:", system.hamiltonian.shape)
    print("Hermiticity error:", hermiticity_error)
    print("Initial-state norm:", initial_norm)
    print(
        "Particle-number commutator:",
        number_conservation_error,
    )
    print(
        "Initial particle number:",
        initial_particle_number,
    )

    print("\nElectronic Fock basis:")
    for index, occupation in enumerate(system.electronic_basis):
        marker = ""

        initial_occupation = [0] * config.n_orbitals
        for orbital in config.occupied_orbitals:
            initial_occupation[orbital] = 1

        if tuple(initial_occupation) == occupation:
            marker = "  <-- initial state"

        print(f"{index}: |{''.join(map(str, occupation))}> {marker}")


if __name__ == "__main__":
    main()