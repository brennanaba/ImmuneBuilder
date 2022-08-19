import pdbfixer
from simtk.openmm import app, LangevinIntegrator, CustomExternalForce
from simtk import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)


def refine_once(input_file, output_file):
    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Using amber14 recommended protein force field
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")

    # Fill in the gaps with OpenMM Modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    # Set up force field
    system = forcefield.createSystem(modeller.topology)

    # Keep atoms close to initial prediction
    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", 1.0 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["CA", "CB", "N", "C"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    # Set up integrator
    integrator = LangevinIntegrator(0, 0.01, 0.0)

    # Set up the simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize the energy
    simulation.minimizeEnergy()

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(),
                              out_handle, keepIds=True)


def peptide_bonds_check(file_name, tol = 0.1):
    with open(file_name) as file:
        txt = file.readlines()

    Ns = [x for x in txt if x[13:16] == "N  "]
    Cs = [x for x in txt if x[13:16] == "C  "]
    all_good = True

    for i,n_line in enumerate(Ns[1:]):
        c_line = Cs[i]

        n_chain = n_line[21:23]
        c_chain = c_line[21:23]

        if c_chain != n_chain:
            continue

        x_diff = float(c_line[30:38]) - float(n_line[30:38])
        y_diff = float(c_line[38:46]) - float(n_line[38:46])
        z_diff = float(c_line[46:54]) - float(n_line[46:54])
        bond_error = abs((x_diff**2 + y_diff**2 + z_diff**2)**(1/2) - 1.32901)

        if bond_error > tol:
            all_good = False
            break
    return all_good


def refine(input_file, output_file, n=5):
    for _ in range(n):
        try:
            refine_once(input_file, output_file)
        except Exception:
            print("REFINEMENT FAILED ONCE: Trying again")
            continue
        else:
            if peptide_bonds_check(output_file):
                break
            else:
                input_file = output_file


