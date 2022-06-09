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
    force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", 2.5 * spring_unit)
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


def refine(input_file, output_file, n=5):
    for _ in range(n):
        try:
            refine_once(input_file, output_file)
        except Exception:
            continue
        else:
            break


