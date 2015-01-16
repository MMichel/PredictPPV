PredictPPV
==========

Python package amino acid contact map quality prediction.
[Soon also available via the Python packaging index (PyPi)](https://pypi.python.org/pypi/PredictPPV).


Installation
============

Soon PredictPPV can be installed from PyPi with:

``pip -U PredictPPV``

or

``easy_install -U PredictPPV``


Usage
=====

Within python:

```from contactvis import predictppv```

```plot_contact_map.plot_map(fasta_filename, contact_filename, factor, c2_filename='', psipred_filename='', pdb_filename='', is_heavy=False, chain='', sep=',', outfilename='')```

Terminal:

```plot_contact_map [-h] [-i] [-o OUTFILE] [-f FACTOR] [--c2 C2]
                    [--psipred_horiz PSIPRED_HORIZ] [--pdb PDB]
                    [--heavy] [--chain CHAIN]
                    fasta_file contact_file```



To reproduce the examples in ``test`` run the following commands:


Simple map of the given contact file with coloring according to contact probability:

```python ../plot_contact_map.py sequence.fasta predicted.contacts -o cm_simple```


Comparison to contacts from the native PDB structure (pairwise CB-atom distance with 8Å cutoff):

```python ../plot_contact_map.py sequence.fasta predicted.contacts -o cm_pdb.pdf --pdb native_structure.pdb```


Compare two different predicted contact maps to each other and to a native PDB structure and include secondary structure information along the diagonal (red: helix, blue: sheet):

```python ../plot_contact_map.py sequence.fasta predicted.contacts -o cm_compare_pdb.pdf --pdb native_structure.pdb --c2 predicted.contacts2 --psipred_horiz psipred.horiz```
