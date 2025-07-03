from mechanism import Mechanism

def test_load():
    mech = Mechanism('data/gri30.yaml')
    assert 'CH4' in mech.species_names
