"""Unit tests for the interpretable descriptor block (pure CPU, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abdev.features.descriptors import descriptors_for_pair, _net_charge, _approx_pi

VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
      "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYYGMDVWGQGTTVTVSS")
VL = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGS"
      "GTDFTLTISSLQPEDFATYYCQQSYSTPLTFGQGTKVEIK")


def test_descriptor_count_and_keys():
    d = descriptors_for_pair(VH, VL)
    assert len(d) > 100
    assert "H_charge_pH" in d and "CDR3_net_charge" in d
    assert d["H_len"] == len(VH)


def test_charge_monotonic_in_ph():
    # Net charge decreases as pH rises.
    assert _net_charge(VH, 4.0) > _net_charge(VH, 10.0)


def test_pi_in_range():
    pi = _approx_pi(VH)
    assert 3.0 < pi < 12.0


def test_empty_sequence_safe():
    d = descriptors_for_pair("", "")
    assert d["H_len"] == 0.0


if __name__ == "__main__":
    test_descriptor_count_and_keys()
    test_charge_monotonic_in_ph()
    test_pi_in_range()
    test_empty_sequence_safe()
    print("All descriptor tests passed.")
