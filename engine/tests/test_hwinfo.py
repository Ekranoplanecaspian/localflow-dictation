"""The capability report (B1): what this computer has."""

import os

import pytest

from localflow import hwinfo, modelchoice


@pytest.mark.parametrize(("vendor", "name", "dedicated_mb", "integrated"), [
    ("nvidia", "NVIDIA GeForce RTX 4060 Laptop GPU", 7956, False),
    ("nvidia", "NVIDIA GeForce MX450", 512, False),  # small, but NVIDIA is always a card of its own
    ("amd", "AMD Radeon(TM) 890M Graphics", 338, True),
    ("amd", "AMD Radeon(TM) Graphics", 2048, True),  # a laptop with a large carve-out set in BIOS
    ("amd", "AMD Radeon RX 7600", 8176, False),
    ("amd", "AMD Radeon PRO W7600", 8176, False),
    ("intel", "Intel(R) UHD Graphics 620", 128, True),
    ("intel", "Intel(R) Arc(TM) Graphics", 2048, True),  # Meteor Lake's built-in Arc
    ("intel", "Intel(R) Arc(TM) A770 Graphics", 16256, False),
    ("intel", "Intel(R) Arc(TM) B580 Graphics", 12032, False),
    ("qualcomm", "Qualcomm(R) Adreno(TM) X1-85 GPU", 2048, True),
])
def test_integrated_or_discrete(vendor, name, dedicated_mb, integrated):
    assert hwinfo._is_integrated(vendor, name, dedicated_mb) is integrated


def test_integrated_graphics_can_use_what_it_borrows():
    igpu = hwinfo.Gpu("AMD Radeon(TM) 890M Graphics", "amd", 338, 15932, integrated=True)
    card = hwinfo.Gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 7956, 15932, integrated=False)
    assert igpu.usable_mb == 338 + 15932
    assert card.usable_mb == 7956  # borrowed RAM is far too slow for a discrete card to count


def test_free_disk_of_a_folder_that_does_not_exist_yet(tmp_path):
    free = hwinfo.disk_free_gb(tmp_path / "not" / "made" / "yet")
    assert free is not None and free > 0


def test_describe_reads_like_a_sentence(tmp_path):
    report = hwinfo.Report(
        cpu=hwinfo.Cpu("Some CPU", cores=4, threads=8, isa=("sse4.2", "avx", "avx2")),
        ram_gb=7.8,
        gpus=(hwinfo.Gpu("Intel(R) UHD Graphics 620", "intel", 128, 3968, integrated=True),),
    )
    text = "\n".join(hwinfo.describe(report, tmp_path))
    assert "Some CPU: 4 cores, 8 threads; instruction sets SSE4.2, AVX, AVX2" in text
    assert "Memory     7.8 GB" in text
    assert "Intel(R) UHD Graphics 620 (intel, integrated, shares RAM)" in text
    assert "free where models are kept" in text


def test_describe_says_when_nothing_is_known():
    report = hwinfo.Report(cpu=hwinfo.Cpu("x", 2, 2, None), ram_gb=4.0)
    text = "\n".join(hwinfo.describe(report))
    assert "instruction sets unknown" in text
    assert "Graphics   none found" in text


def test_hardware_carries_the_report():
    report = hwinfo.Report(cpu=hwinfo.Cpu("Some CPU", 6, 12, ("avx2",)), ram_gb=16.0,
                           gpus=(hwinfo.Gpu("AMD Radeon RX 7600", "amd", 8176, 8000, False),))
    hw = modelchoice.Hardware(cpu_cores=6, ram_gb=16.0, vram_mb=None, report=report)
    d = hw.as_dict()
    assert d["cpu"]["isa"] == ["avx2"]
    assert d["gpus"][0]["vendor"] == "amd"
    # without one (tests, older callers) the dict is what it always was
    assert modelchoice.Hardware(cpu_cores=6, ram_gb=16.0, vram_mb=None).as_dict() == {
        "cpu_cores": 6, "ram_gb": 16.0, "vram_mb": None}


@pytest.mark.skipif(os.name != "nt", reason="reads Windows")
def test_detect_on_this_machine(monkeypatch):
    monkeypatch.setattr(hwinfo, "physical_cores", hwinfo._real_physical_cores)  # conftest pins 12
    report = hwinfo.detect()
    assert report.cpu.cores >= 1 and report.cpu.threads >= report.cpu.cores
    assert report.ram_gb > 1
    for g in report.gpus:  # the Basic Render Driver is never listed
        assert g.vendor != "other" or "Microsoft" not in g.name
