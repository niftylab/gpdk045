"""Regression tests for the GPDK045 M5-M7 routing grid stack."""

from pathlib import Path

import yaml


TECH_YAML = Path(__file__).resolve().parents[1] / "laygo2_tech" / "laygo2_tech.yaml"
VIA_BOOTSTRAP = (
    Path(__file__).resolve().parents[1]
    / "laygo2_tech"
    / "ensure_upper_via_cells.py"
)
LIBNAME = "gpdk045_microtemplates_dense"


def _tech_params():
    return yaml.safe_load(TECH_YAML.read_text(encoding="utf-8"))


def _assert_grid_stack(grid, horizontal_layer, vertical_layer, via_cell):
    assert {layer[0] for layer in grid["horizontal"]["layer"]} == {
        horizontal_layer
    }
    assert {layer[0] for layer in grid["horizontal"]["pin_layer"]} == {
        horizontal_layer
    }
    assert {layer[0] for layer in grid["vertical"]["layer"]} == {
        vertical_layer
    }
    assert {layer[0] for layer in grid["vertical"]["pin_layer"]} == {
        vertical_layer
    }
    assert {
        cellname
        for row in grid["via"]["map"]
        for cellname in row
    } == {via_cell}


def test_gpdk045_defines_basic_and_cmos_grids_through_metal7():
    params = _tech_params()
    grids = params["grids"][LIBNAME]

    _assert_grid_stack(
        grids["routing_56_basic"],
        horizontal_layer="Metal6",
        vertical_layer="Metal5",
        via_cell="via_M5_M6_0",
    )
    _assert_grid_stack(
        grids["routing_56_cmos"],
        horizontal_layer="Metal6",
        vertical_layer="Metal5",
        via_cell="via_M5_M6_0",
    )
    _assert_grid_stack(
        grids["routing_67_basic"],
        horizontal_layer="Metal6",
        vertical_layer="Metal7",
        via_cell="via_M6_M7_0",
    )
    _assert_grid_stack(
        grids["routing_67_cmos"],
        horizontal_layer="Metal6",
        vertical_layer="Metal7",
        via_cell="via_M6_M7_0",
    )

    assert len(grids["routing_56_cmos"]["horizontal"]["elements"]) == 10
    assert len(grids["routing_67_cmos"]["horizontal"]["elements"]) == 10


def test_gpdk045_exports_upper_metal_and_via_layers():
    params = _tech_params()
    templates = params["templates"][LIBNAME]
    mpl = params["export"]["mpl"]

    assert "via_M5_M6_0" in templates
    assert "via_M6_M7_0" in templates
    for layer_name in ("Metal6", "Metal7", "Via6"):
        assert layer_name in mpl["colormap"]
    assert mpl["order"][-5:] == [
        "Metal6",
        "M6",
        "Via6",
        "Metal7",
        "M7",
    ]


def test_upper_via_bootstrap_is_idempotent_and_uses_the_g45_via_definition():
    source = VIA_BOOTSTRAP.read_text(encoding="utf-8")
    compile(source, str(VIA_BOOTSTRAP), "exec")
    assert 'CELLNAME = "via_M6_M7_0"' in source
    assert 'VIA_DEF_NAME = "M7_M6"' in source
    assert "techFindViaDefByName" in source
    assert "dbCreateVia" in source
    assert 'result="existing"' in source
    assert 'result="created"' in source
