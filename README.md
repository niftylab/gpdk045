# gpdk045
BAG primitives for gpdk045 technology

The LAYGO technology file defines `routing_56_basic`, `routing_56_cmos`,
`routing_67_basic`, and `routing_67_cmos` for Metal5 through Metal7 routing.
Before exporting a layout that uses `routing_67_*` to Virtuoso, initialize the
native upper-via microtemplate from a live BAG workspace:

```bash
python gpdk045/laygo2_tech/ensure_upper_via_cells.py
```

The command creates `gpdk045_microtemplates_dense/via_M6_M7_0/layout` with the
GPDK045 standard-via definition `M7_M6`, or verifies the existing cell.
