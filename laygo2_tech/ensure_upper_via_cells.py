"""Ensure native GPDK045 microtemplate cells exist for upper routing vias.

Run this script from an initialized GPDK045 BAG workspace with a live
Virtuoso SKILL server.  The operation is idempotent and preserves an existing
cell when it already contains the expected standard via definition.
"""

from bag.core import BagProject


LIBNAME = "gpdk045_microtemplates_dense"
CELLNAME = "via_M6_M7_0"
VIA_DEF_NAME = "M7_M6"


def ensure_upper_via_cell(project):
    expression = f'''let((lib tf cv viaDef existing result)
      lib=ddGetObj("{LIBNAME}")
      unless(lib error("missing OA library: {LIBNAME}"))
      tf=techGetTechFile(lib)
      unless(tf error("missing technology file for: {LIBNAME}"))
      viaDef=techFindViaDefByName(tf "{VIA_DEF_NAME}")
      unless(viaDef error("missing standard via definition: {VIA_DEF_NAME}"))

      cv=dbOpenCellViewByType("{LIBNAME}" "{CELLNAME}" "layout" nil "r")
      if(cv then
        existing=cv~>vias~>viaHeader~>viaDefName
        dbClose(cv)
        unless(equal(existing list("{VIA_DEF_NAME}"))
          error("{LIBNAME}/{CELLNAME}: unexpected via definitions %L" existing)
        )
        result="existing"
      else
        cv=dbOpenCellViewByType(
          "{LIBNAME}" "{CELLNAME}" "layout" "maskLayout" "w"
        )
        unless(cv error("cannot create {LIBNAME}/{CELLNAME}/layout"))
        dbCreateVia(cv viaDef 0:0 "R0")
        dbSave(cv)
        dbClose(cv)
        result="created"
      )
      result
    )'''
    return project.impl_db._eval_skill(expression)


if __name__ == "__main__":
    outcome = ensure_upper_via_cell(BagProject())
    print(f"{LIBNAME}/{CELLNAME}: {outcome}", flush=True)
