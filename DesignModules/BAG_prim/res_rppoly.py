# -*- coding: utf-8 -*-

import os
import pkg_resources

from bag.design.module import ResPhysicalModuleBase


# noinspection PyPep8Naming
class BAG_prim__res_rppoly(ResPhysicalModuleBase):
    """design module for BAG_prim__res_rppoly.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'res_rppoly.yaml'))

    def __init__(self, database, parent=None, prj=None, **kwargs):
        ResPhysicalModuleBase.__init__(self, database, self.yaml_file, parent=parent, prj=prj, **kwargs)

