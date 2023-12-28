# Import Cadence library into the YAML file.
# Initialization. Taeho Shin. 07-26-2023
# Colormap import method added. Taeho Shin. 10-17-2023
# Importing functions are separated. Taeho Shin. 10-23-2023

import numpy as np
import yaml
import laygo2

grid_libname         = "gpdk045_microtemplates_dense" # target library
template_libname     = "gpdk045_microtemplates_dense"
output_name          = "laygo2_tech_tmp_231022"       # output yaml name will be "output_name.yaml" in the bottommost line.
placementgrid_prefix = "placement"                    # prefix of the placement grid in the target library.
routegrid_prefix     = "route"                        # prefix of the routing grid in the target library.
layer_boundary       = ["prBoundary", "boundary"]      # boundary layer of the target library. called from laygo2_tech/__init__.py (?)
save_rect            = False                          # save the rectangles in the cell if true.
print_log            = False                          # print messages.
mpl_flag             = False                          # flag for mpl colormap creation.
mpt = False

def import_grids(grid_libname=grid_libname, placementgrid_prefix="placement",
                 routegrid_prefix="route", ext_input=dict(), print_log=True, mpt=False):
    """
    Import grids from the specified library and construct a grid dictionary.

    Parameters
    ----------
    grid_libname : str
        The name of the library containing grids.
    placementgrid_prefix : str, optional
        The prefix of the placement grid.
    routegrid_prefix : str, optional
        The prefix of the routing grid.
    ext_input : dict, optional
        The user-defined extension of the metal layer.
        Recommended format: {"M1": 100, "M2": 100}
    print_log : boolean, optional
        flag for printing logs.
    mpt : boolean, optional
        flag for mpt support.
    
    """

    print(f"LAYGO2 is now importing grids from \"{grid_libname}\" into yaml file.")
    db = laygo2.interface.bag.load(libname=grid_libname, mpt=mpt) # create laygo2.database.Library to handle the technology library.
    grids = dict()
    grids[grid_libname] = dict()

    for cellname in db.keys():
        cell = db[cellname]

        if cellname.startswith(placementgrid_prefix): # placement grid
            if print_log:
                print("import placementgrid: " + cellname)
            # directly create laygo2.laygo2.object.PlacementGrid ??
            """
            name = cellname
            for rect in cell.rects.values():
                if rect.layer == layer_boundary: bnd = rect
            xgrid = laygo2.laygo2.object.OneDimGrid(name='xgrid', scope=bnd.xy[1][0], elements=bnd.xy[1][0])
            ygrid = laygo2.laygo2.object.OneDimGrid(name='ygrid', scope=bnd.xy[1][1], elements=bnd.xy[1][1])
            g = laygo2.laygo2.object.PlacementGrid(name=name, hgrid=ygrid, vgrid=xgrid)
            """
            gdict = dict()
            for rect in cell.rects.values():
                if rect.layer == layer_boundary:
                    rect = rect

            gdict['horizontal'] = dict()
            gdict['horizontal']['elements'] = [int(rect.xy[0][1])]
            gdict['horizontal']['scope']    = [int(rect.xy[0][1]), int(rect.xy[1][1])]

            gdict['type']                   = "placement"

            gdict['vertical']               = dict()
            gdict['vertical']['elements']   = [int(rect.xy[0][0])]
            gdict['vertical']['scope']      = [int(rect.xy[0][0]), int(rect.xy[1][0])]

            grids[grid_libname][cellname] = gdict

        elif cellname.startswith(routegrid_prefix): # routing grid
            if print_log:
                print("import routinggrid: " + cellname)
            gdict = dict()
            gdict['type']         = "routing"
            gdict['primary_grid'] = "horizontal"
            xelem, xwidth, xlayer, xcolor, xpin, xext, xext0 = [], [], [], [], [], [], []
            yelem, ywidth, ylayer, ycolor, ypin, yext, yext0 = [], [], [], [], [], [], []
            for rect in cell.rects.values():
                if rect.layer == layer_boundary: # find the scope of the grid.
                    bnd = rect.xy
                    xscope = [int(bnd[0][0]), int(bnd[1][0])]
                    yscope = [int(bnd[0][1]), int(bnd[1][1])]
                else: # routing layers
                    if rect.width <= rect.height: # xgrid
                        xelem.append(int(rect.center[0]))
                        xwidth.append(int(rect.width))
                        xlayer.append(rect.layer)
                        xcolor.append(rect.color if rect.color != None else "not MPT")
                        xpin.append([rect.layer[0], "pin"])

                        if rect.layer[0] in ext_input:
                            xext.append(int(ext_input[rect.layer[0]]))
                            # xext0.append(int(ext0_input[rect.layer[0]]))
                            xext0.append(xext[-1]*4)
                        else:
                            xext.append(int(rect.width//2))
                            xext0.append(int(rect.width*2))

                    else: # ygrid
                        yelem.append(int(rect.center[1]))
                        ywidth.append(int(rect.height))
                        ylayer.append(rect.layer)
                        ycolor.append(rect.color if rect.color != None else "not MPT")
                        ypin.append([rect.layer[0], "pin"])

                        if rect.layer[0] in ext_input:
                            yext.append(int(ext_input[rect.layer[0]]))
                            # yext0.append(int(ext0_input[rect.layer[0]]))
                            yext0.append(yext[-1]*4)
                        else:
                            yext.append(int(rect.height//2))
                            yext0.append(int(rect.height*2))

            # sort by element
            xidx = np.array(range(len(xwidth)))
            yidx = np.array(range(len(ywidth)))

            xg = np.vstack((xelem, xwidth, xext, xext0, xidx))
            yg = np.vstack((yelem, ywidth, yext, yext0, yidx))
            xg = xg.T[xg.T[:, 0].argsort()].T
            yg = yg.T[yg.T[:, 0].argsort()].T
            xelem, xwidth, xext, xext0, xidx = xg[0,:], xg[1,:], xg[2,:], xg[3,:], xg[4,:]
            yelem, ywidth, yext, yext0, yidx = yg[0,:], yg[1,:], yg[2,:], yg[3,:], yg[4,:]
            if print_log:
                print("    xelem:  " + str(xelem))
                print("    xwidth: " + str(xwidth))
                print("    xext:   " + str(xext))
                print("    xidx:   " + str(xidx))
                print("    yelem:  " + str(yelem))
                print("    ywidth: " + str(ywidth))
                print("    yext:   " + str(yext))
                print("    yidx:   " + str(yidx))

            xcolors, ycolors = [], []
            for i in range(len(xcolor)): xcolors.append(xcolor[int(xg[4,i])])
            for i in range(len(ycolor)): ycolors.append(ycolor[int(yg[4,i])])

            gdict['vertical'] = dict()
            gdict['vertical']['elements']     = xelem.tolist()
            gdict['vertical']['scope']        = xscope
            gdict['vertical']['width']        = xwidth.tolist()
            gdict['vertical']['extension']    = xext.tolist()
            gdict['vertical']['extension0']   = xext0.tolist()
            gdict['vertical']['layer']        = xlayer
            gdict['vertical']['pin_layer']    = xpin
            gdict['vertical']['xcolor']       = xcolors

            gdict['horizontal'] = dict()
            gdict['horizontal']['elements']   = yelem.tolist()
            gdict['horizontal']['scope']      = yscope
            gdict['horizontal']['width']      = ywidth.tolist()
            gdict['horizontal']['extension']  = yext.tolist()
            gdict['horizontal']['extension0'] = yext0.tolist()
            gdict['horizontal']['layer']      = ylayer
            gdict['horizontal']['pin_layer']  = ypin
            gdict['horizontal']['ycolor']     = ycolors

            # viamap
            # sort by the dimmension of the xwidth.
            gdict['via'] = dict()
            viamap = dict()
            for via in cell.instances.values(): 
                if via.cellname not in viamap: viamap[via.cellname] = via.center
                else: viamap[via.cellname] = np.vstack((viamap[via.cellname], via.center))

            for name, loc in viamap.items():
                if not loc.ndim==1:
                    viamap[name] = loc[loc[:,1].argsort()].tolist()
                else:
                    viamap[name] = viamap[name].tolist()

            # gdict['via']['map'] = [[0]*len(yelem)]*len(xelem) # initialize the dictionary as the list.
            # upper definition does not work.
            gdict['via']['map'] = np.zeros((len(xelem),len(yelem))).tolist() # initialize the dictionary as the list.
            for name, loc in viamap.items():
                if isinstance(loc[0], int): _loc = [loc] # change a single element into list.
                else: _loc = loc

                for point in _loc:
                    xidx = xelem.tolist().index(point[0]) # find the x index from xelem.
                    yidx = yelem.tolist().index(point[1]) # find the y index from yelem.
                    gdict['via']['map'][xidx][yidx] = name

            newcellname = f"routing_{cellname[7]}{cellname[10:]}" # This is an ad hoc.
            grids[grid_libname][newcellname] = gdict 

        else:
            pass # Do nothing for cells that are not grids.
    return grids

def import_templates(template_libname=template_libname, cellname=None, save_rect=False, mpt=False):
    """
    Import templates from the specified library and construct a template dictionary.

    Parameters
    ----------
    template_libname : str
        The name of the library containing template.
    cellname : list, optional
        Cellname list to be imported.
    save_rect : boolean, optional
        flag for importing routing wires.
    mpt : boolean, optional
        flag for mpt support.
    
    """

    print(f"LAYGO2 is now importing templates from \"{template_libname}\" into yaml file.")
    templates = dict()
    templates[template_libname] = dict()

    # for libname in template_libname:
    db = laygo2.interface.bag.load(libname=template_libname, cellname=None, mpt=mpt) # create laygo2.database.Library to handle the technology library.
    for cellname in db.keys():
        cell = db[cellname]
        if (not cellname.startswith(placementgrid_prefix)) and (not cellname.startswith(routegrid_prefix)):
            tdict = dict()
            # find the boundary
            for key, rect in cell.rects.items():
                if rect.layer == layer_boundary: # calculate the bbox using predefined boundary layer.
                    tdict['unit_size'] = np.round((rect.xy[1]-rect.xy[0]), decimals=1).tolist()
                    tdict['xy']        = rect.xy.tolist()
                elif save_rect: # save routing wires if the flag is true.
                    # TODO: contain only routing layers. exclude layers such as NP, PP, OD and etc.
                    tdict['rects'] = dict()
                 # TODO: sort by layer?
                    tdict['rects'][key] = dict()
                    tdict['rects'][key]['xy']    = rect.bbox.tolist()
                    tdict['rects'][key]['layer'] = rect.layer
                elif "VIA" in rect.layer[0]: # via instance. This condition works when only VIAx layer (not auto via) is in the template via instance. 
                    tdict['rects'] = dict() 
                    tdict['rects'][key] = dict()
                    tdict['rects'][key]['xy']    = rect.bbox.tolist()
                    tdict['rects'][key]['layer'] = rect.layer
                else:
                    tdict['rects'] = dict() # just create blank dictionary if the flag is false.
            tdict['pins'] = dict() # initialize the pin dictionary
            for pin in cell.pins.values():
                tdict['pins'][pin.name] = dict()
                tdict['pins'][pin.name]['netname'] = pin.netname
                tdict['pins'][pin.name]['layer']   = pin.layer.tolist()
                tdict['pins'][pin.name]['xy']      = pin.bbox.tolist()
            if not tdict.get("xy"):
                tdict['xy'] = [[0,0],[0,0]]
            templates[template_libname][cellname] = tdict

    return templates

def import_cmap(filename="./display.drf", layer_frontend=["prBoundary", "NW", "PP", "NP", "OD", "PO"], layer_stack=10, starting_idx=1):
    """
    Import colormap and construct a colarmap dictionary.

    Parameters
    ----------
    filename : str, optional
        The name of colarmap file.
    layer_frontend : list, optional
        The list of frontend layers.
    layer_stack : int, optional
        The number of metal stack.
    starting_idx : int, optional
        The number of starting index of metal stack.
    
    """

    print(f"LAYGO2 is now importing colormap into yaml file.")
    # colormap for mpl
    cdict_layer = {}
    cdict_primary = {}
    lines = []

    layer = []
    layer += ["__instance__", "__instance_pin__"]
    layer += layer_frontend
    for idx in range(starting_idx, layer_stack+starting_idx):
        layer.append(f"M{idx}")
        if idx != layer_stack+starting_idx-1:
            layer.append(f"VIA{idx}") # exclude the last stack via

    with open(filename, 'r') as file:
        for line in file:
            words = line.strip().split() 
            lines.append(words)
            for _layer in layer: # add frontend layers?
                if f"{_layer}_drawing" in words:
                    cdict_layer[_layer] = words[-3]
            if "PacketName" in words and "LineStyle" in words:
                    # TODO: get hatch for matplotlib.rectangle
                    pass
    
    for line in lines:
        for value in cdict_layer.values():
            if value in line and line.index(value) == 2:
                try:
                    cdict_primary[value] = [round(int(line[3])/255, 4), round(int(line[4])/255, 4) , round(int(line[5])/255, 4)]
                except ValueError:
                    pass
    
    cdict_layer_sorted = dict(sorted(cdict_layer.items(), key=lambda item:layer.index(item[0])))
    cdict_primary_sorted = dict(sorted(cdict_primary.items()))
    
    cdict = {}
    for _layer, strcolor in cdict_layer.items():
        cdict[_layer] = [cdict_primary_sorted[strcolor], cdict_primary_sorted[strcolor], 0.3] # [edgecolor, facecolor, alpha]
    cdict["__instance__"] = ["black", "none", 1]
    cdict["__instance_pin__"] = ["gray", "none", 1]
    cdict_sorted = dict(sorted(cdict.items(), key=lambda item:layer.index(item[0])))

    mpl_dict = dict()
    mpl_dict['colormap'] = cdict_sorted
    mpl_dict['order'] = layer
    return mpl_dict

# Import grids and (or) templates.
ext_input = dict()
# ext_input["M1"] = 100
# ext_input["M2"] = 80
# ext_input["M3"] = 70
# ext_input["M4"] = 90

grids             = import_grids(ext_input=ext_input)
templates         = import_templates()
tech_params       = dict()
tech_params['templates'] = templates
tech_params['grids']     = grids

# Parameters for mpl.
layer_frontend = ["prBoundary", "NW", "PP", "NP", "OD", "PO"]
layer_stack    = 9 
starting_idx   = 1 

# Import colormap.
if mpl_flag:
    mpl_dict = import_cmap(layer_stack=layer_stack, starting_idx=starting_idx)
    tech_params['mpl'] = mpl_dict

print(f"{len(templates[template_libname])} template instances and {len(grids[grid_libname])} grids are imported from the library.")
    
# yaml.Dumper.ignore_aliases = lambda *args : True

# Dump the tech_params into yaml file.
with open(f"./laygo2_tech/{output_name}.yaml", 'w') as stream:
    try:
        yaml.dump(tech_params, stream, default_flow_style=None, indent=4)
        # yaml.dump(tech_params, stream)
    except yaml.YAMLError as exc:
        print(exc)
# TODO: Can we just update the yaml not dumping? Maybe this takes longer since the former yaml should be loaded.
