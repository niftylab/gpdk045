#!/usr/bin/python
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

import laygo2.object.template
import laygo2.object.physical
import laygo2.object.database
#from . import laygo2_tech_grids

import numpy as np
import yaml
import pprint
import copy

# Technology parameters
tech_fname = './laygo2_tech/laygo2_tech.yaml'
with open(tech_fname, 'r') as stream:
    try:
        tech_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
libname = list(tech_params['templates'].keys())[0]  # libname
templates = tech_params['templates'][libname]
grids = tech_params['grids'][libname]

# Flexible templates
# Template functions for primitive devices
def _mos_flex_update_params(params):
    """Make a complete parameter table for mos"""
    if 'nf' not in params:  # number of fingers
        params['nf'] = 1
    if 'nfdmyl' not in params:  # number of left-dummy fingers
        params['nfdmyl'] = 0
    if 'nfdmyr' not in params:  # number of right-dummy fingers
        params['nfdmyr'] = 0
    if 'trackswap' not in params:  # source-drain track swap
        params['trackswap'] = False
    if 'tie' not in params:  # tie to power rail
        params['tie'] = None
    if 'bndl' not in params:  # create local left boundary
        params['bndl'] = True
    if 'bndr' not in params:  # create local right boundary
        params['bndr'] = True
    if 'gbndl' not in params:  # create global left boundary
        params['gbndl'] = False
    if 'gbndr' not in params:  # create global right boundary
        params['gbndr'] = False
    if 'unit_size_core' not in params:  # core unit size
        params['unit_size_core'] = np.array(templates['nmos4_fast_center_nf2']['unit_size'])
    if 'unit_size_dmy' not in params:  # dummy size
        params['unit_size_dmy'] = np.array(templates['nmos4_fast_dmy_nf2']['unit_size'])
    if 'unit_size_bndl' not in params:  # left boundary unit size
        params['unit_size_bndl'] = np.array(templates['nmos4_fast_boundary']['unit_size'])
    if 'unit_size_bndr' not in params:  # right boundary unit size
        params['unit_size_bndr'] = np.array(templates['nmos4_fast_boundary']['unit_size'])
    if 'unit_size_gbndl' not in params:  # left boundary unit size
        params['unit_size_gbndl'] = np.array(templates['nmos4_fast_left']['unit_size'])
    if 'unit_size_gbndr' not in params:  # right boundary unit size
        params['unit_size_gbndr'] = np.array(templates['nmos4_fast_right']['unit_size'])
    return params

def _mos_flex_route(devtype, params, offset=[0, 0]):
    """internal function to create routing structure of mosfets"""
    params = _mos_flex_update_params(params)

    # Routing offsets
    yoffset = offset[1]
    offset = np.array([0, yoffset])
    offset_rail = np.array([0, 0])
    offset_dmyl = np.array([0, yoffset])
    offset_dmyr = np.array([0, yoffset])
    if params['gbndl']:
        offset[0] += params['unit_size_gbndl'][0]
    offset_rail[0] = offset[0]
    if params['bndl']:
        offset[0] += params['unit_size_bndl'][0]
    offset_dmyl[0] = offset[0]
    offset[0] += params['unit_size_dmy'][0] * round(params['nfdmyl']/2)
    offset_dmyr[0] = offset[0] + params['unit_size_core'][0] * round(params['nf']/2)
    nelements = dict()
    # Basic terminals
    if devtype == 'nmos' or devtype == 'pmos':
        ref_temp_name = 'nmos4_fast_center_nf2'  # template for param calculations
        ref_dmy_temp_name = 'nmos4_fast_center_nf2'  # dummy template for param calculations
        ref_pin_name = 'S0'  # left-most pin for parameter calculations
        name_list = ['G', 'S', 'D']  
        if params['trackswap']:  # source-drain track swap
            yidx_list = [3, 2, 1]  # y-track list
        else:
            yidx_list = [3, 1, 2]
        pin_name_list = ['G0', 'S0', 'D0']  # pin nam list to connect
    elif devtype == 'ptap' or devtype == 'ntap':
        ref_temp_name = 'ptap_fast_center_nf2_v2'
        ref_dmy_temp_name = 'ptap_fast_center_nf2_v2'
        ref_pin_name = 'TAP0'
        name_list = ['TAP0', 'TAP1']
        if params['trackswap']:  # source-drain track swap
            yidx_list = [2, 1]
        else:
            yidx_list = [1, 2]
        pin_name_list = ['TAP0', 'TAP1']
    for _name, _yidx, _pin_name in zip(name_list, yidx_list, pin_name_list):
        if params['tie'] == _name: 
            continue  # do not generate routing elements
        # compute routing cooridnates
        x0 = templates[ref_temp_name]['pins'][_pin_name]['xy'][0][0]
        x1 = templates[ref_temp_name]['pins'][_pin_name]['xy'][1][0]
        x = round((x0 + x1)/2) + offset[0] # center coordinate 
        x0 = x
        x1 = x + params['unit_size_core'][0] * round(params['nf']/2-1)
        if _pin_name == ref_pin_name:  # extend route to S1 
            x1 += params['unit_size_core'][0]
        
        y = grids['routing_12_cmos']['horizontal']['elements'][_yidx] + offset[1]
        vextension = round(grids['routing_12_cmos']['horizontal']['width'][_yidx]/2)
        if x0 == x1:  # zero-size wire
            hextension = grids['routing_12_cmos']['vertical']['extension0'][0] 
            if _pin_name == 'G0' and params['nf'] == 2: # extend G route when nf is 2 to avoid DRC errror
                hextension = hextension + 55
            elif _pin_name == 'D0' and params['nf'] == 2 and params['tie'] != 'D': # extend D route when nf is 2 and not tied with D
                hextension = hextension + 55
            elif _pin_name == 'S0' and params['nf'] == 2 and params['tie'] != 'S': # extend S route when nf is 2 and not tied with S
                hextension = hextension + 55 
        else:
            hextension = grids['routing_12_cmos']['vertical']['extension'][0] + 25
        rxy = [[x0, y], [x1, y]]
        rlayer=grids['routing_12_cmos']['horizontal']['layer'][_yidx]
        # metal routing
        color = grids['routing_12_cmos']['horizontal']['ycolor'][_yidx]
        rg = laygo2.object.Rect(xy=rxy, layer=rlayer, name='R' + _name + '0', 
                                hextension=hextension, vextension=vextension, color=color)
        nelements['R' + _name + '0'] = rg
        # via
        vname = grids['routing_12_cmos']['via']['map'][0][_yidx]
        idx = round(params['nf']/2)
        if _pin_name == ref_pin_name:  # extend route to S1 
            idx += 1
        ivia = laygo2.object.Instance(name='IV' + _name + '0', xy=[x, y], libname=libname, cellname=vname, 
                                      shape=[idx, 1], pitch=params['unit_size_core'], 
                                      unit_size=params['unit_size_core'], pins=None, transform='R0')
        nelements['IV'+_name + '0'] = ivia
    # Horizontal rail
    x0 = templates[ref_temp_name]['pins'][ref_pin_name]['xy'][0][0]
    x1 = templates[ref_temp_name]['pins'][ref_pin_name]['xy'][1][0]
    x = round((x0 + x1)/2) + offset_rail[0] # center coordinate 
    x0 = x
    x1 = x + params['unit_size_core'][0] * round(params['nf']/2)
    x1 += params['unit_size_dmy'][0] * round(params['nfdmyl']/2)
    x1 += params['unit_size_dmy'][0] * round(params['nfdmyr']/2)
    if params['bndl']:
        x1 += params['unit_size_bndl'][0]
    if params['bndr']:
        x1 += params['unit_size_bndr'][0]
    y=grids['routing_12_cmos']['horizontal']['elements'][0] + offset_rail[1]
    vextension = round(grids['routing_12_cmos']['horizontal']['width'][0]/2)
    hextension = grids['routing_12_cmos']['vertical']['extension'][0] 
    rxy = [[x0, y], [x1, y]]
    rlayer=grids['routing_12_cmos']['horizontal']['layer'][0]
    color = grids['routing_12_cmos']['horizontal']['ycolor'][0]
    # metal routing
    rg = laygo2.object.Rect(xy=rxy, layer=rlayer, name='RRAIL0', 
                            hextension=hextension, vextension=vextension, color=color)
    nelements['RRAIL0'] = rg
    # Tie to rail
    if params['tie'] is not None:
        # routing
        if params['tie'] == 'D':
            idx = round(params['nf']/2)
            _pin_name = 'D0'
        if params['tie'] == 'S':
            idx = round(params['nf']/2) + 1
            _pin_name = 'S0'
        if params['tie'] == 'TAP0':
            idx = round(params['nf']/2) + 1
            _pin_name = 'TAP0'
        if params['tie'] == 'TAP1':
            idx = round(params['nf']/2)
            _pin_name = 'TAP1'
        x0 = templates[ref_temp_name]['pins'][_pin_name]['xy'][0][0]
        x1 = templates[ref_temp_name]['pins'][_pin_name]['xy'][1][0]
        x = round((x0 + x1)/2) + offset[0]  # center coordinate 
        _x = x
        for i in range(idx):
            hextension = round(grids['routing_12_cmos']['vertical']['width'][0]/2)
            vextension = grids['routing_12_cmos']['horizontal']['extension'][0] 
            y0=grids['routing_12_cmos']['horizontal']['elements'][0] + offset_rail[1]
            y1=grids['routing_12_cmos']['horizontal']['elements'][1] + offset[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer=grids['routing_12_cmos']['vertical']['layer'][0]
            color = grids['routing_12_cmos']['vertical']['xcolor'][0] 
            rg = laygo2.object.Rect(xy=rxy, layer=rlayer, name='RTIE' + str(i), 
                                    hextension=hextension, vextension=vextension, color=color)
            nelements['RTIE' + str(i)] = rg
            _x += params['unit_size_core'][0]
        # via
        vname = grids['routing_12_cmos']['via']['map'][0][0]
        ivia = laygo2.object.Instance(name='IVTIE0', xy=[x, y0], libname=libname, cellname=vname, 
                                      shape=[idx, 1], pitch=params['unit_size_core'], 
                                      unit_size=[0, 0], pins=None, transform='R0')
        nelements['IVTIE'+_name + '0'] = ivia
    # Tie to rail - dummy left
    if params['nfdmyl'] > 0:  
        if devtype == 'nmos' or devtype == 'pmos':
            if params['bndl']:  # terminated by boundary
                pin_name = 'S0'
                idx_offset = 0
            else:
                pin_name = 'G0'
                idx_offset = -1
        elif devtype == 'ptap' or devtype == 'ntap':
            if params['bndl']:  # terminated by boundary
                pin_name = 'TAP0'
                idx_offset = 0
            else:
                pin_name = 'TAP1'
                idx_offset = -1
        x0 = templates[ref_dmy_temp_name]['pins'][pin_name]['xy'][0][0]
        x1 = templates[ref_dmy_temp_name]['pins'][pin_name]['xy'][1][0]
        x = round((x0 + x1)/2) + offset_dmyl[0]  # center coordinate 
        _x = x
        idx = round(params['nfdmyl']) + idx_offset
        for i in range(idx):
            hextension = round(grids['routing_12_cmos']['vertical']['width'][0]/2)
            vextension = grids['routing_12_cmos']['horizontal']['extension'][0] 
            y0=grids['routing_12_cmos']['horizontal']['elements'][0] + offset_rail[1]
            y1=grids['routing_12_cmos']['horizontal']['elements'][1] + offset_dmyl[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer=grids['routing_12_cmos']['vertical']['layer'][0]
            color = grids['routing_12_cmos']['vertical']['xcolor'][0]
            rg = laygo2.object.Rect(xy=rxy, layer=rlayer, name='RTIEDMYL' + str(i), 
                                    hextension=hextension, vextension=vextension, color=color)
            nelements['RTIEDMYL' + str(i)] = rg
            _x = _x + round(params['unit_size_dmy'][0]/2)
        # via
        vname = grids['routing_12_cmos']['via']['map'][0][0]
        ivia = laygo2.object.Instance(name='IVTIEDMYL0', xy=[x, y0], libname=libname, cellname=vname, 
                                      shape=[idx, 1], pitch=params['unit_size_dmy']*np.array([0.5, 1]), 
                                      unit_size=[0, 0], pins=None, transform='R0')
        nelements['IVTIEDMYL'+_name + '0'] = ivia
    # Tie to rail - dummy right
    if params['nfdmyr'] > 0:
        if devtype == 'nmos' or devtype == 'pmos':
            if params['bndr']:  # terminated by boundary
                pin_name = 'G0'  
                idx_offset = 0
            else:
                pin_name = 'G0'
                idx_offset = -1
        elif devtype == 'ptap' or devtype == 'ntap':
            if params['bndr']:  # terminated by boundary
                pin_name = 'TAP1'
                idx_offset = 0
            else:
                pin_name = 'TAP1'
                idx_offset = -1
        x0 = templates[ref_dmy_temp_name]['pins'][pin_name]['xy'][0][0]
        x1 = templates[ref_dmy_temp_name]['pins'][pin_name]['xy'][1][0]
        x = round((x0 + x1)/2) + offset_dmyr[0]  # center coordinate 
        _x = x
        idx = round(params['nfdmyr']) + idx_offset
        for i in range(idx):
            hextension = round(grids['routing_12_cmos']['vertical']['width'][0]/2)
            vextension = grids['routing_12_cmos']['horizontal']['extension'][0] 
            y0=grids['routing_12_cmos']['horizontal']['elements'][0] + offset_rail[1]
            y1=grids['routing_12_cmos']['horizontal']['elements'][1] + offset_dmyr[1]
            rxy = [[_x, y0], [_x, y1]]
            rlayer=grids['routing_12_cmos']['vertical']['layer'][0]
            color = grids['routing_12_cmos']['vertical']['xcolor'][0]
            rg = laygo2.object.Rect(xy=rxy, layer=rlayer, name='RTIEDMYR' + str(i), hextension=hextension, vextension=vextension, color=color)
            nelements['RTIEDMYR' + str(i)] = rg
            _x = _x + round(params['unit_size_dmy'][0]/2)
        # via
        vname = grids['routing_12_cmos']['via']['map'][0][0]
        ivia = laygo2.object.Instance(name='IVTIEDMYR0', xy=[x, y0], libname=libname, cellname=vname, 
                                      shape=[idx, 1], pitch=params['unit_size_dmy']*np.array([0.5, 1]), 
                                      unit_size=[0, 0], pins=None, transform='R0')
        nelements['IVTIEDMYR'+_name + '0'] = ivia
    return nelements


def mos_flex_bbox_func(params):
    """Computes x and y coordinate values from params."""
    params = _mos_flex_update_params(params)
    if 'w' in params:  # finger width
        nfin = np.ceil(params['w']/100)
    else:
        nfin = 4
    xy = np.array([[0, 0], [410, 1000]])
    xy[1, 0] = xy[0, 0] + params['unit_size_core'][0] * params['nf']/2
    xy[1, 1] = xy[0, 1] + 600 + 100*nfin  #adjust based on the number of fins
    if params['gbndl']:
        xy[1, 0] += params['unit_size_gbndl'][0]
    if params['bndl']:
        xy[1, 0] += params['unit_size_bndl'][0]
    if params['nfdmyl'] > 0:
        xy[1, 0] += params['unit_size_dmy'][0] * round(params['nfdmyl']/2)
    if params['nfdmyr'] > 0:
        xy[1, 0] += params['unit_size_dmy'][0] * round(params['nfdmyr']/2)
    if params['bndr']:
        xy[1, 0] += params['unit_size_bndr'][0]
    if params['gbndr']:
        xy[1, 0] += params['unit_size_gbndr'][0]
    return xy


def mos_flex_pins_func(devtype, params):
    """Generate a pin dictionary from params."""
    params = _mos_flex_update_params(params)
    if 'w' in params:  # finger width
        nfin = np.ceil(params['w']/100)
    else:
        nfin = 4

    pins = dict()
    # generate a virtual routing structure for reference
    route_obj = _mos_flex_route(devtype=devtype, params=params, offset=[0, 100*(nfin-4)])
    #print(route_obj['RG0'].xy)
    if 'RG0' in route_obj:  # gate
        g_obj = route_obj['RG0']
        pins['G'] = laygo2.object.Pin(xy=g_obj.xy, layer=g_obj.layer, netname='G')
    if 'RD0' in route_obj:  # drain
        d_obj = route_obj['RD0']
        pins['D'] = laygo2.object.Pin(xy=d_obj.xy, layer=d_obj.layer, netname='D')
    if 'RS0' in route_obj:  # source
        s_obj = route_obj['RS0']
        pins['S'] = laygo2.object.Pin(xy=s_obj.xy, layer=s_obj.layer, netname='S')
    if 'RRAIL0' in route_obj:  # rail
        r_obj = route_obj['RRAIL0']
        pins['RAIL'] = laygo2.object.Pin(xy=r_obj.xy, layer=r_obj.layer, netname='RAIL')
    return pins


def mos_flex_generate_func(devtype, name=None, shape=None, pitch=None, transform='R0', params=None):
    """Generates a flexible mos instance from the input parameters."""
    # Compute parameters
    params = _mos_flex_update_params(params)
    if 'w' in params:  # finger width
        nfin = np.ceil(params['w']/100)
        w = params['w']
    else:
        nfin = 4
        w = 360
    
    # Bbox
    inst_xy = mos_flex_bbox_func(params=params)
    inst_unit_size = [inst_xy[1, 0] - inst_xy[0, 0], inst_xy[1, 1] - inst_xy[0, 1]]
    
    # Create the base mosfet structure.
    nelements = dict()
    # Implant - the gpdk045 tech does not have implant layers. Skip.
    if devtype == 'nmos_flex':
        rimp = laygo2.object.Rect(xy=inst_xy, layer=['Nimp', 'drawing'], name='RIM0_FLEX_IMP') 
        nelements['RIM0_FLEX_IMP'] = rimp
    elif devtype == 'pmos_flex':
        rimp = laygo2.object.Rect(xy=inst_xy, layer=['Pimp', 'drawing'], name='RIM0_FLEX_IMP') 
        nelements['RIM0_FLEX_IMP'] = rimp
    # Well
    if devtype == 'pmos_flex':
        rnw = laygo2.object.Rect(xy=inst_xy, layer=['Nwell', 'drawing'], name='RIM0_FLEX_NW') 
        nelements['RIM0_FLEX_NW'] = rnw
    
    cursor = [0, 0]
    # Left global boundary
    if params['gbndl']:
        cursor[0] += params['unit_size_gbndl'][0]
    # Left local boundary
    if params['bndl']:
        cursor[0] += params['unit_size_bndl'][0]
    # Left dummy
    if params['nfdmyl'] > 0:
        if devtype == 'nmos_flex':
            cellname = 'nmos1v_lvt'
        elif devtype == 'pmos_flex':
            cellname = 'pmos1v_lvt'
        idmyl_pcell_params = {
            "fw":w*1e-9,
            "fingers":int(params['nfdmyl']),
        }
        idmyl_xy_offset = np.array([80, 235])  # offset between core position and pcell origin.
        idmyl_flex = laygo2.object.Instance(name='IM0', xy=cursor+idmyl_xy_offset, libname='gpdk045',
                                            cellname=cellname, shape=[1, 1], 
                                            pitch=params['unit_size_dmy'], 
                                            unit_size=params['unit_size_dmy'], 
                                            #params = {"pcell_params":idmyl_pcell_params},
                                            params = idmyl_pcell_params,
                                            pins=None, transform='R0')
        nelements['IDMYL0_FLEX'] = idmyl_flex
        # Left dummy - Gate structure
        # Gate poly
        for i in range(round(params['nfdmyl'])):
            rg_xy = [[cursor[0] + 80+round(params['unit_size_dmy'][0]/2)*i, 135],\
                     [cursor[0] + 125+round(params['unit_size_dmy'][0]/2)*i, inst_unit_size[1]-150]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                    name='RIDMYL0_FLEX_G_PO0_'+str(i)) 
            nelements['RIDMYL0_FLEX_G_PO0_'+str(i)] = rg
        for i in range(round(params['nfdmyl']/2)):
            rg_xy = [[cursor[0] + 80+params['unit_size_dmy'][0]*i, inst_unit_size[1]-270],\
                     [cursor[0] + 330+params['unit_size_dmy'][0]*i, inst_unit_size[1]-150]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                    name='RIDMYL0_FLEX_G_PO1_'+str(i)) 
            nelements['RIDMYL0_FLEX_G_PO1_'+str(i)] = rg
        # Gate metal/contact
        for i in range(round(params['nfdmyl']/2)):
            rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-275],\
       	             [cursor[0] + 295+params['unit_size_dmy'][0]*i, inst_unit_size[1]-145]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Metal1', 'drawing'], name='RIDMYL0_FLEX_G_M1_0_'+str(i)) 
            nelements['RIDMYL0_FLEX_G_M1_0_'+str(i)] = rg
            rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
       	             [cursor[0] + 175+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIDMYL0_FLEX_G_V0_0_'+str(i)) 
            nelements['RIDMYL0_FLEX_G_V0_0_'+str(i)] = rg
            rg_xy = [[cursor[0] + 240+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
       	             [cursor[0] + 300+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIDMYL0_FLEX_G_V0_1_'+str(i)) 
            nelements['RIDMYL0_FLEX_G_V0_1_'+str(i)] = rg
        cursor[0] += params['unit_size_dmy'][0] * params['nfdmyl']/2
    # Core mosfet - Cell name
    if devtype == 'nmos_flex':
        cellname = 'nmos1v_lvt'
    elif devtype == 'pmos_flex':
        cellname = 'pmos1v_lvt'
    icore_pcell_params = {
        "fw":w*1e-9,
        "fingers":int(params['nf']),
    }
    # Core mosfet - core pcell
    icore_xy_offset = np.array([80, 235])  # offset between core position and pcell origin.
    icore_flex = laygo2.object.Instance(name='IM0', xy=cursor+icore_xy_offset, libname='gpdk045',
                                   cellname=cellname, shape=[1, 1], 
                                   pitch=params['unit_size_core'], 
                                   unit_size=params['unit_size_core'], 
                                   #params = {"pcell_params":icore_pcell_params},
                                   params = icore_pcell_params,
                                   pins=None, transform='R0')
    nelements['IM0_FLEX'] = icore_flex
    # Core mosfet - Gate structure
    # Gate poly
    for i in range(round(params['nf'])):
        rg_xy = [[cursor[0] + 80+round(params['unit_size_core'][0]/2)*i, 135],\
                 [cursor[0] + 125+round(params['unit_size_core'][0]/2)*i, inst_unit_size[1]-150]]
        rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                name='RIM0_FLEX_G_PO0_'+str(i)) 
        nelements['RIM0_FLEX_G_PO0_'+str(i)] = rg
    for i in range(round(params['nf']/2)):
        rg_xy = [[cursor[0] + 80+params['unit_size_core'][0]*i, inst_unit_size[1]-270],\
                 [cursor[0] + 330+params['unit_size_core'][0]*i, inst_unit_size[1]-150]]
        rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                name='RIM0_FLEX_G_PO1_'+str(i)) 
        nelements['RIM0_FLEX_G_PO1_'+str(i)] = rg
    # Gate metal
    for i in range(round(params['nf']/2)):
        rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-275],\
                 [cursor[0] + 295+params['unit_size_dmy'][0]*i, inst_unit_size[1]-145]]
        rg = laygo2.object.Rect(xy=rg_xy, layer=['Metal1', 'drawing'], name='RIM0_FLEX_G_M1_0_'+str(i)) 
        nelements['RIM0_FLEX_G_M1_0_'+str(i)] = rg
        rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
                 [cursor[0] + 175+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
        rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIM0_FLEX_G_V0_0_'+str(i)) 
        nelements['RIM0_FLEX_G_V0_0_'+str(i)] = rg
        rg_xy = [[cursor[0] + 240+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
                 [cursor[0] + 300+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
        rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIM0_FLEX_G_V0_1_'+str(i)) 
        nelements['RIM0_FLEX_G_V0_1_'+str(i)] = rg
    cursor[0] += params['unit_size_core'][0] * params['nf']/2

    # Right dummy
    if params['nfdmyr'] > 0:
        if devtype == 'nmos_flex':
            cellname = 'nmos1v_lvt'
        elif devtype == 'pmos_flex':
            cellname = 'pmos1v_lvt'
        idmyr_pcell_params = {
            "fw":w*1e-9,
            "fingers":int(params['nfdmyl']),
        }
        idmyr_xy_offset = np.array([80, 235])  # offset between core position and pcell origin.
        idmyr_flex = laygo2.object.Instance(name='IM0', xy=cursor+idmyr_xy_offset, libname='gpdk045',
                                            cellname=cellname, shape=[1, 1], 
                                            pitch=params['unit_size_dmy'], 
                                            unit_size=params['unit_size_dmy'], 
                                            params = idmyr_pcell_params,
                                            pins=None, transform='R0')
        nelements['IDMYR0_FLEX'] = idmyr_flex
        # Right dummy - Gate structure
        # Gate poly
        for i in range(round(params['nfdmyr'])):
            rg_xy = [[cursor[0] + 80+round(params['unit_size_dmy'][0]/2)*i, 135],\
                     [cursor[0] + 125+round(params['unit_size_dmy'][0]/2)*i, inst_unit_size[1]-150]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                    name='RIDMYR0_FLEX_G_PO0_'+str(i)) 
            nelements['RIDMYR0_FLEX_G_PO0_'+str(i)] = rg
        for i in range(round(params['nfdmyr']/2)):
            rg_xy = [[cursor[0] + 80+params['unit_size_dmy'][0]*i, inst_unit_size[1]-270],\
                     [cursor[0] + 330+params['unit_size_dmy'][0]*i, inst_unit_size[1]-150]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Poly', 'drawing'], 
                                    name='RIDMYR0_FLEX_G_PO1_'+str(i)) 
            nelements['RIDMYR0_FLEX_G_PO1_'+str(i)] = rg
        # Gate metal/contact
        for i in range(round(params['nfdmyr']/2)):
            rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-275],\
       	             [cursor[0] + 295+params['unit_size_dmy'][0]*i, inst_unit_size[1]-145]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Metal1', 'drawing'], name='RIDMYR0_FLEX_G_M1_0_'+str(i)) 
            nelements['RIDMYR0_FLEX_G_M1_0_'+str(i)] = rg
            rg_xy = [[cursor[0] + 115+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
       	             [cursor[0] + 175+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIDMYR0_FLEX_G_V0_0_'+str(i)) 
            nelements['RIDMYR0_FLEX_G_V0_0_'+str(i)] = rg
            rg_xy = [[cursor[0] + 240+params['unit_size_dmy'][0]*i, inst_unit_size[1]-240],\
       	             [cursor[0] + 300+params['unit_size_dmy'][0]*i, inst_unit_size[1]-180]]
            rg = laygo2.object.Rect(xy=rg_xy, layer=['Cont', 'drawing'], name='RIDMYR0_FLEX_G_V0_1_'+str(i)) 
            nelements['RIDMYR0_FLEX_G_V0_1_'+str(i)] = rg
        cursor[0] += params['unit_size_dmy'][0] * params['nfdmyr']/2
    # Right local boundary
    if params['bndr']:
        cursor[0] += params['unit_size_bndr'][0]
    # Right global boundary
    if params['gbndr']:
        cursor[0] += params['unit_size_gbndr'][0]
     
    # Routing
    if devtype == 'nmos_flex':
        devtype = 'nmos'
    if devtype == 'pmos_flex':
        devtype = 'pmos'
    nelements.update(_mos_flex_route(devtype=devtype, params=params, offset=[0, 100*(nfin-4)]))
    nelements.update(_mos_flex_route(devtype=devtype, params=params, offset=[0, 100*(nfin-4)]))

    # Create pins
    pins = mos_flex_pins_func(devtype=devtype, params=params)
    #nelements.update(pins)  # Add physical pin structures to the virtual object.

    # Unit size
    inst_unit_size = [inst_xy[1, 0] - inst_xy[0, 0], inst_xy[1, 1] - inst_xy[0, 1]]
    
    # Pitch
    if pitch is None:
        pitch = inst_unit_size
    
    # Generate and return the final instance
    inst = laygo2.object.VirtualInstance(name=name, xy=np.array([0, 0]), libname=libname, cellname='myvcell_'+devtype,
                                         native_elements=nelements, shape=shape, pitch=pitch,
                                         transform=transform, unit_size=inst_unit_size, pins=pins)
    return inst

def nmos_flex_generate_func(name=None, shape=None, pitch=None, transform='R0', params=None):
    return mos_flex_generate_func(devtype='nmos_flex', name=name, shape=shape, pitch=pitch, transform=transform, params=params)

def pmos_flex_generate_func(name=None, shape=None, pitch=None, transform='R0', params=None):
    return mos_flex_generate_func(devtype='pmos_flex', name=name, shape=shape, pitch=pitch, transform=transform, params=params)

# Create template library
def load_flex_templates():
    """Load flexible templates to a template library object"""
    tlib = laygo2.object.database.TemplateLibrary(name=libname)
    # Flexible transistors (transistors with variable finger widths)
    tnmos_flex = laygo2.object.template.UserDefinedTemplate(name='nmos_flex', bbox_func=mos_flex_bbox_func, 
                                   pins_func=mos_flex_pins_func, generate_func=nmos_flex_generate_func)
    tpmos_flex = laygo2.object.template.UserDefinedTemplate(name='pmos_flex', bbox_func=mos_flex_bbox_func, 
                                   pins_func=mos_flex_pins_func, generate_func=pmos_flex_generate_func)
    tlib.append([tnmos_flex, tpmos_flex])
    return tlib


